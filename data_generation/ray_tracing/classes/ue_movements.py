import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# ----------------------------------------------------------------------------
# Utility: handle pavement & environment & building collisions
# ----------------------------------------------------------------------------
def handle_collisions(
    prev_pos: np.ndarray,
    new_pos: np.ndarray,
    vel: np.ndarray,
    dims: Optional[np.ndarray],
    state: Dict[str, Any],
    move_type: str,
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    - Prevent UE from moving beyond pavement or environment.
    - On pavement edges, redirect movement along the edge (90° turn), not reverse.
    - Stop at buildings: revert to prev_pos and zero velocity.
    """
    adj_pos = new_pos.copy()
    adj_vel = vel.copy()

    # 1) Building collision: stop inside building
    if buildings:
        for b in buildings:
            if (b['xmin'] <= adj_pos[0] <= b['xmax'] and
                b['ymin'] <= adj_pos[1] <= b['ymax']):
                
                if move_type == 'circular':
                    # Revert to previous position
                    adj_pos = prev_pos.copy()
                    # Flip the direction of circular motion
                    if 'effective_clockwise' in state:
                         state['effective_clockwise'] = not state['effective_clockwise']
                    # Stop for this step, reversal will occur in the next
                    adj_vel = np.zeros_like(vel)
                    # No need to check other buildings for this collision type if reversing
                    break 
                else:
                    # --- Original Building collision: turn 90 degrees ---
                    adj_pos = prev_pos.copy() # Revert to previous position

                    # Determine which boundary of the building was hit
                    # Simplified: find the smallest overlap to guess the wall hit
                    dx_min = adj_pos[0] - b['xmin']
                    dx_max = b['xmax'] - adj_pos[0]
                    dy_min = adj_pos[1] - b['ymin']
                    dy_max = b['ymax'] - adj_pos[1]

                    overlaps = {
                        'left': dx_min,
                        'right': dx_max,
                        'bottom': dy_min,
                        'top': dy_max
                    }
                    
                    # Ensure we are considering the position *before* it went into the building
                    # by using prev_pos for overlap calculation relative to building edges
                    # This logic might need refinement for edge cases or very fast UEs.
                    # For now, let's assume prev_pos is just outside.

                    # If prev_pos was to the left of b['xmin'] and new_pos is inside
                    hit_left = prev_pos[0] < b['xmin'] and new_pos[0] >= b['xmin']
                    # If prev_pos was to the right of b['xmax'] and new_pos is inside
                    hit_right = prev_pos[0] > b['xmax'] and new_pos[0] <= b['xmax']
                    # If prev_pos was below b['ymin'] and new_pos is inside
                    hit_bottom = prev_pos[1] < b['ymin'] and new_pos[1] >= b['ymin']
                    # If prev_pos was above b['ymax'] and new_pos is inside
                    hit_top = prev_pos[1] > b['ymax'] and new_pos[1] <= b['ymax']

                    # Current velocity direction
                    # vel_angle = np.arctan2(vel[1], vel[0])
                    # new_vel_angle = vel_angle

                    if hit_left or hit_right: # Hit a vertical wall (xmin or xmax)
                        # Place UE just outside the wall it hit
                        adj_pos[0] = b['xmin'] - 1e-3 if hit_left else b['xmax'] + 1e-3
                        # Turn 90 degrees: if moving along x, now move along y
                        adj_vel[0] = 0
                        adj_vel[1] = np.sign(vel[1]) * speed if np.abs(vel[1]) > 1e-6 else (speed if prev_pos[1] < (b['ymin'] + b['ymax'])/2 else -speed) # try to move away from center
                        if adj_vel[1] == 0 : adj_vel[1] = speed # default if centered
                    elif hit_bottom or hit_top: # Hit a horizontal wall (ymin or ymax)
                        # Place UE just outside the wall it hit
                        adj_pos[1] = b['ymin'] - 1e-3 if hit_bottom else b['ymax'] + 1e-3
                        # Turn 90 degrees: if moving along y, now move along x
                        adj_vel[1] = 0
                        adj_vel[0] = np.sign(vel[0]) * speed if np.abs(vel[0]) > 1e-6 else (speed if prev_pos[0] < (b['xmin'] + b['xmax'])/2 else -speed)
                        if adj_vel[0] == 0 : adj_vel[0] = speed # default if centered
                    else: # If somehow still inside or complex collision, just stop
                        adj_pos = prev_pos.copy()
                        adj_vel = np.zeros_like(vel)

                    # Ensure speed is maintained if a turn happened
                    if np.linalg.norm(adj_vel) > 1e-6:
                        adj_vel = (adj_vel / np.linalg.norm(adj_vel)) * speed
                    break # Collision handled for this building

    # 2) Pavement edge: when hitting min or max, turn along the edge
    if pavement:
        for i, axis in enumerate(('x', 'y')):
            mn = pavement.get(f'{axis}_min', 0.0)
            mx = pavement.get(f'{axis}_max', dims[i] if dims is not None else np.inf)
            if adj_pos[i] <= mn or adj_pos[i] >= mx:
                # Determine perpendicular axis index
                perp = 1 - i
                # New direction is along perpendicular axis, preserve original speed
                speed = np.linalg.norm(vel)
                sign = np.sign(vel[perp]) or 1.0
                new_vel = np.zeros_like(adj_vel)
                new_vel[perp] = sign * speed
                # Clamp position to pavement edge
                adj_pos[i] = np.clip(adj_pos[i], mn, mx)
                adj_vel = new_vel
                return adj_pos, adj_vel, state

    # 3) Environment limits: stop or reverse
    if dims is not None:
        for i in (0, 1):
            if adj_pos[i] <= 0 or adj_pos[i] >= dims[i]:
                if move_type == 'linear':
                    adj_vel = np.zeros(3)
                else:
                    adj_vel *= -1
                adj_pos[i] = np.clip(adj_pos[i], 0, dims[i])

    return adj_pos, adj_vel, state

# ----------------------------------------------------------------------------
# Movement update functions
# ----------------------------------------------------------------------------

def update_linear(
    pos: np.ndarray, vel: np.ndarray, speed: float, dt: float,
    dims: Optional[np.ndarray], params: Dict[str, Any], state: Dict[str, Any],
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if np.linalg.norm(vel) < 1e-9:
        return pos, vel, state
    new_pos = pos + vel * dt
    return handle_collisions(pos, new_pos, vel, dims, state, 'linear', pavement, buildings)


def update_forward_back(
    pos: np.ndarray, vel: np.ndarray, speed: float, dt: float,
    dims: Optional[np.ndarray], params: Dict[str, Any], state: Dict[str, Any],
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dist = params.get('distance', 10.0)
    norm_vel = np.linalg.norm(vel)
    dir_vec = (vel / norm_vel) if norm_vel > 0 else np.zeros(3)
    step = speed * dt
    state['moved'] = state.get('moved', 0.0) + step

    # phase flip when reaching target distance
    if state['moved'] >= dist:
        overshoot = state['moved'] - dist
        step -= overshoot
        state['moved'] = 0.0
        dir_vec *= -1
        vel = -vel
        state['phase'] = 'back' if state.get('phase') == 'forward' else 'forward'

    new_pos = pos + dir_vec * step
    return handle_collisions(pos, new_pos, vel, dims, state, 'forward_back', pavement, buildings)


def update_circular(
    pos: np.ndarray, vel: np.ndarray, speed: float, dt: float,
    dims: Optional[np.ndarray], params: Dict[str, Any], state: Dict[str, Any],
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    center = state.get('center', pos)
    r = params.get('radius', 10.0)
    angle = state.get('angle', 0.0)
    omega = speed / r
    # Use effective_clockwise from state, which can be flipped by collision handler
    current_clockwise = state.get('effective_clockwise', params.get('clockwise', True))
    angle += (-omega if current_clockwise else omega) * dt
    angle %= 2 * np.pi

    new_pos = center + np.array([r * np.cos(angle), r * np.sin(angle), pos[2]])
    
    # Correctly calculate tangential velocity based on the current effective_clockwise direction
    d_angle_dt = (-omega if current_clockwise else omega)
    vel_vec = np.array([
        -r * np.sin(angle) * d_angle_dt,  # dx/dt = -r * sin(angle) * (d_angle/dt)
         r * np.cos(angle) * d_angle_dt,  # dy/dt =  r * cos(angle) * (d_angle/dt)
         0.0
    ])
    state['angle'] = angle

    return handle_collisions(pos, new_pos, vel_vec, dims, state, 'circular', pavement, buildings)

def update_zigzag(
    pos: np.ndarray, vel: np.ndarray, speed: float, dt: float,
    dims: Optional[np.ndarray], params: Dict[str, Any], state: Dict[str, Any],
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    leg_length = params.get('leg_length', 10.0)
    angle_rad = np.deg2rad(params.get('angle_deg', 30.0))
    num_legs = params.get('num_legs', 4) # Default to 4 legs for a couple of zigs and zags

    if 'current_leg' not in state: # Initialisation
        state['current_leg'] = 0
        state['distance_on_leg'] = 0.0
        main_dir_param = np.array(params.get('main_direction', [1,0,0]), dtype=float)
        main_dir_norm = np.linalg.norm(main_dir_param)
        if main_dir_norm < 1e-9: main_dir_param = np.array([1,0,0]) # Default if zero vector
        state['main_direction_vector'] = main_dir_param / (main_dir_norm if main_dir_norm > 1e-9 else 1.0)
        state['main_direction_vector'][2] = 0 # Ensure main direction is XY plane
        state['main_direction_vector'] /= (np.linalg.norm(state['main_direction_vector']) if np.linalg.norm(state['main_direction_vector']) > 1e-9 else 1.0)
        
        # Initial direction: main_direction rotated by +angle for the first leg
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                  0,                 1]
        ])
        state['current_direction_vector'] = np.dot(rotation_matrix, state['main_direction_vector'])[:2] # Work with 2D for rotation
        state['current_direction_vector'] = np.append(state['current_direction_vector'], 0.0) # Back to 3D

    if state['current_leg'] >= num_legs:
        # Zigzag finished, become static or could follow main_direction linearly
        return pos, np.zeros_like(vel), state # Stop after num_legs

    step_distance = speed * dt
    state['distance_on_leg'] += step_distance

    current_leg_direction_3d = np.array(state['current_direction_vector'], dtype=float)
    current_velocity = current_leg_direction_3d * speed
    new_pos = pos + current_velocity * dt

    if state['distance_on_leg'] >= leg_length:
        overshoot = state['distance_on_leg'] - leg_length
        new_pos -= current_leg_direction_3d * overshoot # Correct for overshoot
        state['current_leg'] += 1
        state['distance_on_leg'] = 0.0

        if state['current_leg'] < num_legs:
            # Alternate angle for next leg: if leg index is even, rotate by +angle, if odd, by -angle
            # This makes it go main_dir+angle, then main_dir-angle, etc.
            # For a classic zig-zag, it should be +angle, -angle, +angle, -angle relative to main_direction
            current_angle_offset = angle_rad if (state['current_leg'] % 2 == 0) else -angle_rad
            rotation_matrix = np.array([
                [np.cos(current_angle_offset), -np.sin(current_angle_offset), 0],
                [np.sin(current_angle_offset),  np.cos(current_angle_offset), 0],
                [0,                             0,                            1]
            ])
            next_leg_dir_2d = np.dot(rotation_matrix, state['main_direction_vector'])[:2]
            state['current_direction_vector'] = np.append(next_leg_dir_2d, 0.0) # Back to 3D
            current_velocity = state['current_direction_vector'] * speed # Update velocity for collision handler
        else:
            # Last leg finished
            current_velocity = np.zeros_like(vel)

    return handle_collisions(pos, new_pos, current_velocity, dims, state, 'zigzag', pavement, buildings)

# ----------------------------------------------------------------------------
# Registry & API
# ----------------------------------------------------------------------------
MOVEMENTS = {
    'linear': update_linear,
    'forward_back': update_forward_back,
    'circular': update_circular,
    'zigzag': update_zigzag
}

def update_movement(
    move_type: str,
    pos: np.ndarray,
    vel: np.ndarray,
    speed: float,
    dt: float,
    dims: Optional[np.ndarray],
    params: Dict[str, Any],
    state: Dict[str, Any],
    pavement: Optional[Dict[str, float]] = None,
    buildings: Optional[List[Dict[str, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    handler = MOVEMENTS.get(move_type)
    if not handler:
        raise ValueError(f"Unsupported movement type: {move_type}")
    return handler(pos, vel, speed, dt, dims, params, state, pavement, buildings)
