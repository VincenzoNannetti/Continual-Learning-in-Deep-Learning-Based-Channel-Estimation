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
                return prev_pos.copy(), np.zeros_like(adj_vel), state

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
    r = params.get('radius', 1.0)
    if r <= 1e-9 or speed < 1e-9:
        return pos, np.zeros(3), state
    center = np.array(state.get('center', [0.0, 0.0, pos[2]]))
    angle = state.get('angle', 0.0)
    omega = speed / r
    angle += (-omega if params.get('clockwise', True) else omega) * dt
    angle %= 2 * np.pi

    new_pos = center + np.array([r * np.cos(angle), r * np.sin(angle), pos[2]])
    vel_vec = np.array([-omega * r * np.sin(angle), omega * r * np.cos(angle), 0.0])
    state['angle'] = angle

    return handle_collisions(pos, new_pos, vel_vec, dims, state, 'circular', pavement, buildings)

# ----------------------------------------------------------------------------
# Registry & API
# ----------------------------------------------------------------------------
MOVEMENTS = {
    'linear': update_linear,
    'forward_back': update_forward_back,
    'circular': update_circular
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
