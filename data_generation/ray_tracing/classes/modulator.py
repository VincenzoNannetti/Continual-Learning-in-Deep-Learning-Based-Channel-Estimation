"""
Filename: ./data_generation/ray_tracing/classes/modulator.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Modulator Class for the ray tracing model.
"""

import numpy as np

class Modulator:
    def __init__(self, mod_type, no_symbols):
        # modulation parameters
        self.modulation_type    = mod_type     # type of modulation used
        self.num_symbols        = no_symbols   # M in M-QAM

        self.mapping            = self.generate_qam_mapping()
        self.reverse_mapping    = {v: k for k, v in self.mapping.items()}

    def modulate(self, input_data, num_symbols=None):
        if self.modulation_type == "QAM":
            # check if M in M-QAM is a power of 2 an non-zero
            if not (self.num_symbols > 0 and ((self.num_symbols & (self.num_symbols - 1)) == 0)):
                raise ValueError("M must be a power of 2 (e.g., 4, 16, 64).")

            # calculate bits per QAM symbol
            bits_per_qam_symbol = int(np.log2(self.num_symbols))

            # determine the number of bits needed
            if input_data == "rand":
                if num_symbols is None:
                    # Default behavior: Keep original random size if no num_symbols provided
                    # (You might want to define a specific default size here instead)
                    num_bits_needed = 17000 # Original hardcoded value
                    print(f"Warning: Modulator generating default {num_bits_needed} random bits.")
                else:
                    # Calculate bits needed for the requested number of QAM symbols
                    num_bits_needed = num_symbols * bits_per_qam_symbol

                temp_signal = np.random.randint(0, 2, size=num_bits_needed, dtype=np.uint8)
            else:
                # Convert text input to bits
                temp_signal = np.frombuffer(input_data.encode(), dtype=np.uint8)
                temp_signal = np.unpackbits(temp_signal)
                # Note: num_symbols argument is ignored for non-"rand" input

            # pad with zeros to make divisible by bits_per_qam_symbol
            if len(temp_signal) % bits_per_qam_symbol != 0:
                pad_size = bits_per_qam_symbol - (len(temp_signal) % bits_per_qam_symbol)
                temp_signal = np.pad(temp_signal, (0, pad_size), mode="constant", constant_values=0)

            # reshape to matrix where each row represents bits for one QAM symbol
            temp_signal = temp_signal.reshape(-1, bits_per_qam_symbol)

            # Map bits to complex symbols
            modulated_signal = []
            for symbol_bits in temp_signal:
                binary_str = ''.join(map(str, symbol_bits))
                modulated_signal.append(self.mapping[binary_str])

            # return it as a numpy array
            return np.array(modulated_signal)
        else:
            raise ValueError("Invalid Modulation Type")

    def demodulate(self, received_signal, pilot_indices):
        if self.modulation_type == "QAM":
            # Create an array of all constellation points (complex symbols)
            symbols = np.array(list(self.reverse_mapping.keys()), dtype=complex)
            demodulated_bits = []

            data_grid = np.delete(received_signal, pilot_indices, axis=0)

            # flatten the grid into a 1D array - 'F' for column wise
            received_data = data_grid.flatten('F')

            # iterate over each received data symbol
            for received in received_data:
                # compute Euclidean distances between the received symbol and every QAM constellation symbol
                distances = np.abs(received - symbols)
                closest_index = np.argmin(distances)
                closest_symbol = symbols[closest_index]

                # get the binary string corresponding to the closest constellation point
                binary_str = self.reverse_mapping[closest_symbol]
                demodulated_bits.append(binary_str)

            bit_sequence = np.array([int(bit) for bits in demodulated_bits for bit in bits], dtype=np.uint8)
            return bits_to_text(bit_sequence)
        else:
            raise ValueError("Invalid Modulation Type")

    # function to generate M-QAM mapping
    def generate_qam_mapping(self):
        # generate the constellation points (Gray coding)
        levels = np.arange(-np.sqrt(self.num_symbols) + 1, np.sqrt(self.num_symbols), 2)
        constellation = np.array([x + 1j * y for y in levels for x in levels])

        # normalise constellation to unit average power
        constellation /= np.sqrt((np.abs(constellation) ** 2).mean())

        # create binary-to-symbol mapping
        mapping = {}
        for i, point in enumerate(constellation):
            binary_symbol = format(i, f"0{int(np.log2(self.num_symbols))}b")
            mapping[binary_symbol] = point

        return mapping

def bits_to_text(bit_sequence):
    # Ensure length is a multiple of 8 (pad with zeros if necessary)
    pad_length = (8 - len(bit_sequence) % 8) % 8
    padded_bits = np.pad(bit_sequence, (0, pad_length), mode="constant")

    # Convert bits into bytes
    byte_array = np.packbits(padded_bits)

    # Decode bytes to string (UTF-8)
    try:
        text = byte_array.tobytes().decode("utf-8")
    except UnicodeDecodeError:
        text = byte_array.tobytes().decode("latin-1")

    return text
