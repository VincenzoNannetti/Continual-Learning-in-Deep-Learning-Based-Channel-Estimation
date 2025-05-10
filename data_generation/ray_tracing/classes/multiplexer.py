"""
Filename: ./data_generation/ray_tracing/classes/multiplexer.py
Author: Vincenzo Nannetti
Date: 09/05/2025 (Refactored Date)
Description: Class to multiplex input signal into OFDM symbols.
"""

import numpy as np

class Multiplexer:
    def __init__(self, no_subc, cp):
        self.no_subcarriers = no_subc
        self.cyclic_prefix  = cp
        
        # For storage
        self.arranged_grid  = None
        self.OFDM_symbols   = None

    def multiplex(self, input_signal):
        """
        Multiplexes input signal into OFDM symbols.
        For dataset generation, we just need to format the data for channel application (actual messages are not sent).
        """
        temp_signal = input_signal.copy()
        
        # Calculate number of OFDM symbols needed
        data_positions_per_symbol = self.no_subcarriers
        total_ofdm_symbols = int(np.ceil(len(temp_signal) / data_positions_per_symbol))
        
        # Pad the input signal if needed
        padded_length = total_ofdm_symbols * data_positions_per_symbol
        if len(temp_signal) < padded_length:
            temp_signal = np.pad(temp_signal, (0, padded_length - len(temp_signal)), 'constant')
        
        # Reshape into a grid
        self.arranged_grid = temp_signal.reshape(self.no_subcarriers, total_ofdm_symbols)
        
        # IFFT to get OFDM symbols
        self.OFDM_symbols = np.fft.ifft(self.arranged_grid, axis=0)
        
        # Apply cyclic prefix if specified
        if self.cyclic_prefix > 0:
            # Create an array to hold time-domain signal with CP
            symbol_length = self.no_subcarriers + self.cyclic_prefix
            time_domain_with_cp = np.zeros((symbol_length, total_ofdm_symbols), dtype=complex)
            
            for i in range(total_ofdm_symbols):
                # Extract the current OFDM symbol
                current_symbol = self.OFDM_symbols[:, i]
                
                # Add cyclic prefix (copy from end of symbol)
                cp_portion = current_symbol[-self.cyclic_prefix:]
                time_domain_with_cp[:self.cyclic_prefix, i] = cp_portion
                time_domain_with_cp[self.cyclic_prefix:, i] = current_symbol
            
            # Flatten to time-domain signal with CP
            time_domain_signal = time_domain_with_cp.flatten()
        else:
            # Flatten to time-domain signal without CP
            time_domain_signal = self.OFDM_symbols.flatten()
        
        return time_domain_signal
        
    def get_ofdm_symbols(self):
        return self.OFDM_symbols

    def get_arranged_grid(self):
        return self.arranged_grid