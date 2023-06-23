import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low

        self.offset = tile_width / num_tilings
        num_offsets = np.ceil((state_high-state_low)/self.offset)
        self.weights = np.zeros((np.append(num_offsets.astype(int), (num_tilings))))
        print(self.weights.size)

    def __call__(self,s):
        return np.sum(self.weights[self.get_tile_index(s)])

    def update(self,alpha,G,s_tau):
        update_val = (alpha * (G - self(s_tau)) )
        self.weights[self.get_tile_index(s_tau)] += update_val

        return None

    def get_tile_index(self, state):
        raw_index = (state - self.state_low) // self.offset
        return tuple(raw_index.astype(int))
