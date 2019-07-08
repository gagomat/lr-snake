import numpy as np


class Buffer(object):

    def __init__(self, buffer_size, input_shape, output_shape, gamma=0.99):
        self.buffer_size = buffer_size
        self.images = np.zeros((buffer_size, *input_shape))
        self.actions = np.zeros(buffer_size)
        self.avantages = np.zeros(buffer_size)
        self.recompenses = np.zeros(buffer_size)
        self.gamma = gamma

        self.i_save = 0
        self.i_game_start = 0

    def save(self, img, action, recompense):
        self.images[self.i_save] = img
        self.actions[self.i_save] = action
        self.recompenses[self.i_save] = recompense
        self.i_save += 1

    def get(self):
        self.i_save, self.i_game_start = 0, 0
        self.avantages = (self.avantages - np.mean(self.avantages)) / np.std(self.avantages)
        return self.images, self.actions, self.avantages

    def end_game(self):
        sl_game = slice(self.i_game_start, self.i_save)
        self.avantages[sl_game] = self._calc_avantages(self.recompenses[sl_game])
        self.i_game_start = self.i_save

    def full(self):
        return False if self.i_save < self.buffer_size else True

    def _calc_avantages(self, recomp):

        avantage = np.zeros(recomp.shape)

        for i, _ in enumerate(recomp):
            for j, v in enumerate(recomp[i:]):
                avantage[i] += self.gamma ** j * v

        return avantage


