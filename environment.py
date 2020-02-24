import warnings
warnings.filterwarnings('ignore')

import numpy as np
import unittest
import random

import sys
from six import StringIO

class Env:

    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(self):
        self.edge_length = 5
        self.reset()
        self.action_space = 25
        self.observation_space = self.state_one_hot.shape
        self.reward_range = (0, 100)

    def step(self, action):
        # action: position of empty cell, to fill with rolled result
        if self.state[action] == 0:
            val = self.state[25]
            self.state[action] = val
            self.state_one_hot[action][0] = 0
            self.state_one_hot[action][val] = 1
            self.roll_dices()
            done = not np.any(self.state == 0)
            reward = 0 if not done else self.get_score()
        else:
            # rule violation, sudden death
            reward = -1000
            done = True

        info = {}
        return self.state_one_hot.copy(), reward, done, info

    def reset(self, start_dice=None):
        self.state = np.zeros((26,), dtype=np.ubyte)
        self.state_one_hot = np.zeros((26, 12), dtype=np.ubyte)
        for i in range(25):
            self.state_one_hot[i, 0] = 1
        self.roll_dices()
        return self.state_one_hot.copy()

    def render(self, mode='human'):
        out = []
        border = '+--' * self.edge_length + '+\n'
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        matrix = np.reshape(self.state[:-1], (5, 5))
        for row in matrix:
            out.append(border)
            line = '|'.join(['{0:>2}'.format(col + 1) if col else '  ' for col in row])
            line = '|' + line + '|\n'
            out.append(line)
        out.append(border)
        out.append(f'next dice: {self.d1}+{self.d2}={self.state[25] + 1}\n')
        out.append('\n\n')
        outfile.writelines(out)

    def roll_dices(self):
        # remove old one hot dice result
        self.state_one_hot[25, self.state[25]] = 0
        # roll two D6 and add values
        self.d1 = random.randint(1, 6)
        self.d2 = random.randint(1, 6)
        result = self.d1 + self.d2
        self.state[25] = result - 1
        # set new one hot dice result
        self.state_one_hot[25, result - 1] = 1

    def get_score_for_row(self,row):
        # sort
        row = np.sort(row)
        # check for street:
        if np.count_nonzero(np.ediff1d(row) == 1) == 4:
            # street
            if np.any(row == 7):
                # street with a 7
                return 8
            else:
                # street without a 7
                return 12
        # find n_lets
        n_lets = np.unique(row, return_counts=True)[1]
        # eliminate singles
        n_lets = np.extract(n_lets > 1, n_lets)
        if len(n_lets) == 0:
            # blank
            return 0
        elif len(n_lets) == 1:
            # one n-let
            size = n_lets[0]
            if size == 2:
                # one pair
                return 1
            if size == 3:
                # one triplet
                return 3
            if size == 4:
                # one qudruplet
                return 6
            if size == 5:
                # one quintuplet
                return 10
            # just singles
            return 0
        else:
            # two n_lets
            if np.any(n_lets == 3):
                # full house
                return 8
            else:
                # two pairs
                return 3

    def get_score(self):
        score = 0
        matrix = np.reshape(self.state[:-1], (self.edge_length, self.edge_length))
        for m in (matrix, matrix.T):
            for row in m:
                score += self.get_score_for_row(row)
        for m in (matrix, np.flip(matrix, axis=1)):
            score += 2 * self.get_score_for_row(np.diag(m))
        return score

    def get_actions(self):
        return np.argwhere(self.state[:-1])

    def get_hash(self):
        s = ''
        for i in range(26):
            if i == 25:
                s += '|'
            s += '%.2d' % self.state[i]
        return s

    def filter_probs(self, probs):
        filter = self.state[:-1] > 0
        probs[filter] = 0
        sum = np.sum(probs)
        if sum > 0:
            probs = probs / sum
        else:
            probs = np.full(25, 1 / 25)
        return probs

# unit tests
class TestGetScoreForRow(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def assertScore(self, row, score):
        self.assertEqual(self.env.get_score_for_row(np.asarray(row, dtype=np.ubyte)), score)

    def test_blank(self):
        self.assertScore([12, 10, 11, 2, 3], 0)
        self.assertScore([2, 3, 4, 6, 7], 0)

    def test_street(self):
        self.assertScore([2, 3, 4, 5, 6], 12)
        self.assertScore([5, 4, 8, 7, 6], 8)

    def test_pair(self):
        self.assertScore([12, 2, 11, 2, 10], 1)
        self.assertScore([2, 3, 4, 5, 2], 1)

    def test_triplet(self):
        self.assertScore([2, 3, 2, 5, 2], 3)

    def test_quadruplet(self):
        self.assertScore([2, 3, 2, 2, 2], 6)

    def test_quintuplet(self):
        self.assertScore([2, 2, 2, 2, 2], 10)

    def test_full_house(self):
        self.assertScore([2, 3, 2, 3, 3], 8)

    def test_two_pairs(self):
        self.assertScore([2, 7, 2, 3, 3], 3)


class TestGetScore(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def assertScore(self, matrix, score):
        self.env.state = np.asarray(matrix + [0], dtype=np.ubyte)
        self.assertEqual(self.env.get_score(), score)

    def test_real_life_example(self):
        self.assertScore(
            [2, 8, 9, 9, 7,
             2, 3, 0, 7, 8,
             4, 4, 4, 4, 4,
             7, 4, 7, 5, 9,
             4, 9, 7, 7, 6,],
            3 + 1 + 1 + 1 + 0 + # cols
            1 + 0 + 10 + 1 + 1 + # rows
            12 * 2 + 8 * 2) # diags

    def test_real_life_example2(self):
        self.assertScore(
            [4, 4, 4, 5, 2,
             5, 10, 4, 10, 6,
             7, 5, 10, 3, 4,
             5, 8, 11, 4, 3,
             10, 10, 6, 8, 7,],
            1 + 1 + 1 + 0 + 0 + # cols
            3 + 1 + 0 + 0 + 1 + # rows
            3 * 2 + 3 * 2) # diags

class TestStep(unittest.TestCase):

    def setUp(self):
        self.env = Env()
        self.sum = 0

    def test_step_1(self):
        rolled = self.env.state[-1]
        self.sum += rolled
        state_one_hot, reward, done, _ = self.env.step(12)
        state = self.env.state
        self.assertEqual(np.sum(state_one_hot[:-1]), 25)
        self.assertEqual(np.sum(state[:-1]), self.sum)

    def test_step_2(self):
        rolled = self.env.state[-1]
        self.sum += rolled
        state_one_hot, reward, done, _ = self.env.step(0)
        state = self.env.state
        self.assertEqual(np.sum(state_one_hot[:-1]), 25)
        self.assertEqual(np.sum(state[:-1]), self.sum)

    def test_step_3(self):
        state_one_hot = self.env.reset()
        state = self.env.state
        self.assertEqual(np.sum(state_one_hot[:-1]), 25)
        self.assertEqual(np.sum(state[:-1]), 0)

    def test_done(self):
        self.env.reset()
        for action in range(25):
            state, reward, done, _ = self.env.step(action)
            self.assertEqual(done, action == 24)

class TestGetActions(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def test_get_actions(self):
        for action in range(0, 25, 2):
            self.env.step(action)
        actions = self.env.get_actions()
        self.assertEqual(len(actions), 13)
        self.assertFalse(np.any(actions % 2 == 1))


if __name__ == '__main__':
    unittest.main()
