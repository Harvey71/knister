import numpy as np
import unittest
from gym.envs.toy_text import discrete
import sys
from six import StringIO

class Env(discrete.DiscreteEnv):

    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(self):
        self.edge_length = 5
        self.reset()

    def step(self, action):
        # action: position of empty cell, to fill with rolled result
        self.state[action] = self.state[-1]
        done = not np.any(self.state == 0)
        reward = self.get_score()
        self.roll_dices()
        return self.state, reward, done

    def reset(self):
        # state:
        #   first edge_lentgh^2 values = noted cell values, 0 for empty cell
        #   last value: sum of current dice results
        self.state = np.asarray([0] * (self.edge_length ** 2 + 1), dtype=np.int8)
        self.roll_dices()
        return self.state


    def render(self, mode='human'):
        border = '+--' * self.edge_length + '+\n'
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        matrix = np.reshape(self.state[:-1], (self.edge_length, self.edge_length))
        out = ['\n']
        for row in matrix:
            out.append(border)
            line = '|'.join(['{0:>2}'.format(col) if col else '  ' for col in row])
            line = '|' + line + '|\n'
            out.append(line)
        out.append(border)
        outfile.writelines(out)



    def roll_dices(self):
        # roll two D6 and add values
        result = np.sum(np.random.randint(1, 6, 2, dtype=np.int8))
        self.state[-1] = result

    def get_score_for_row(self,row):
        # filter for used cells
        row = np.extract(row > 0, row)
        # sort
        row = np.sort(row)
        if len(row) == self.edge_length:
            # check for street:
            if np.count_nonzero(np.ediff1d(row) == 1) == self.edge_length - 1:
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

# unit tests
class TestGetScoreForRow(unittest.TestCase):

    def setUp(self):
        self.env = Env()

    def assertScore(self, row, score):
        self.assertEqual(self.env.get_score_for_row(np.asarray(row, dtype=np.int8)), score)

    def test_blank(self):
        self.assertScore([0, 0, 0, 0, 0], 0)
        self.assertScore([0, 0, 0, 0, 2], 0)
        self.assertScore([2, 3, 4, 5, 0], 0)
        self.assertScore([2, 3, 4, 6, 7], 0)

    def test_street(self):
        self.assertScore([2, 3, 4, 5, 6], 12)
        self.assertScore([5, 4, 8, 7, 6], 8)

    def test_pair(self):
        self.assertScore([0, 2, 0, 2, 0], 1)
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
        self.env.state = np.asarray(matrix + [0], dtype=np.int8)
        self.assertEqual(self.env.get_score(), score)

    def test_blank(self):
        self.assertScore(
            [2, 0, 0, 0, 3,
             0, 0, 0, 0, 0,
             3, 4, 6, 7, 8,
             0, 0, 0, 0, 0,
             5, 0, 0, 0, 4,], 0)

    def test_diags(self):
        # quality street plus full house
        self.assertScore(
            [2, 0, 0, 0, 7,
             0, 3, 0, 7, 0,
             0, 0, 4, 0, 0,
             0, 4, 0, 5, 0,
             4, 0, 0, 0, 6,], 12 * 2 + 8 * 2)

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

class TestStep(unittest.TestCase):

    def setUp(self):
        self.env = Env()
        self.sum = 0

    def test_step_1(self):
        rolled = self.env.state[-1]
        self.sum += rolled
        state, reward, done = self.env.step(12)
        self.assertEqual(state[12], rolled)
        self.assertEqual(np.sum(state[:-1]), self.sum)

    def test_step_2(self):
        rolled = self.env.state[-1]
        self.sum += rolled
        state, reward, done = self.env.step(0)
        self.assertEqual(state[0], rolled)
        self.assertEqual(np.sum(state[:-1]), self.sum)

    def test_step_3(self):
        self.env.reset()
        self.assertEqual(self.env.state[0], 0)
        self.assertEqual(self.env.state[12], 0)

    def test_done(self):
        self.env.reset()
        for action in range(25):
            state, reward, done = self.env.step(action)
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
