#!/usr/bin/env python
"""
A basic adaptive bot. This is part of the second worksheet.

"""

from api import State, util
from operator import mul
import numpy
import random, os
import itertools

from sklearn.externals import joblib

DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model.pkl'


class Bot:
    __max_depth = -1
    __randomize = True

    __model = None

    def __init__(self, randomize=True, depth=4, model_file=DEFAULT_MODEL):

        print(model_file)
        self.__randomize = randomize
        self.__max_depth = depth

        # Load the model
        self.__model = joblib.load(model_file)

    def get_move(self, state):

        val, move = self.value(state)

        return move

    def value(self, state, alpha=float('-inf'), beta=float('inf'), depth=0):
        """
        Return the value of this state and the associated move
        :param state:
        :param alpha: The highest score that the maximizing player can guarantee given current knowledge
        :param beta: The lowest score that the minimizing player can guarantee given current knowledge
        :param depth: How deep we are in the tree
        :return: val, move: the value of the state, and the best move.
        """
        if state.finished():
            return (1.0, None) if state.winner() == 1 else (-1.0, None)

        if depth == self.__max_depth:
            return self.heuristic(state), None

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        for move in moves:
            next_state = state.next(move)
            value, m = self.value(next_state, alpha, beta, depth + 1)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = best_value
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = best_value

            # Prune the search tree
            # We know this state will never be chosen, so we stop evaluating its children
            if beta <= alpha:
                break

        return best_value, best_move

    def heuristic(self, state):
        # Convert the state to a feature vector
        feature_vector = [features(state)]

        # These are the classes: ('won', 'lost')
        classes = list(self.__model.classes_)

        # Ask the model for a prediction
        # This returns a probability for each class
        prob = self.__model.predict_proba(feature_vector)[0]
        # print('{} {} {}'.format(classes, prob, util.ratio_ships(state, 1)))

        # Weigh the win/loss outcomes (-1 and 1) by their probabilities
        res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]
        # print(res)

        return res


def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.
    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    # Features
    p1_garrisons, p2_garrisons = 0.0, 0.0
    p1_turns_per_ship, p2_turns_per_ship = 0.0, 0.0
    p1_center_planets, p2_center_planets = 0.0, 0.0
    p1_fleets, p2_fleets = 0.0, 0.0
    # p1_fleet_distance, p2_fleet_distance = 0.0, 0.0

    for mine in state.planets(1):
        p1_garrisons += state.garrison(mine)
        p1_turns_per_ship += 1.0 / mine.turns_per_ship()
        coords = mine.coords()
        if 0.25 < coords[0] < 0.75 and 0.25 < coords[1] < 0.75:
            p1_center_planets += 1

    for his in state.planets(2):
        p2_garrisons += state.garrison(his)
        p2_turns_per_ship += 1.0 / his.turns_per_ship()
        coords = his.coords()
        if 0.25 < coords[0] < 0.75 and 0.25 < coords[1] < 0.75:
            p2_center_planets += 1

    for fleet in state.fleets():
        planet = fleet.target()
        if fleet.owner() == 1:
            p1_fleets += fleet.size()
        else:
            p2_fleets += fleet.size()

    feature_list = [p1_garrisons, p2_garrisons, p1_fleets, p2_fleets, p1_turns_per_ship, p2_turns_per_ship,
                          p1_center_planets, p2_center_planets]

    i = 0
    result = []

    for subset in itertools.combinations_with_replacement(feature_list, 3):
        product = reduce(mul, subset)
        result.append(product)
        i += 1

    return result
