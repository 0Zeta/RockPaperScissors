import operator
import numpy as np
import pandas as pd
import cmath
from collections import namedtuple

from rpskaggle.helpers import Policy, one_hot, EQUAL_PROBS


class AntiGeometryPolicy(Policy):
    """
    a counter to the popular Geometry bot
    written by @robga
    adapted from https://www.kaggle.com/robga/beating-geometry-bot/output
    """

    def __init__(self):
        super().__init__()
        self.name = "anti_geometry_policy"
        self.is_deterministic = False
        self.opp_hist = []
        self.my_opp_hist = []
        self.offset = 0
        self.last_feat = None
        self.basis = np.array([1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)])
        self.HistMatchResult = namedtuple("HistMatchResult", "idx length")

    def find_all_longest(self,seq, max_len=None):
        result = []
        i_search_start = len(seq) - 2
        while i_search_start > 0:
            i_sub = -1
            i_search = i_search_start
            length = 0
            while i_search >= 0 and seq[i_sub] == seq[i_search]:
                length += 1
                i_sub -= 1
                i_search -= 1
                if max_len is not None and length > max_len: break
            if length > 0: result.append(self.HistMatchResult(i_search_start + 1, length))
            i_search_start -= 1
        return sorted(result, key=operator.attrgetter("length"), reverse=True)

    def complex_to_probs(self, z):
        probs = (2 * (z * self.basis.conjugate()).real + 1) / 3
        if min(probs) < 0: probs -= min(probs)
        return probs / sum(probs)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:

        if len(history) == 0:
            return EQUAL_PROBS
        else:
            self.action = int(history.loc[step - 1, 'action'])
            self.my_opp_hist.append((int(history.loc[step - 1, 'opponent_action']), self.action))
            self.opp_hist.append(self.action)

            if self.last_feat is not None:
                this_offset = (self.basis[(self.opp_hist[-1] + 1) % 3]) * self.last_feat.conjugate()
                self.offset = (1 - .01) * self.offset + .01 * this_offset

            hist_match = self.find_all_longest(self.my_opp_hist, 20)
            if not hist_match:
                pred = 0
            else:
                feat = self.basis[self.opp_hist[hist_match[0].idx]]
                self.last_feat = self.complex_to_probs(feat / abs(feat)) @ self.basis
                pred = self.last_feat * self.offset * cmath.exp(2j * cmath.pi * 1 / 9)

            probs = self.complex_to_probs(pred)
            if probs[np.argmax(probs)] > .334:
                return one_hot((int(np.argmax(probs)) + 1) % 3)
            else:
                return probs
