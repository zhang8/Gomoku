#######
# Copyright 2020 Jian Zhang, All rights reserved
##

import logging
import math

from MCTSBase import *
log = logging.getLogger(__name__)


class MCTSNode(TreeNode):
    def __init__(self, standardState):
        xs, ys = np.nonzero(np.sum(standardState, axis=0) == 0)
        self.As = np.stack([xs, ys], axis=1)
        N = len(self.As)
        self.Qsa = np.zeros((N,), dtype=np.float32)
        self.Nsa = np.zeros((N,), dtype=np.int32)
        self.Ps = None
        self.Ns = 0

        self.terminal = False
        self.v = 0

    def is_terminal(self):
        return self.terminal

    def value(self):
        return self.v

    def find_action(self):
        cur_best = -float('inf')
        best_act = 0
        # pick the action with the highest upper confidence bound
        for i in range(len(self.As)):
            if self.Nsa[i] > 0:
                u = self.Qsa[i] + cpuct * self.Ps[i] * math.sqrt(self.Ns) / (1 + self.Nsa[i])
            else:
                u = cpuct * self.Ps[i] * math.sqrt(self.Ns + EPS)
            if u > cur_best:
                cur_best = u
                best_act = i

        self.best_act = best_act
        return tuple(self.As[best_act])

    def update(self, v):
        best_act = self.best_act
        self.Qsa[best_act] = (self.Nsa[best_act] * self.Qsa[best_act] + v) / (self.Nsa[best_act] + 1)
        self.Nsa[best_act] += 1
        self.Ns += 1


from model import GomokuModel
class MCTS(MCTSBase):
    """
    Monte Carlo Tree Search
    """
    def __init__(self, game):
        super().__init__(game)
        self.nnet = GomokuModel('net.tf')
        self.Ss = {}

    def reset(self):
        self.Ss = {}

    def get_visit_count(self, state):
        s = self.game.stringRepresentation(state)
        n = self.Ss[s]
        probs_full = np.zeros((self.game.board_sz, self.game.board_sz), dtype=np.float32)
        probs_full[n.As[:, 0], n.As[:, 1]] = n.Nsa
        return probs_full

    def get_treenode(self, standardState):
        s = self.game.stringRepresentation(standardState)
        return self.Ss.get(s, None)

    def new_tree_node(self, standardState, game_end):
        s = self.game.stringRepresentation(standardState)
        self.Ss[s] = MCTSNode(standardState)
        n = self.Ss[s]

        if game_end is None:
            Ps, v = self.nnet.predict(standardState)
            n.Ps = Ps[n.As[:, 0], n.As[:, 1]]
            n.terminal = False
            n.v = v
        else:
            n.terminal = True
            if game_end == 0:
                n.v = game_end
            else:
                n.v = -game_end

        return n


if __name__ == '__main__':
    from gomoku import Gomoku
    g = Gomoku()
    ts = MCTS(g)

    print(ts.getActionProb(g.board, 20))
