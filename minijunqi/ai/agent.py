



# -*- coding: utf-8 -*-
from __future__ import annotations
try:
    from ..constants import Player
except Exception:
    class Player:
        ORANGE = 0
        PURPLE = 1
        GREEN = 2
        BLUE = 3
try:
    from ..game import Game
except Exception:
    Game = object

from .policy import SharedPolicy

class Agent:
    """Neural agent wrapper for four-country chess with perspective canonicalization and 20-step history."""
    def __init__(self, device='cpu', temperature: float=1.0, net=None):
        if net is not None:
            self.policy = SharedPolicy(net=net, device=device)
        else:
            self.policy = SharedPolicy(device=device)
        self.temperature = temperature
    
    def load(self, ckpt: str): 
        self.policy.load(ckpt)
    
    def reset(self): 
        self.policy.reset_history()

    def _next_piece(self, game: Game, player: Player):
        try:
            from ..constants import DEPLOY_SEQUENCE
            pool = game.state.pools[player]
            for pid in DEPLOY_SEQUENCE:
                if pool.get(pid, 0) > 0:
                    return pid
            for pid, cnt in pool.items():
                if cnt > 0: 
                    return pid
        except Exception:
            return None
        return None

    def select_deploy(self, game: Game, player: Player):
        pid = self._next_piece(game, player)
        if pid is None:
            return None, (-1, -1), None
        rc, pc = self.policy.select_deploy(
            game.state.board, player, pid, no_battle_ratio=0.0, temperature=self.temperature
        )
        # pc为概率
        return pid, rc, pc

    def select_move(self, game: Game, player: Player):
        state = game.state
        try:
            ratio = float(state.no_battle_counter) / max(1, game.cfg.no_battle_draw_steps)
        except Exception:
            ratio = 0.0
        src, dst, pc, pt = self.policy.select_move(
            state.board, player, side_to_move=state.turn, no_battle_ratio=ratio, temperature=self.temperature
        )
        return src, dst, pc, pt
