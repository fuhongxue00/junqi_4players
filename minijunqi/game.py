
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from .board import Board, Coord
from .constants import (
    Player, PieceID, INITIAL_POOL, DEFAULT_NO_BATTLE_DRAW, 
    BOARD_H, BOARD_W, PLAYER_ORDER, TEAMMATES
)

@dataclass
class GameConfig:
    no_battle_draw_steps: int = DEFAULT_NO_BATTLE_DRAW

@dataclass
class GameState:
    board: Board = field(default_factory=Board)
    turn: Player = Player.ORANGE
    phase: str = 'deploy'
    pools: Dict[Player, Dict[PieceID, int]] = field(default_factory=lambda: {
        Player.ORANGE: dict(INITIAL_POOL),
        Player.PURPLE: dict(INITIAL_POOL),
        Player.GREEN: dict(INITIAL_POOL),
        Player.BLUE: dict(INITIAL_POOL),
    })
    no_battle_counter: int = 0
    winner: Optional[Player] = None
    winning_team: Optional[Tuple[Player, Player]] = None  # 获胜队伍
    end_reason: Optional[str] = None
    eliminated_players: List[Player] = field(default_factory=list)  # 已淘汰的玩家

class Game:
    def __init__(self, cfg: GameConfig = GameConfig()):
        self.cfg = cfg
        self.state = GameState()
    
    def can_deploy_any(self, player: Player) -> bool:
        return any(v > 0 for v in self.state.pools[player].values())
    
    def deploy(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        if self.state.phase != 'deploy': 
            return False
        pool = self.state.pools[player]
        if pool.get(pid, 0) <= 0: 
            return False
        if not self.state.board.place(player, pid, rc): 
            return False
        pool[pid] -= 1
        
        # 切换到下一个玩家
        current_idx = PLAYER_ORDER.index(player)
        next_idx = (current_idx + 1) % len(PLAYER_ORDER)
        self.state.turn = PLAYER_ORDER[next_idx]
        
        # 检查是否所有玩家都部署完毕
        if all(not self.can_deploy_any(p) for p in PLAYER_ORDER):
            self.state.phase = 'play'
            self.state.turn = Player.ORANGE
        
        return True
    
    # def legal_moves(self, player: Player):
    #     if self.state.phase != 'play': 
    #         return []
    #     b = self.state.board
    #     moves = []
    #     for r in range(BOARD_H):
    #         for c in range(BOARD_W):
    #             p = b.get((r, c))
    #             if p is None or p.owner != player or not p.can_move(): 
    #                 continue
    #             for nb in b.neighbors((r, c)):
    #                 if b.can_move_from_to(player, (r, c), nb):
    #                     moves.append(((r, c), nb))
    #     return moves
    
    def check_player_elimination(self, player: Player) -> bool:
        """检查玩家是否被淘汰"""
        # 检查军旗是否被夺
        board = self.state.board
        for r, c in board.iter_coords():
            p = board.get((r, c))
            if p and p.owner == player and p.pid == PieceID.JUNQI:
                return False  # 军旗还在，未被淘汰
        
        # 检查是否有可移动的棋子,这个逻辑不对，到时候要改一下
        if not board.has_legal_move(player):
            return True  # 无棋可走，被淘汰
        
        return False
    
    def check_team_elimination(self, team: Tuple[Player, Player]) -> bool:
        """检查队伍是否被淘汰"""
        return all(p in self.state.eliminated_players for p in team)
    
    def eliminate_player(self, player: Player):
        """淘汰玩家"""
        if player not in self.state.eliminated_players:
            self.state.eliminated_players.append(player)
            print(f"玩家 {player.name} 被淘汰")
    
    def step(self, src: Coord, dst: Coord) -> Dict:
        assert self.state.phase == 'play', 'not in play phase'
        player = self.state.turn
        
        ev = self.state.board.move(player, src, dst)
        if not ev.get('ok'): 
            return ev
        
        if ev['type'] == 'move': 
            self.state.no_battle_counter += 1
        else:
            self.state.no_battle_counter = 0
            if ev.get('flag_captured'):
                # 军旗被夺，检查被攻击方是否被淘汰
                attacked_player = None
                for r, c in self.state.board.iter_coords():
                    p = self.state.board.get((r, c))
                    if p and p.pid == PieceID.JUNQI and (r, c) == dst:
                        attacked_player = p.owner
                        break
                
                if attacked_player:
                    self.eliminate_player(attacked_player)
        
        # 检查当前玩家是否被淘汰
        if self.check_player_elimination(player):
            self.eliminate_player(player)
        
        # 检查是否有队伍获胜
        if not self.state.winner:
            orange_green_team = (Player.ORANGE, Player.GREEN)
            purple_blue_team = (Player.PURPLE, Player.BLUE)
            
            if self.check_team_elimination(orange_green_team):
                self.state.winner = Player.PURPLE  # 紫色队伍获胜
                self.state.winning_team = purple_blue_team
                self.state.end_reason = 'team_eliminated'
            elif self.check_team_elimination(purple_blue_team):
                self.state.winner = Player.ORANGE  # 橙色队伍获胜
                self.state.winning_team = orange_green_team
                self.state.end_reason = 'team_eliminated'
        
        # 检查平局
        if self.state.winner is None and self.state.no_battle_counter >= self.cfg.no_battle_draw_steps:
            self.state.end_reason = 'draw'
        
        # 切换到下一个玩家
        if self.state.winner is None and self.state.end_reason is None:
            current_idx = PLAYER_ORDER.index(player)
            next_idx = (current_idx + 1) % len(PLAYER_ORDER)
            next_player = PLAYER_ORDER[next_idx]
            
            # 跳过已淘汰的玩家
            while next_player in self.state.eliminated_players:
                next_idx = (next_idx + 1) % len(PLAYER_ORDER)
                next_player = PLAYER_ORDER[next_idx]
            
            self.state.turn = next_player
        
        return ev
    
    def is_over(self) -> bool:
        return self.state.winner is not None or self.state.end_reason is not None
    
    def resign(self, player: Player):
        """玩家认输"""
        self.eliminate_player(player)
        
        # 检查是否导致队伍失败
        orange_green_team = (Player.ORANGE, Player.GREEN)
        purple_blue_team = (Player.PURPLE, Player.BLUE)
        
        if player in orange_green_team and self.check_team_elimination(orange_green_team):
            self.state.winner = Player.PURPLE
            self.state.winning_team = purple_blue_team
            self.state.end_reason = 'resign'
        elif player in purple_blue_team and self.check_team_elimination(purple_blue_team):
            self.state.winner = Player.ORANGE
            self.state.winning_team = orange_green_team
            self.state.end_reason = 'resign'
