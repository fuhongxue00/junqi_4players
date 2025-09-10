# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import copy
from .constants import (
    BOARD_H, BOARD_W, Player, PieceID, STRENGTH, SpecialArea,
    HEADQUARTERS_POSITIONS, DEPLOY_AREA_ROWS, DEPLOY_AREA_COLS,
    TEAMMATES, PLAYER_ORDER
)
from .pieces import Piece

Coord = Tuple[int, int]

class Board:
    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        self.special_areas: List[List[SpecialArea]] = [[SpecialArea.NORMAL for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        self._init_special_areas()
        self._init_railroad()


    # 定义铁路信息，self.railroads为铁路列表，每条铁路是若干坐标的有序列表
    # self.railroad_map为每个坐标对应的铁路编号集合（可能有交叉）
    def _init_railroad(self):
        """初始化铁路信息"""
        # 定义铁路（这里只是示例，具体铁路布局可根据四国军棋实际规则调整）
        # 例：横向和纵向的主干铁路
        self.railroads: List[List[Coord]] = []
        self.railroads.append([(6,c) for c in range(1,BOARD_W-1)])
        self.railroads.append([(10,c) for c in range(1,BOARD_W-1)])
        self.railroads.append([(8,6),(8,8),(8,10)])
        self.railroads.append([(r,6) for r in range(1,BOARD_H-1)])
        self.railroads.append([(r,10) for r in range(1,BOARD_H-1)])
        self.railroads.append([(6,8),(8,8),(10,8)])

        self.railroads.append([(6,c) for c in range(1,6)]+[(r,6) for r in range(5,0,-1)])
        self.railroads.append([(10,c) for c in range(1,6)]+[(r,6) for r in range(11,16)])
        self.railroads.append([(10,c) for c in range(15,10,-1)]+[(r,10) for r in range(11,16)])
        self.railroads.append([(6,c) for c in range(15,10,-1)]+[(r,10) for r in range(5,0,-1)])

        for r in (1,5,11,15):
            self.railroads.append([(r,c) for c in range(6,11)])
        for c in (1,5,11,15):
            self.railroads.append([(r,c) for r in range(6,11)])

        # print(self.railroads)
        self.pos_to_nodes = {}
        for rid, path in enumerate(self.railroads):
            for idx, rc in enumerate(path):
                self.pos_to_nodes.setdefault(rc, []).append((rid, idx))


    def _init_special_areas(self):
        """初始化特殊区域"""
        # 设置大本营
        for player, positions in HEADQUARTERS_POSITIONS.items():
            for r, c in positions:
                self.special_areas[r][c] = SpecialArea.HEADQUARTERS


        for player in PLAYER_ORDER:
            for r in range(6):
                for c in range(6):
                    r_global, c_global = self.rotate_coord_to_global(player, (r, c))
                    self.special_areas[r_global][c_global] = SpecialArea.FORBIDDEN
            r = 7
            for c in range(6,11):
                r_global, c_global = self.rotate_coord_to_global(player, (r, c))
                self.special_areas[r_global][c_global] = SpecialArea.FORBIDDEN

            for r,c in [(12,7), (12,9),(13,8),(14,7),(14,9)]:
                r_global, c_global = self.rotate_coord_to_global(player, (r, c))
                self.special_areas[r_global][c_global] = SpecialArea.CAMP
        # TODO: 行营和禁入区的位置需要后续配置
        # 目前先预留，后续可以根据需要添加具体位置
    
    def clone(self) -> 'Board':
        return copy.deepcopy(self)
    
    def in_bounds(self, r, c): 
        return 0 <= r < BOARD_H and 0 <= c < BOARD_W
    
    def get(self, rc: Coord): 
        """获取全局坐标位置的棋子（内部使用）"""
        r, c = rc
        return self.grid[r][c]
    
    def set(self, rc: Coord, p: Optional[Piece]): 
        """设置全局坐标位置的棋子（内部使用）"""
        r, c = rc
        self.grid[r][c] = p
    
    def get_piece(self, player: Player, rc: Coord) -> Optional[Piece]:
        """获取玩家视角坐标位置的棋子"""
        global_rc = self.rotate_coord_to_global(player, rc)
        return self.get(global_rc)
    
    def set_piece(self, player: Player, rc: Coord, piece: Optional[Piece]) -> None:
        """设置玩家视角坐标位置的棋子"""
        global_rc = self.rotate_coord_to_global(player, rc)
        self.set(global_rc, piece)
    
    def iter_coords(self):
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                yield (r, c)
    
    def get_special_area(self, rc: Coord) -> SpecialArea:
        r, c = rc
        return self.special_areas[r][c]
    
    def is_headquarters(self, rc: Coord) -> bool:
        return self.get_special_area(rc) == SpecialArea.HEADQUARTERS
    
    def is_camp(self, rc: Coord) -> bool:
        return self.get_special_area(rc) == SpecialArea.CAMP
    
    def is_forbidden(self, rc: Coord) -> bool:
        return self.get_special_area(rc) == SpecialArea.FORBIDDEN
    
    def is_in_deploy_area(self, player: Player, rc: Coord) -> bool:
        """检查坐标是否在玩家的可部署区域内（玩家视角）"""
        r, c = rc
        # 检查是否在玩家视角的最后六行中间五列
        return r in DEPLOY_AREA_ROWS and c in DEPLOY_AREA_COLS
    
    def can_place(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        # 将玩家视角坐标转换为全局坐标
        # print(f"can_place: player={player}, pid={pid}, rc={rc}")
        global_rc = self.rotate_coord_to_global(player, rc)
        r, c = global_rc
        
        # print(f"global_rc={global_rc}, r={r}, c={c}")
        
        if not self.in_bounds(r, c): 
            return False
        if self.get(global_rc) is not None: 
            return False
        if self.is_forbidden(global_rc): 
            return False
        if self.is_camp(global_rc): 
            return False
        
        # 检查是否在可部署区域内（玩家视角，使用玩家传的rc）
        if not self.is_in_deploy_area(player, rc):
            return False
        
        # 军旗只能放在大本营
        if pid == PieceID.JUNQI:
            return self.is_headquarters(global_rc)

        if pid == PieceID.DILEI:
            tempr , _ = rc
            return tempr in range(15,17)
        if pid == PieceID.ZHADAN:
            tempr , _ = rc
            return tempr in range(12,17)
        return True
    
    def place(self, player: Player, pid: PieceID, rc: Coord) -> bool:
        if not self.can_place(player, pid, rc): 
            print(f"失败place: player={player}, pid={pid}, rc={rc}, can_place=False")
            return False
        # 使用玩家视角的set方法
        # print(f"place: player={player}, pid={pid}, rc={rc}")
        self.set_piece(player, rc, Piece(pid, player))
        return True
    
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def neighbors(self, rc: Coord):
        r, c = rc
        out = []
        for dr, dc in Board.DIRS:
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc): 
                out.append((rr, cc))
        return out
    
    def can_move_from_to(self, player: Player, src: Coord, dst: Coord) -> bool:
        # 使用玩家视角的get方法
        # print(f"127can_move_from_to: player={player}, src={src}, dst={dst}")
        p = self.get_piece(player, src)
        assert self.is_camp(dst) == self.is_camp(self.rotate_coord_to_global(player, dst))
        if p is None or p.owner != player or not p.can_move(): 
            return False
        if self.is_forbidden(dst): 
            return False
        q = self.get_piece(player, dst)
        if q is not None:
            # 不能攻击队友
            if q.owner == player or q.owner == TEAMMATES.get(player):
                return False
            # 行营规则：如果目标位置是行营且有棋子，不能进攻
            if self.is_camp(dst) and q is not None:
                return False

        if dst in self.neighbors(src):
            return True

        src_r, src_c = src
        dst_r, dst_c = dst
        if self.is_camp(src) or self.is_camp(dst):

            dr = abs(src_r - dst_r)
            dc = abs(src_c - dst_c)
            if dr <= 1 and dc <= 1 and not (dr == 0 and dc == 0):
                
                return True

        for path in self.railroads:  # 每一条铁路是一串有序坐标
            if (src_r, src_c) in path and (dst_r, dst_c) in path:
                i = path.index((src_r, src_c))
                j = path.index((dst_r, dst_c))
                lo, hi = (i, j) if i <= j else (j, i)
                blocked = False
                for k in range(lo + 1, hi):
                    if self.get_piece(player,path[k]) is not None:
                        blocked = True
                        break
                if not blocked:
                    return True

        # 3. 工兵全铁路连通搜索：中间节点必须为空，只有最终 dst 允许有棋子
        if p.pid == PieceID.GONGBING:
            if src in self.pos_to_nodes and dst in self.pos_to_nodes:
                from collections import deque
                q = deque([src])
                visited = {src}
                while q:
                    cur = q.popleft()
                    if cur == dst:
                        return True  # 找到一条不被阻挡的铁路路径，直接返回

                    for rid, idx in self.pos_to_nodes.get(cur, []):
                        path = self.railroads[rid]
                        for nxt_idx in (idx - 1, idx + 1):
                            if 0 <= nxt_idx < len(path):
                                nxt = path[nxt_idx]
                                if nxt in visited:
                                    continue
                                # 中间节点必须为空；只有最终 dst 允许占有棋子
                                if nxt != dst and self.get_piece(player, nxt) is not None:
                                    continue
                                visited.add(nxt)
                                q.append(nxt)
        return False
    
    def compare(self, a: Piece, d: Piece) -> str:
        if d.pid == PieceID.JUNQI: 
            return 'attacker'
        if a.pid == PieceID.ZHADAN or d.pid == PieceID.ZHADAN: 
            return 'both'
        if a.pid == PieceID.GONGBING and d.pid == PieceID.DILEI:
            return 'attacker'
        sa, sd = STRENGTH[a.pid], STRENGTH[d.pid]
        if sa > sd: 
            return 'attacker'
        if sa < sd: 
            return 'defender'
        return 'both'
    
    def move(self, player: Player, src: Coord, dst: Coord) -> Dict:
        if not self.can_move_from_to(player, src, dst):
            return {'ok': False, 'reason': 'illegal'}
        
        # 使用玩家视角的get方法
        p = self.get_piece(player, src)
        q = self.get_piece(player, dst)
        
        if q is None:
            self.set_piece(player, dst, p)
            self.set_piece(player, src, None)
            return {'ok': True, 'type': 'move', 'flag_captured': False}
        
        outcome = self.compare(p, q)
        flag_captured = (q.pid == PieceID.JUNQI)
        
        if outcome == 'attacker':
            self.set_piece(player, dst, p)
            self.set_piece(player, src, None)
            return {'ok': True, 'type': 'capture', 'result': 'attacker', 'flag_captured': flag_captured,'p_owner':p.owner,'q_owner':q.owner}
        elif outcome == 'defender':
            self.set_piece(player, src, None)
            return {'ok': True, 'type': 'capture', 'result': 'defender', 'flag_captured': False,'p_owner':p.owner,'q_owner':q.owner}
        else:
            self.set_piece(player, src, None)
            self.set_piece(player, dst, None)
            return {'ok': True, 'type': 'capture', 'result': 'both', 'flag_captured': flag_captured,'p_owner':p.owner,'q_owner':q.owner}
    
    def rotate_board_for_player(self, player: Player, obs: List[List[int]]) -> List[List[int]]:
        """为指定玩家旋转棋盘视角"""
        if player == Player.ORANGE:
            return obs  # 橙色不需要旋转
        elif player == Player.PURPLE:
            # 顺时针旋转90度
            return [[obs[BOARD_H-1-c][r] for c in range(BOARD_W)] for r in range(BOARD_H)]
        elif player == Player.GREEN:
            # 旋转180度
            return [[obs[BOARD_H-1-r][BOARD_W-1-c] for c in range(BOARD_W)] for r in range(BOARD_H)]
        elif player == Player.BLUE:
            # 逆时针旋转90度
            return [[obs[c][BOARD_W-1-r] for c in range(BOARD_W)] for r in range(BOARD_H)]
        return obs
    
    def rotate_coord_to_global(self, player: Player, coord: Coord) -> Coord:
        """将玩家视角的坐标转换为全局坐标"""
        r, c = coord
        if player == Player.ORANGE:
            return (r, c)  # 橙色不需要旋转
        elif player == Player.PURPLE:
            # 紫色在右侧，玩家坐标逆时针旋转90度得到全局坐标
            return (BOARD_H - 1 - c, r)
        elif player == Player.GREEN:
            # 绿色在上方，玩家坐标旋转180度得到全局坐标
            return (BOARD_H - 1 - r, BOARD_W - 1 - c)
        elif player == Player.BLUE:
            # 蓝色在左侧，玩家坐标顺时针旋转90度得到全局坐标
            return (c, BOARD_W - 1 - r)
        return coord
    
    def rotate_coord_from_global(self, player: Player, coord: Coord) -> Coord:
        """将全局坐标转换为玩家视角的坐标"""
        r, c = coord
        if player == Player.ORANGE:
            return (r, c)  # 橙色不需要旋转
        elif player == Player.PURPLE:
            # 紫色：全局坐标顺时针旋转90度得到玩家坐标
            return (c, BOARD_H - 1 - r)
        elif player == Player.GREEN:
            # 绿色：全局坐标旋转180度得到玩家坐标
            return (BOARD_H - 1 - r, BOARD_W - 1 - c)
        elif player == Player.BLUE:
            # 蓝色：全局坐标逆时针旋转90度得到玩家坐标
            return (BOARD_W - 1 - c, r)
        return coord
    
    def observe(
        self,
        viewer: Player,
        reveal_all: bool = False,
        hide_enemy_positions: bool = False,
    ):
        obs = [[0 for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                p = self.grid[r][c]
                if p is None:
                    obs[r][c] = int(PieceID.EMPTY)
                elif reveal_all or p.owner == viewer:
                    obs[r][c] = int(p.pid)
                else:
                    # 队友的棋子对当前玩家可见
                    if p.owner == TEAMMATES.get(viewer) and not hide_enemy_positions:
                        obs[r][c] = int(p.pid)
                    else:
                        obs[r][c] = int(PieceID.EMPTY if hide_enemy_positions else PieceID.UNKNOWN_ENEMY)
        
        # 旋转视角
        return self.rotate_board_for_player(viewer, obs)
    
    def has_legal_move(self, player: Player) -> bool:
        """判断该玩家是否有至少一个合法走子"""
        for r, c in self.iter_coords():
            p = self.get((r, c))
            if p is None or p.owner != player or not p.can_move():
                continue
            global_src = (r, c)
            # 将全局坐标转换为玩家视角坐标
            player_src = self.rotate_coord_from_global(player, global_src)
            for global_dst in self.neighbors(global_src):
                # 将全局坐标转换为玩家视角坐标
                player_dst = self.rotate_coord_from_global(player, global_dst)
                if self.can_move_from_to(player, player_src, player_dst):
                    return True
        return False