# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Deque
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

try:
    from ..constants import BOARD_H, BOARD_W, Player, PieceID, SpecialArea, TEAMMATES,PLAYER_ORDER
except Exception:
    class Player:
        ORANGE = 0
        PURPLE = 1
        GREEN = 2
        BLUE = 3
    BOARD_H, BOARD_W = 17, 17

try:
    from ..board import Board
except Exception as e:
    print(f"board异常: {e}") 
    Board = object

from .net import PolicyNet, HISTORY_STEPS, PIECE_PLANES_PER_FRAME, STATE_EXTRA_C, SPECIAL_AREA_C, RESERVED_EXTRA_C

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
HW = BOARD_H * BOARD_W

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits.clone()
    logits[mask==0] = float('-inf')
    return F.softmax(logits, dim=dim)

def encode_board_planes(board: Board, viewer: Player, reveal_all: bool=False, is_deploy: bool=False) -> torch.Tensor:
    """Return (30,H,W) one-hot planes for four-country chess."""
    H, W = BOARD_H, BOARD_W
    C = PIECE_PLANES_PER_FRAME  # 30 channels
    x = np.zeros((C, H, W), dtype=np.float32)
    
    try:
        obs = board.observe(viewer, reveal_all=reveal_all, hide_enemy_positions=is_deploy and not reveal_all)
        
        # 通道分配：
        # 0-11: 己方12种棋子
        # 12-23: 队友12种棋子  
        # 24-25: 未知上家、下家棋子
        # 26: 无棋子
        # 27-29: 预留3个通道
        
        for r in range(H):
            for c in range(W):
                p = board.get_piece(viewer, (r, c))
                if p is None:
                    x[26, r, c] = 1.0  # 无棋子
                elif reveal_all or p.owner == viewer:
                    # 己方棋子
                    piece_idx = int(p.pid) - 1  # PieceID从1开始，数组从0开始
                    if 0 <= piece_idx < 12:
                        x[piece_idx, r, c] = 1.0
                    else:
                        print(f"policy异常piece_idx: {piece_idx}")
                elif p.owner == TEAMMATES.get(viewer):
                    # 队友棋子
                    piece_idx = int(p.pid) - 1 + 12  # 队友通道12-23
                    if 12 <= piece_idx < 24:
                        x[piece_idx, r, c] = 1.0
                    else:
                        print(f"队友policy异常piece_idx: {piece_idx}")
                else:
                    # 未知敌方棋子
                    if obs[r][c] == int(PieceID.UNKNOWN_ENEMY):
                        # 得到当前玩家在PLAYER_ORDER中的索引
                        current_player_index = PLAYER_ORDER.index(viewer)
                        # 得到上家和下家
                        upper_player = PLAYER_ORDER[(current_player_index - 1) % len(PLAYER_ORDER)]
                        lower_player = PLAYER_ORDER[(current_player_index + 1) % len(PLAYER_ORDER)]
                        if p.owner == upper_player:  # 
                            x[24, r, c] = 1.0
                        elif p.owner == lower_player:  # 
                            x[25, r, c] = 1.0
                    else:
                        x[26, r, c] = 1.0  # 无棋子
    except Exception as e:
        print(f"encode_board_planes异常: {e}") 
    
    return torch.from_numpy(x)  # (30,H,W)

def stack_history(history: Deque[torch.Tensor], current_30chw: torch.Tensor, steps: int = HISTORY_STEPS) -> torch.Tensor:
    while len(history) >= steps:
        history.popleft()
    history.append(current_30chw)
    H, W = current_30chw.shape[-2:]
    frames = list(history)
    if len(frames) < steps:
        pad = [torch.zeros_like(current_30chw) for _ in range(steps - len(frames))]
        frames = pad + frames
    return torch.cat(frames, dim=0)  # (steps*30, H, W)

class SharedPolicy:
    def __init__(self, net=None,device='cpu'):
        self.net = PolicyNet().to(device) if net is None else net
        self.device = device
        # self.net.eval()
        self._hist: Dict[Player, Deque[torch.Tensor]] = {
            Player.ORANGE: deque(maxlen=HISTORY_STEPS),
            Player.PURPLE: deque(maxlen=HISTORY_STEPS),
            Player.GREEN: deque(maxlen=HISTORY_STEPS),
            Player.BLUE: deque(maxlen=HISTORY_STEPS),
        }

    def reset_history(self):
        for k in list(self._hist.keys()):
            self._hist[k].clear()

    def load(self, path: str):
        sd = torch.load(path, map_location=self.device)
        self.net.load_state_dict(sd, strict=True)



    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def _encode_obs(self, board, viewer, side_to_move, no_battle_ratio: float, is_deploy: bool):
        planes30 = encode_board_planes(board, viewer, reveal_all=False, is_deploy=is_deploy)  # (30,H,W)
        
        # 四国军棋不需要旋转，因为board.observe已经处理了视角旋转
        hist = stack_history(self._hist[viewer], planes30)  # (600,H,W)
        H, W = BOARD_H, BOARD_W
        
        # 状态信息
        s_side = torch.full((1,H,W), 1.0 if side_to_move == viewer else 0.0, dtype=torch.float32)
        s_ratio = torch.full((1,H,W), float(no_battle_ratio), dtype=torch.float32)
        s_deploy = torch.full((1,H,W), 1.0 if is_deploy else 0.0, dtype=torch.float32)
        state = torch.cat([s_side, s_ratio, s_deploy], dim=0)  # (3,H,W)
        
        # 特殊区域信息
        special_areas = torch.zeros((SPECIAL_AREA_C, H, W), dtype=torch.float32)
        try:
            for r in range(H):
                for c in range(W):
                    area = board.get_special_area((r, c))
                    if area == SpecialArea.HEADQUARTERS:
                        special_areas[0, r, c] = 1.0
                    elif area == SpecialArea.CAMP:
                        special_areas[1, r, c] = 1.0
                    elif area == SpecialArea.FORBIDDEN:
                        special_areas[2, r, c] = 1.0
        except Exception as e:
            print(f"encode_board_planes异常: {e}") 
        
        # 预留通道
        reserved = torch.zeros((RESERVED_EXTRA_C, H, W), dtype=torch.float32)
        
        x = torch.cat([hist, state, special_areas, reserved], dim=0)  # (613,H,W)
        return x

    def select_deploy(self, board, viewer, piece_to_place, no_battle_ratio: float = 0.0, temperature: float = 1.0):
        x = self._encode_obs(board, viewer, side_to_move=viewer, no_battle_ratio=no_battle_ratio, is_deploy=True).unsqueeze(0).to(self.device)
        out = self.net(x)  # extras default zeros
        cell_mask = torch.zeros((HW,), dtype=torch.float32, device=self.device)
        
        # 检查合法部署位置
        try:
            for r in range(BOARD_H):
                for c in range(BOARD_W):
                    ok = board.can_place(viewer, piece_to_place, (r, c))
                    # if ok:
                        # print(f"can_place: viewer={viewer}, piece_to_place={piece_to_place}, (r, c)={(r, c)}")
                    cell_mask[r*BOARD_W+c] = 1.0 if ok else 0.0
        except Exception as e  :
            print(f"select_deploy异常: {e}") 
            cell_mask[:] = 1.0
        
        lc = out['deploy_cell_logits'].squeeze(0) / max(1e-6, temperature)
        pc_tensor = masked_softmax(lc, cell_mask, dim=0)
        pc = pc_tensor.cpu().detach().numpy()
        
        if pc.sum() <= 0:
            sel_idx = int((cell_mask > 0).nonzero()[0]) if (cell_mask > 0).any() else 0
        else:
            sel_idx = int(np.random.choice(HW, p=pc))
        
        rr, cc = divmod(sel_idx, BOARD_W)
        return (rr, cc), pc_tensor

    def select_move(self, board, viewer, side_to_move, no_battle_ratio: float, temperature: float = 1.0):
        x = self._encode_obs(board, viewer, side_to_move=side_to_move, no_battle_ratio=no_battle_ratio, is_deploy=False).unsqueeze(0).to(self.device)
        out1 = self.net(x)
        
        # 选择棋子
        select_mask = torch.zeros((BOARD_H, BOARD_W), dtype=torch.float32, device=self.device)
        try:
            for r in range(BOARD_H):
                for c in range(BOARD_W):
                    p = board.get_piece(viewer, (r, c))
                    if not p or p.owner != viewer or not p.can_move():
                        continue
                    # 检查是否有合法移动
                    for rr in range(BOARD_H):
                        for cc in range(BOARD_W):
                            if board.can_move_from_to(viewer, (r, c), (rr, cc)):
                                select_mask[r, c] = 1.0
                                break
        except Exception as e:
            print(f"select_move异常: {e}") 
            select_mask[:] = 1.0
        
        ls = out1['select_piece_logits'].squeeze(0) / max(1e-6, temperature)
        ps_tensor = masked_softmax(ls, select_mask.flatten(), dim=0)
        ps = ps_tensor.cpu().detach().numpy()
        
        if ps.sum() <= 0:
            idx = int(select_mask.flatten().argmax().item())
        else:
            # print(f"ls: {ls}")
            # print(f"select_mask: {select_mask}")
            # print(f"ps: {ps}")
            idx = int(np.random.choice(HW, p=ps))
        
        rr_sel, cc_sel = divmod(idx, BOARD_W)
        src = (rr_sel, cc_sel)

        # 选择目标位置
        sel_onehot = torch.zeros((1, 1, BOARD_H, BOARD_W), dtype=torch.float32, device=self.device)
        sel_onehot[:, 0, rr_sel, cc_sel] = 1.0
        out2 = self.net(x, target_extra=sel_onehot)

        target_mask = torch.zeros((BOARD_H, BOARD_W), dtype=torch.float32, device=self.device)
        try:

            for rr in range(BOARD_H):
                for cc in range(BOARD_W):
                    if board.can_move_from_to(viewer, src, (rr, cc)):
                        target_mask[rr, cc] = 1.0

        except Exception as e:
            print(f"select_move异常: {e}") 
            target_mask[:] = 1.0
        
        lt = out2['target_cell_logits'].squeeze(0) / max(1e-6, temperature)
        pt_tensor = masked_softmax(lt, target_mask.flatten(), dim=0)
        pt = pt_tensor.cpu().detach().numpy()
        
        if pt.sum() <= 0:
            j = int(target_mask.flatten().argmax().item())
        else:
            j = int(np.random.choice(HW, p=pt))
        
        rr_t, cc_t = divmod(j, BOARD_W)
        dst = (rr_t, cc_t)
        
        return src, dst, ps_tensor, pt_tensor
