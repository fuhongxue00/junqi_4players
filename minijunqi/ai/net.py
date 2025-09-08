
# # -*- coding: utf-8 -*-
# """极小UNet骨干 + 多头策略（部署/选子/方向）+ 价值头。
# 输入通道：10个棋子ID（含EMPTY与UNKNOWN_ENEMY）+ side_to_move + no_battle_ratio + is_deploy_phase = 13
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ..constants import BOARD_H, BOARD_W

# HW = BOARD_H * BOARD_W

# class ConvBNAct(nn.Module):
#     def __init__(self, c_in, c_out, k=3):
#         super().__init__()
#         self.cv = nn.Conv2d(c_in, c_out, k, padding=k//2)
#         self.bn = nn.BatchNorm2d(c_out)
#         self.act = nn.ReLU(inplace=True)
#     def forward(self, x):
#         return self.act(self.bn(self.cv(x)))

# class TinyUNet(nn.Module):
#     def __init__(self, c_in=13, base=16):
#         super().__init__()
#         self.down1 = nn.Sequential(ConvBNAct(c_in, base), ConvBNAct(base, base))
#         self.pool1 = nn.MaxPool2d(2)  # 6x6 -> 3x3
#         self.mid = nn.Sequential(ConvBNAct(base, base*2), ConvBNAct(base*2, base*2))
#         self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)  # 3x3 -> 6x6
#         self.fuse = ConvBNAct(base*2, base)
#         self.head_deploy_cell = nn.Linear(base*BOARD_H*BOARD_W, HW)
#         self.head_select = nn.Linear(base*BOARD_H*BOARD_W, HW)
#         self.head_dir = nn.Linear(base*BOARD_H*BOARD_W, 4)
#         self.head_value = nn.Linear(base*BOARD_H*BOARD_W, 1)
#     def forward(self, x):
#         d1 = self.down1(x)
#         m = self.mid(self.pool1(d1))
#         up = self.up(m)
#         if up.shape[-2:] != d1.shape[-2:]:
#             up = F.interpolate(up, size=d1.shape[-2:], mode='nearest')
#         f = self.fuse(torch.cat([d1, up], dim=1))
#         flat = f.flatten(1)
#         return {
#             'deploy_cell_logits': self.head_deploy_cell(flat),
#             'select_piece_logits': self.head_select(flat),
#             'move_dir_logits': self.head_dir(flat),
#             'value': self.head_value(flat).squeeze(-1)
#         }


# -*- coding: utf-8 -*-
"""
TinyUNetV2 trunk (added depth) + three per-cell UNet heads (deploy/select/target) + value head.
Input channels: HISTORY_C (600) + 3 state channels + 3 special area channels + 7 reserved zeros = 613 by default.
Heads accept extra channels; for target head, channel 0 is the selected-piece one-hot map.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ..constants import BOARD_H, BOARD_W
except Exception:
    # Fallback in case constants path differs; default to 17x17 for four-country chess
    BOARD_H, BOARD_W = 17, 17

HW = BOARD_H * BOARD_W

# ==== input channel config ====
HISTORY_STEPS = 20              # number of past frames to stack (increased for four-country)
PIECE_PLANES_PER_FRAME = 30     # 己方12种+队友12种+未知上家+未知下家+无棋子+预留3个 = 30
STATE_EXTRA_C = 3               # side_to_move, no_battle_ratio, is_deploy_phase
SPECIAL_AREA_C = 3              # 大本营、行营、禁入区的onehot编码
RESERVED_EXTRA_C = 7            # reserved zeros
C_IN_DEFAULT = HISTORY_STEPS * PIECE_PLANES_PER_FRAME + STATE_EXTRA_C + SPECIAL_AREA_C + RESERVED_EXTRA_C  # 600 + 3 + 3 + 7 = 613

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.cv = nn.Conv2d(c_in, c_out, k, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.cv(x)))

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c_in, c_out, k),
            ConvBNAct(c_out, c_out, k),
        )
    def forward(self, x):
        return self.block(x)

class TinyUNetV2(nn.Module):
    """
    Enhanced UNet for 17x17 four-country chess; channels follow base -> base*2 -> base*4.
    Added more layers and increased base channels for larger board.
    Returns spatial feature map (base, H, W) and a scalar value head.
    """
    def __init__(self, c_in: int = C_IN_DEFAULT, base: int = 32):  # Increased base channels
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.down1 = DoubleConv(c_in, c1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 17x17 -> 8x8
        self.down2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        self.mid1 = DoubleConv(c2, c3)
        self.mid2 = DoubleConv(c3, c3)  # **added layer**
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)  # 4x4 -> 8x8
        self.fuse2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)  # 8x8 -> 17x17
        self.fuse1 = DoubleConv(c1 + c1, c1)
        # value head with more capacity
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c1 * BOARD_H * BOARD_W, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, x):
        d1 = self.down1(x)              # (B, c1, 17, 17)
        m1 = self.pool1(d1)             # (B, c1, 8, 8)
        d2 = self.down2(m1)             # (B, c2, 8, 8)
        m2 = self.pool2(d2)             # (B, c2, 4, 4)
        m = self.mid1(m2)               # (B, c3, 4, 4)
        m = self.mid2(m)                # (B, c3, 4, 4)
        
        up2 = self.up2(m)               # (B, c2, 8, 8)
        if up2.shape[-2:] != d2.shape[-2:]:
            up2 = F.interpolate(up2, size=d2.shape[-2:], mode='nearest')
        f2 = self.fuse2(torch.cat([d2, up2], dim=1))  # (B, c2, 8, 8)
        
        up1 = self.up1(f2)              # (B, c1, 17, 17)
        if up1.shape[-2:] != d1.shape[-2:]:
            up1 = F.interpolate(up1, size=d1.shape[-2:], mode='nearest')
        f1 = self.fuse1(torch.cat([d1, up1], dim=1))  # (B, c1, 17, 17)
        
        v = self.value_head(f1).squeeze(-1)           # (B,)
        return f1, v

class HeadUNetSmall(nn.Module):
    """
    Enhanced UNet head for 17x17 four-country chess that maps trunk feature map (+ optional extra channels)
    to a single-channel (logits) HxW map via a 1x1 conv at the end.
    """
    def __init__(self, trunk_c: int, extra_c: int = 0, base: int = 16):  # Increased base channels
        super().__init__()
        c_in = trunk_c + extra_c
        c1, c2, c3 = base, base * 2, base * 4
        self.down1 = DoubleConv(c_in, c1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 17x17 -> 8x8
        self.down2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        self.mid = DoubleConv(c2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)  # 4x4 -> 8x8
        self.fuse2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)  # 8x8 -> 17x17
        self.fuse1 = DoubleConv(c1 + c1, c1)
        self.out1x1 = nn.Conv2d(c1, 1, kernel_size=1)
    
    def forward(self, trunk_f: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """
        trunk_f: (B, trunk_c, H, W)
        extra:   (B, extra_c, H, W) or None
        returns: (B, H, W) logits
        """
        if extra is None:
            B, _, H, W = trunk_f.shape
            extra = torch.zeros((B, 0, H, W), dtype=trunk_f.dtype, device=trunk_f.device)
        x = torch.cat([trunk_f, extra], dim=1)
        
        d1 = self.down1(x)              # (B, c1, 17, 17)
        m1 = self.pool1(d1)             # (B, c1, 8, 8)
        d2 = self.down2(m1)             # (B, c2, 8, 8)
        m2 = self.pool2(d2)             # (B, c2, 4, 4)
        m = self.mid(m2)                # (B, c3, 4, 4)
        
        up2 = self.up2(m)               # (B, c2, 8, 8)
        if up2.shape[-2:] != d2.shape[-2:]:
            up2 = F.interpolate(up2, size=d2.shape[-2:], mode='nearest')
        f2 = self.fuse2(torch.cat([d2, up2], dim=1))  # (B, c2, 8, 8)
        
        up1 = self.up1(f2)              # (B, c1, 17, 17)
        if up1.shape[-2:] != d1.shape[-2:]:
            up1 = F.interpolate(up1, size=d1.shape[-2:], mode='nearest')
        f1 = self.fuse1(torch.cat([d1, up1], dim=1))  # (B, c1, 17, 17)
        
        logits = self.out1x1(f1).squeeze(1)
        return logits  # (B, H, W)

class PolicyNet(nn.Module):
    """
    Full network = trunk + three per-cell heads + scalar value head for four-country chess.
    """
    def __init__(
        self,
        c_in: int = C_IN_DEFAULT,
        trunk_base: int = 32,  # Increased for four-country chess
        head_base: int = 16,   # Increased for four-country chess
        deploy_extra_c: int = 4,
        select_extra_c: int = 4,
        target_extra_c: int = 4,  # channel 0 reserved for selected-piece map
    ):
        super().__init__()
        self.trunk = TinyUNetV2(c_in=c_in, base=trunk_base)
        trunk_c = trunk_base
        self.deploy_head = HeadUNetSmall(trunk_c, extra_c=deploy_extra_c, base=head_base)
        self.select_head = HeadUNetSmall(trunk_c, extra_c=select_extra_c, base=head_base)
        self.target_head = HeadUNetSmall(trunk_c, extra_c=target_extra_c, base=head_base)
        self.deploy_extra_c = deploy_extra_c
        self.select_extra_c = select_extra_c
        self.target_extra_c = target_extra_c

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
        deploy_extra: torch.Tensor | None = None,
        select_extra: torch.Tensor | None = None,
        target_extra: torch.Tensor | None = None,
    ):
        trunk_f, value = self.trunk(x)  # (B, trunk_c, H, W), (B,)
        B, _, H, W = trunk_f.shape
        def ensure_extra(e: torch.Tensor | None, c: int):
            if e is None:
                return torch.zeros((B, c, H, W), dtype=trunk_f.dtype, device=trunk_f.device)
            elif e.shape[1] < c:
                # 通道数不足，右侧补零
                pad = torch.zeros((B, c - e.shape[1], H, W), dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=1)
            elif e.shape[1] > c:
                # 通道数多余，截断
                e = e[:, :c]
                print("net通道截断警告")
            return e
        deploy_extra = ensure_extra(deploy_extra, self.deploy_extra_c)
        select_extra = ensure_extra(select_extra, self.select_extra_c)
        target_extra = ensure_extra(target_extra, self.target_extra_c)
        dep_logits_hw = self.deploy_head(trunk_f, deploy_extra)
        sel_logits_hw = self.select_head(trunk_f, select_extra)
        tgt_logits_hw = self.target_head(trunk_f, target_extra)
        # flatten H*W for convenience, callers may reshape back
        dep_logits = dep_logits_hw.flatten(1)  # (B, HW)
        sel_logits = sel_logits_hw.flatten(1)  # (B, HW)
        tgt_logits = tgt_logits_hw.flatten(1)  # (B, HW)
        return {
            'deploy_cell_logits': dep_logits,
            'select_piece_logits': sel_logits,
            'target_cell_logits': tgt_logits,
            'value': value,
        }
