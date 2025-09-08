
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from .constants import PieceID, Player

@dataclass
class Piece:
    pid: PieceID
    owner: Player
    revealed: bool = True
    def can_move(self) -> bool:
        # 军旗和地雷不能移动
        return self.pid not in (PieceID.JUNQI, PieceID.DILEI)
    def __str__(self) -> str:
        return f"{self.owner.name}:{self.pid.name}"
