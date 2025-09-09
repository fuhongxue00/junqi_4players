
# -*- coding: utf-8 -*-
from enum import IntEnum

BOARD_H = 17
BOARD_W = 17

class Player(IntEnum):
    ORANGE = 0  # 橙色 - 主视角，在下方
    PURPLE = 1  # 紫色 - 右侧
    GREEN = 2   # 绿色 - 上方，橙色队友
    BLUE = 3    # 蓝色 - 左侧，紫色队友

class PieceID(IntEnum):
    EMPTY = 0
    JUNQI = 1      # 军旗
    SILING = 2     # 司令
    JUNZHANG = 3   # 军长
    SHIZHANG = 4   # 师长
    LVZHANG = 5    # 旅长
    TUANZHANG = 6  # 团长
    YINGZHANG = 7  # 营长
    LIANZHANG = 8  # 连长
    PAIZHANG = 9   # 排长
    GONGBING = 10  # 工兵
    DILEI = 11     # 地雷
    ZHADAN = 12    # 炸弹
    UNKNOWN_ENEMY = 13  # 用于观测中的未知敌子（棋盘内部不放这个）

# 四国军棋棋子配置
INITIAL_POOL = {
    PieceID.JUNQI: 1,      # 军旗1个
    PieceID.SILING: 1,     # 司令1个
    PieceID.JUNZHANG: 1,   # 军长1个
    PieceID.SHIZHANG: 2,   # 师长2个
    PieceID.LVZHANG: 2,    # 旅长2个
    PieceID.TUANZHANG: 2,  # 团长2个
    PieceID.YINGZHANG: 2,  # 营长2个
    PieceID.LIANZHANG: 3,  # 连长3个
    PieceID.PAIZHANG: 3,   # 排长3个
    PieceID.GONGBING: 3,   # 工兵3个
    PieceID.DILEI: 3,      # 地雷3个
    PieceID.ZHADAN: 2,     # 炸弹2个
}

STRENGTH = {
    PieceID.JUNQI: -10,     # 军旗不能移动
    PieceID.SILING: 7,     # 司令
    PieceID.JUNZHANG: 6,   # 军长
    PieceID.SHIZHANG: 5,   # 师长
    PieceID.LVZHANG: 4,    # 旅长
    PieceID.TUANZHANG: 3,  # 团长
    PieceID.YINGZHANG: 2,  # 营长
    PieceID.LIANZHANG: 1,  # 连长
    PieceID.PAIZHANG: 0,   # 排长
    PieceID.GONGBING: -1,   # 工兵
    PieceID.DILEI: 8,      # 地雷不能移动
    PieceID.ZHADAN: 1,     # 炸弹
}

# 特殊区域类型
class SpecialArea(IntEnum):
    NORMAL = 0      # 普通区域
    HEADQUARTERS = 1  # 大本营
    CAMP = 2        # 行营
    FORBIDDEN = 3   # 禁入区

# 大本营位置（每个玩家左右各一个，在己方最后一行的中心邻格）
HEADQUARTERS_POSITIONS = {
    Player.ORANGE: [(16, 7), (16, 9)],  # 橙色在下方
    Player.PURPLE: [(7, 0), (9, 0)],    # 紫色在右侧
    Player.GREEN: [(0, 7), (0, 9)],     # 绿色在上方
    Player.BLUE: [(7, 16), (9, 16)],    # 蓝色在左侧
}

# 可部署区域：己方视角的最后六行的中间五列
DEPLOY_AREA_ROWS = (11, 12, 13, 14, 15, 16)  # 最后六行
DEPLOY_AREA_COLS = (6, 7, 8, 9, 10)          # 中间五列

DEFAULT_NO_BATTLE_DRAW = 20

# 预定义部署顺序（军旗、地雷、炸弹先放，其余按固定顺序展开多份）
DEPLOY_SEQUENCE = (
    [PieceID.JUNQI] +
    [PieceID.DILEI] +
    [PieceID.ZHADAN] +
    [PieceID.SILING] +
    [PieceID.JUNZHANG] +
    [PieceID.SHIZHANG]*2 +
    [PieceID.LVZHANG]*2 +
    [PieceID.TUANZHANG]*2 +
    [PieceID.YINGZHANG]*2 +
    [PieceID.LIANZHANG]*3 +
    [PieceID.PAIZHANG]*3 +
    [PieceID.GONGBING]*3 +
    [PieceID.DILEI]*2 +  # 剩余2个地雷
    [PieceID.ZHADAN]     # 剩余1个炸弹
)

# 队友关系
TEAMMATES = {
    Player.ORANGE: Player.GREEN,
    Player.GREEN: Player.ORANGE,
    Player.PURPLE: Player.BLUE,
    Player.BLUE: Player.PURPLE,
}

# 玩家顺序（逆时针）
PLAYER_ORDER = [Player.ORANGE, Player.PURPLE, Player.GREEN, Player.BLUE]
