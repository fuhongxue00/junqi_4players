
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import os
from .constants import Player, PieceID, BOARD_H, BOARD_W, SpecialArea
from .board import Board

def _load_font(size=20):
    # 直接用 WQY Micro Hei 的文件路径（Ubuntu）
    path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        # 兜底：再试环境变量；最后退回默认字体
        env_font = os.environ.get("MINIJUNQI_FONT")
        if env_font:
            return ImageFont.truetype(env_font, size)
        return ImageFont.load_default()

PIECE_SHORT = {
    PieceID.JUNQI: '旗',
    PieceID.SILING: '司',
    PieceID.JUNZHANG: '军',
    PieceID.SHIZHANG: '师',
    PieceID.LVZHANG: '旅',
    PieceID.TUANZHANG: '团',
    PieceID.YINGZHANG: '营',
    PieceID.LIANZHANG: '连',
    PieceID.PAIZHANG: '排',
    PieceID.GONGBING: '工',
    PieceID.DILEI: '雷',
    PieceID.ZHADAN: '炸',
    PieceID.UNKNOWN_ENEMY: '?'
}

# 玩家颜色
PLAYER_COLORS = {
    Player.ORANGE: (216, 108, 0),    # 橙色
    Player.PURPLE: (128, 0, 128),    # 紫色
    Player.GREEN: (0, 128, 0),       # 绿色
    Player.BLUE: (0, 0, 255),        # 蓝色
}

def ascii_board(board: Board, viewer: Player, reveal_all: bool=False, is_deploy: bool=False) -> str:
    obs = board.observe(viewer, reveal_all=reveal_all, hide_enemy_positions=is_deploy and not reveal_all)
    rows = []
    for r in range(BOARD_H):
        row = []
        for c in range(BOARD_W):
            v = PieceID(obs[r][c])
            if v == PieceID.EMPTY:
                # 显示特殊区域
                area = board.get_special_area((r, c))
                if area == SpecialArea.HEADQUARTERS:
                    row.append('  HQ')
                elif area == SpecialArea.CAMP:
                    row.append('  CP')
                elif area == SpecialArea.FORBIDDEN:
                    row.append('    ')
                else:
                    row.append('  .')
            elif v == PieceID.UNKNOWN_ENEMY:
                row.append('  ?')
            else:
                ch = PIECE_SHORT.get(v, '??')
                p = board.get_piece(viewer, (r, c))
                if p:
                    owner_initial = p.owner.name[:1]  # 取玩家名字首字母
                    row.append(owner_initial + ch)
                else:
                    row.append(' ' + ch)
        rows.append(' '.join(row))
    header = f"视角: {viewer.name}  (reveal_all={reveal_all})"
    return header + "\n" + '\n'.join(rows)

def save_image(board: Board, path: str, viewer: Player, reveal_all: bool=False, title: Optional[str]=None, is_deploy: bool=False):
    cell = 32  # 减小格子大小以适应17x17棋盘
    pad = 20
    w = BOARD_W * cell + pad * 2
    h = BOARD_H * cell + pad * 2 + 30
    img = Image.new('RGB', (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # 绘制网格
    for i in range(BOARD_H + 1):
        y = pad + i * cell
        draw.line((pad, y, pad + BOARD_W * cell, y), fill=(0, 0, 0), width=1)
    for j in range(BOARD_W + 1):
        x = pad + j * cell
        draw.line((x, pad, x, pad + BOARD_H * cell), fill=(0, 0, 0), width=1)
    
    # 绘制特殊区域背景
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            area = board.get_special_area((r, c))
            x0 = pad + c * cell
            y0 = pad + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            
            if area == SpecialArea.HEADQUARTERS:
                draw.rectangle((x0, y0, x1, y1), fill=(255, 200, 200), outline=(255, 0, 0), width=1)
            elif area == SpecialArea.CAMP:
                draw.rectangle((x0, y0, x1, y1), fill=(220, 255, 220), outline=(0, 255, 0), width=1)
            elif area == SpecialArea.FORBIDDEN:
                draw.rectangle((x0, y0, x1, y1), fill=(20, 20, 20), outline=(0, 0, 0), width=1)
            else:
                draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0), width=1)

    # 绘制棋子
    obs = board.observe(viewer, reveal_all=reveal_all, hide_enemy_positions=is_deploy and not reveal_all)
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            v = PieceID(obs[r][c])
            if v == PieceID.EMPTY: 
                continue
                
            x0 = pad + c * cell
            y0 = pad + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            
            p = board.get_piece(viewer, (r, c))
            if p:
                color = PLAYER_COLORS.get(p.owner, (128, 128, 128))
                draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=(0, 0, 0), width=1, fill=color)
                
                ch = PIECE_SHORT.get(v, '?')
                try:
                    font = _load_font(16)  # 减小字体大小
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), ch, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2), ch, fill=(255, 255, 255), font=font)
            elif v == PieceID.UNKNOWN_ENEMY:
                draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=(0, 0, 0), width=1, fill=(200, 200, 200))
                try:
                    font = _load_font(16)
                except:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), '?', font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2), '?', fill=(0, 0, 0), font=font)
    
    # 标题
    t = title or f"{viewer.name} view (reveal_all={reveal_all})"
    draw.text((pad, pad + BOARD_H * cell + 6), t, fill=(0, 0, 0))
    img.save(path)
    return path

def save_four_player_views(board: Board, out_dir: str = '.', stem: str = 'board_latest', is_deploy: bool=False):
    """保存四个玩家的视角"""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    
    # 全信息视角
    paths['all'] = save_image(board, os.path.join(out_dir, f"{stem}_all.png"), 
                             viewer=Player.ORANGE, reveal_all=True, 
                             is_deploy=is_deploy, title='All Info')
    
    # 四个玩家视角
    for player in Player:
        player_name = player.name.lower()
        paths[player_name] = save_image(board, os.path.join(out_dir, f"{stem}_{player_name}.png"), 
                                       viewer=player, reveal_all=False, 
                                       is_deploy=is_deploy, title=f'{player.name} View')
    
    return paths

def save_triple_latest(board: Board, out_dir: str = '.', stem: str = 'board_latest', is_deploy: bool=False):
    """保持向后兼容，但使用四国视角"""
    return save_four_player_views(board, out_dir, stem, is_deploy)
