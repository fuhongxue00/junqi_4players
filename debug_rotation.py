#!/usr/bin/env python3
"""
调试坐标旋转
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from minijunqi.constants import Player, HEADQUARTERS_POSITIONS

def test_rotation_formulas():
    print("测试旋转公式...")
    
    # 测试紫色玩家的旋转
    print("\n紫色玩家:")
    print("期望: 玩家视角(16,7) -> 全局坐标(7,0)")
    print("期望: 玩家视角(16,9) -> 全局坐标(9,0)")
    
    # 当前公式: (r,c) -> (c, BOARD_H-1-r)
    BOARD_H = 17
    BOARD_W = 17
    
    # 紫色玩家视角坐标
    purple_coord1 = (16, 7)
    purple_coord2 = (16, 9)
    
    # 使用当前公式转换
    global1 = (purple_coord1[1], BOARD_H - 1 - purple_coord1[0])
    global2 = (purple_coord2[1], BOARD_H - 1 - purple_coord2[0])
    
    print(f"当前公式结果: {purple_coord1} -> {global1}")
    print(f"当前公式结果: {purple_coord2} -> {global2}")
    print(f"期望结果: {purple_coord1} -> (7, 0)")
    print(f"期望结果: {purple_coord2} -> (9, 0)")
    
    # 检查是否正确
    expected1 = (7, 0)
    expected2 = (9, 0)
    
    if global1 == expected1 and global2 == expected2:
        print("✓ 紫色玩家旋转公式正确")
    else:
        print("✗ 紫色玩家旋转公式错误")
        print(f"需要修正: {global1} -> {expected1}, {global2} -> {expected2}")
    
    # 测试逆旋转
    print("\n测试逆旋转:")
    back1 = (global1[1], BOARD_H - 1 - global1[0])
    back2 = (global2[1], BOARD_H - 1 - global2[0])
    
    print(f"逆旋转: {global1} -> {back1}")
    print(f"逆旋转: {global2} -> {back2}")
    
    if back1 == purple_coord1 and back2 == purple_coord2:
        print("✓ 逆旋转公式正确")
    else:
        print("✗ 逆旋转公式错误")

if __name__ == '__main__':
    test_rotation_formulas()
