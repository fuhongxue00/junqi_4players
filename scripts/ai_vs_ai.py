#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, time, random
from minijunqi.constants import Player, PieceID, BOARD_W, PLAYER_ORDER
from minijunqi.game import Game, GameConfig
from minijunqi.render import ascii_board, save_four_player_views
from minijunqi.replay import ReplayLogger
from minijunqi.ai.agent import Agent
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=1)
    ap.add_argument('--ckpt_orange', type=str, default=None)
    ap.add_argument('--ckpt_purple', type=str, default=None)
    ap.add_argument('--ckpt_green', type=str, default=None)
    ap.add_argument('--ckpt_blue', type=str, default=None)
    ap.add_argument('--ckpt_og_team', type=str, default=None)
    ap.add_argument('--ckpt_pb_team', type=str, default=None)
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--sleep', type=float, default=0)
    ap.add_argument('--replay_out', type=str, default='replays/ai_vs_ai.json')
    ap.add_argument('--renders', type=str, default='renders')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.replay_out) or '.', exist_ok=True)
    os.makedirs(args.renders, exist_ok=True)

    # game = Game(GameConfig())
    logger = ReplayLogger()
    logger.set_players('AI_O', 'AI_P', 'AI_G', 'AI_B')


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # 创建四个玩家的Agent
    agents = {}
    for player in PLAYER_ORDER:
        agents[player] = Agent(device=device)
    
    # 加载检查点
    if args.ckpt_orange: agents[Player.ORANGE].load(args.ckpt_orange)
    if args.ckpt_purple: agents[Player.PURPLE].load(args.ckpt_purple)
    if args.ckpt_green: agents[Player.GREEN].load(args.ckpt_green)
    if args.ckpt_blue: agents[Player.BLUE].load(args.ckpt_blue)
    if args.ckpt_og_team: 
        agents[Player.ORANGE].load(args.ckpt_og_team)
        agents[Player.GREEN].load(args.ckpt_og_team)
    if args.ckpt_pb_team: 
        agents[Player.PURPLE].load(args.ckpt_pb_team)
        agents[Player.BLUE].load(args.ckpt_pb_team)

    # 统计信息
    team_wins = {'orange_green': 0, 'purple_blue': 0}
    global_draw = 0
    global_sum_stepnum = 0
    attack_stats = {player: {'attack': 0, 'beat': 0, 'lose': 0} for player in PLAYER_ORDER}
    
    for i in range(args.count):
        game = Game(GameConfig())
        for agent in agents.values():
            agent.reset()
        
        # 部署阶段
        while game.state.phase == 'deploy':
            player = game.state.turn
            agent = agents[player]
            piece, rc, _ = agent.select_deploy(game, player)
            ok = game.deploy(player, piece, rc)
            if ok: 
                if args.count == 1:
                    logger.log_deploy(player, piece, rc)
            
            # 可视化
            time.sleep(args.sleep)
            if args.count == 1:
                print(ascii_board(game.state.board, viewer=Player.ORANGE, reveal_all=True, is_deploy=True))
                save_four_player_views(game.state.board, out_dir=args.renders, stem='board_latest', is_deploy=True)
        
        print('部署完毕，开始对局。')
        for agent in agents.values():
            agent.reset()
        
        turn_idx = 0
        while not game.is_over():
            if args.step: 
                input('回车走一步...')
            time.sleep(args.sleep)
            
            player = game.state.turn
            agent = agents[player]
            src, dst, _, _ = agent.select_move(game, player)
            ev = game.step(src, dst)
            
            # 统计攻击信息
            if ev['type'] == 'capture':
                attack_stats[player]['attack'] += 1
                if ev['result'] == 'attacker':
                    attack_stats[player]['beat'] += 1
                elif ev['result'] == 'defender':
                    attack_stats[player]['lose'] += 1
            
            
            turn_idx += 1
            
            if args.count == 1:   
                logger.log_move(turn_idx, player, src, dst, ev)
                print(ascii_board(game.state.board, viewer=Player.ORANGE, reveal_all=True))
                save_four_player_views(game.state.board, out_dir=args.renders, stem='board_latest')
        # 记录结果
        if args.count == 1:
            logger.set_outcome(game.state.winner, game.state.winning_team, game.state.end_reason)
            logger.save(args.replay_out)
        
        print('对局结束：', game.state.end_reason, 'winner=', game.state.winner, 'winning_team=', game.state.winning_team, '总步数:', turn_idx)
        
        if game.state.winner is not None:
            if game.state.winning_team and Player.ORANGE in game.state.winning_team:
                team_wins['orange_green'] += 1
            elif game.state.winning_team and Player.PURPLE in game.state.winning_team:
                team_wins['purple_blue'] += 1
        else:
            global_draw += 1
        
        global_sum_stepnum += turn_idx
    
    # 输出统计结果
    print('===========\n')
    print('橙绿队伍获胜场次:', team_wins['orange_green'])
    print('紫蓝队伍获胜场次:', team_wins['purple_blue'])
    print('平局场次:', global_draw)
    print('场均总步数:', global_sum_stepnum / args.count)
    
    for player in PLAYER_ORDER:
        stats = attack_stats[player]
        print(f'{player.name} 场均主动吃子次数:', stats['attack'] / args.count)
        print(f'{player.name} 场均吃子成功:', stats['beat'] / args.count)
        print(f'{player.name} 场均送子次数:', stats['lose'] / args.count)

if __name__ == '__main__': 
    main()