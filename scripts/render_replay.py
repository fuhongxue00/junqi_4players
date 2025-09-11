
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, json, time
from minijunqi.constants import Player, PieceID, PLAYER_ORDER
from minijunqi.game import Game, GameConfig
from minijunqi.render import ascii_board, save_four_player_views
from minijunqi.ai.agent import Agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=str, required=True)
    ap.add_argument('--renders', type=str, default='renders/replay_steps')
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--sleep', type=float, default=0)
    args = ap.parse_args()
    os.makedirs(args.renders, exist_ok=True)
    
    # 加载replay数据
    with open(args.replay, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建Game实例和Agent实例
    game = Game(GameConfig())
    agents = {}
    for player in PLAYER_ORDER:
        agents[player] = Agent(device='cpu')
        agents[player].reset()
    
    # 重建初始部署状态
    print('重建初始部署状态...')
    for side in ['ORANGE', 'PURPLE', 'GREEN', 'BLUE']:
        if side in data['initial_deployments']:
            player = Player[side]
            for item in data['initial_deployments'][side]:
                pid = PieceID[item['piece']]
                r, c = item['pos']
                # 使用game的deploy方法来部署棋子
                game.deploy(player, pid, (r, c))
    
    print('部署完成。')
    print(ascii_board(game.state.board, Player.ORANGE, reveal_all=True))
    save_four_player_views(game.state.board, out_dir=args.renders, stem='replay_deploy')
    
    # 重放对局步骤
    print('开始重放对局...')
    t = 1
    for mv in data['moves']:
        if args.step: 
            input('回车下一手...')
        time.sleep(args.sleep)
        
        player = Player[mv['player']]
        src = tuple(mv['from'])
        dst = tuple(mv['to'])
        
        # 使用game的step方法来执行移动
        ev = game.step(src, dst)
        
        # 获取全局坐标用于渲染
        global_src = game.state.board.rotate_coord_to_global(player, src)
        global_dst = game.state.board.rotate_coord_to_global(player, dst)
        
        print(ascii_board(game.state.board, player, reveal_all=True))
        save_four_player_views(game.state.board, out_dir=args.renders, stem=f'replay_step_', src=global_src, dst=global_dst)
        # save_four_player_views(game.state.board, out_dir=args.renders, stem=f'replay_step_{t}', src=global_src, dst=global_dst)
        t += 1
    
    print('复盘结束。')
if __name__ == '__main__': main()
