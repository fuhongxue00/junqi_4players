
# -*- coding: utf-8 -*-
import argparse, random
import torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from ..constants import Player, PieceID, BOARD_H, BOARD_W, DEFAULT_NO_BATTLE_DRAW, INITIAL_POOL, PLAYER_ORDER, TEAMMATES
from ..game import Game, GameConfig
from ..render import ascii_board
from .net import PolicyNet
from .policy import SharedPolicy
from .agent import Agent

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# def random_deploy(g: Game, player: Player):
#     cells=[(r,c) for r in g.state.board.home_rows(player) for c in range(BOARD_W)]
#     random.shuffle(cells)
#     # flag first
#     flag=None
#     for rc in cells:
#         if g.state.board.can_place(player, PieceID.FLAG, rc): flag=rc; break
#     g.deploy(player, PieceID.FLAG, flag)
#     pool=dict(INITIAL_POOL); pool[PieceID.FLAG]-=1
#     for pid,cnt in pool.items():
#         for _ in range(cnt):
#             for rc in cells:
#                 if g.state.board.can_place(player, pid, rc):
#                     g.deploy(player, pid, rc); break

pid_to_value = {
    PieceID.JUNQI: 150,
    PieceID.SILING: 50,
    PieceID.JUNZHANG: 40,
    PieceID.SHIZHANG: 30,
    PieceID.LVZHANG: 25,
    PieceID.TUANZHANG: 23,
    PieceID.YINGZHANG: 21,
    PieceID.LIANZHANG: 19,
    PieceID.PAIZHANG: 18,
    PieceID.GONGBING: 20,
    PieceID.DILEI: 18,
    PieceID.ZHADAN: 35,
}

def play_episode(net: PolicyNet, device: str,no_grad_net = None):
    traj = []  # (logp, player, information)
    g = Game(GameConfig())
    
    # 创建四个玩家的Agent
    agents = {}
    for player in PLAYER_ORDER:
        if player == Player.ORANGE or player == Player.GREEN:
            agents[player] = Agent(net=net, device=device,temperature=2)
        else:
            agents[player] = Agent(device=device,net=no_grad_net,temperature=2)
        agents[player].reset()
    
    # 部署阶段
    while g.state.phase == "deploy":
        player = g.state.turn
        play_agent = agents[player]
        pid, rc, pc = play_agent.select_deploy(g, player)
        r, c = rc
        idx = r * BOARD_W + c
        logp = torch.log(pc[idx] + 1e-9)
        information = {'phase':'deploy'}
        traj.append((logp, player, information))
        ev = g.deploy(player, pid, rc)
    
    # 重置所有Agent的历史
    for agent in agents.values():
        agent.reset()
    
    # 对战阶段
    while not g.is_over():
        player = g.state.turn

        play_agent = agents[player]
        # 起点，终点，起点概率张量，终点概率张量
        src, dst, ps, pt = play_agent.select_move(g, player)
        s_r, s_c = src
        src_idx = s_r * BOARD_W + s_c
        t_r, t_c = dst
        dst_idx = t_r * BOARD_W + t_c
        logp = torch.log(ps[src_idx] + 1e-9) + torch.log(pt[dst_idx] + 1e-9)
        
        ev = g.step(src, dst)
        information = {'phase':'play','ev':ev,'dst':dst}
        traj.append((logp, player, information))
        # print(ascii_board(g.state.board, viewer=player, reveal_all=True, is_deploy=False))
    
    # 计算奖励
    if g.state.winner is not None:
        pass
    elif g.state.end_reason == 'draw':
        pass
    else:
        print(f"end_reason:{g.state.end_reason}")
        raise ValueError("出现了未知的结束原因")
    
    list_dict_logp_rewards = []
    for index, (logp, player, information) in enumerate(traj):
        reward_onestep = {'player':player,'logp':logp,'win_lose_draw':0,'attack_value':0,'phase':information['phase']}
        
        # 根据胜负关系给出奖励，该奖励无需进一步传播
        if g.state.winner is not None:
            if g.state.winning_team and player in g.state.winning_team:
                if information['phase'] == 'play':
                    reward_onestep['win_lose_draw'] += 50*index / len(traj)
                else:
                    reward_onestep['win_lose_draw'] += 10
            else:
                if information['phase'] == 'play':
                    reward_onestep['win_lose_draw'] -= 50*index / len(traj)
                else:
                    reward_onestep['win_lose_draw'] -= 10
        elif g.state.end_reason == 'draw':
            if information['phase'] == 'play':
                reward_onestep['win_lose_draw'] -= 10*index / len(traj)
            else:
                reward_onestep['win_lose_draw'] -= 2
        
        # 吃子相关奖励
        if information['phase'] == 'play' and information['ev']['type'] == 'capture':
            # 同归于尽或杀入大本营
            if information['ev']['result'] == 'both' or g.state.board.is_headquarters(information['dst']):
                reward_onestep['attack_value'] += pid_to_value[information['ev']['q_pid']]
                reward_onestep['attack_value'] -= 0.9*pid_to_value[information['ev']['p_pid']]            
            elif information['ev']['result'] == 'attacker':
                reward_onestep['attack_value'] += pid_to_value[information['ev']['q_pid']]
            elif information['ev']['result'] == 'defender':
                reward_onestep['attack_value'] -= 0.9*pid_to_value[information['ev']['p_pid']]

        list_dict_logp_rewards.append(reward_onestep)

    returns = []    # (logp, gain)
    temp_capture_backward = {player:0 for player in PLAYER_ORDER}
    # 反向迭代得到gain，其中胜负关系已经考虑步数关系，吃子相关的奖励还需要反向扩散
    for logp_reward in list_dict_logp_rewards[::-1]:
        logp = logp_reward['logp']
        gain = logp_reward['win_lose_draw']
        gain += logp_reward['attack_value']
        if logp_reward['phase'] == 'play':
            the_teammate = TEAMMATES[logp_reward['player']]
            the_nextplayer = PLAYER_ORDER[(PLAYER_ORDER.index(logp_reward['player'])+1)%4]
            the_previousplayer = PLAYER_ORDER[(PLAYER_ORDER.index(logp_reward['player'])-1)%4]
            gain += 0.7*temp_capture_backward[logp_reward['player']]
            gain += 0.3*temp_capture_backward[the_teammate]
            gain -= 0.4*temp_capture_backward[the_nextplayer]
            gain -= 0.2*temp_capture_backward[the_previousplayer]
            temp_capture_backward[logp_reward['player']] += logp_reward['attack_value']
            temp_capture_backward[logp_reward['player']] *= 0.3
        else:
            the_teammate = TEAMMATES[logp_reward['player']]
            the_nextplayer = PLAYER_ORDER[(PLAYER_ORDER.index(logp_reward['player'])+1)%4]
            the_previousplayer = PLAYER_ORDER[(PLAYER_ORDER.index(logp_reward['player'])-1)%4]
            gain += 0.7*temp_capture_backward[logp_reward['player']]
            gain += 0.3*temp_capture_backward[the_teammate]
            gain -= 0.4*temp_capture_backward[the_nextplayer]
            gain -= 0.2*temp_capture_backward[the_previousplayer]
        returns.append((logp, gain))
    return returns

def train(epochs,episodes_per_epoch, out, from_ckpt=None,lr_step=1, lr_gamma=0.5):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('device:',device)
    net = PolicyNet().to(device)
    if from_ckpt :
        net.load_state_dict(torch.load(from_ckpt)) 
    opt=optim.Adam(net.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    for epoch in range(epochs):
        loss = 0.0
        for i in trange(episodes_per_epoch):
            no_grad_net = PolicyNet().to(device)
            if i % 3 == 0:
                print("使用随机模型作为训练对手")
            elif i % 3 == 1:
                no_grad_net=net
                print("自己作为训练对手")
            elif i % 3 == 2:
                no_grad_net.load_state_dict(torch.load("checkpoints/02.pt"))
                print("使用02.pt作为训练对手")
            traj=play_episode(net=net, device=device,no_grad_net=no_grad_net)
            if not traj: continue
            for logp,R in traj:
                # print('logp:',logp.item(),'R:',R)
                loss=loss - logp*R
        loss = loss / episodes_per_epoch
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
            # print('loss:',loss.item())
            # print('lr:',scheduler.get_last_lr()[0])
    import os
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(net.state_dict(), out); print('saved:', out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--episodes_per_epoch', type=int, default=3)
    ap.add_argument('--out', type=str, default='checkpoints/rl.pt')
    ap.add_argument('--from_ckpt',type=str,default=None)
    args=ap.parse_args()
    train(args.epochs,args.episodes_per_epoch, args.out,from_ckpt=args.from_ckpt)
