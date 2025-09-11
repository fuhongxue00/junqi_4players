
# 军棋 AI 开发项目

## 目标
   搞出接近甚至超过人类水平的军棋AI

   已有[mini军棋项目](https://github.com/fuhongxue00/minijunqi_project)，初步验证方案在简化版规则下是有效的

## 最新进度
- 规则层面的bug目前已修复，规则目前只差「40死后暴露军棋位置」这个规则实现比较烦、要改game和policy的encode方法
- rl也已跑通，经过重整奖励（胜负奖励正比于当前步数/总步数，吃子奖励基于棋子价值，并会向更早的步数传播，将军旗的价值设为很高），01.pt可以在十局内2胜8和（打随机模型）
- 进一步地、配置在训练时只为Orange和Green的网络载入和保存ckpt（也就是针对对抗随机模型），从01.pt开始训，02.pt能够做到对随机初始化模型，十局内5胜5和。
- 01.pt和02.pt互打,互有胜负。换色测试之后，整体而言02.pt更强
- 在一个epoch内多次更换对手进行训练，累计梯度后统一更新参数，尝试「10epoch，每个epoch内9个episode，其中3个打随机，3个打01，3个打02，其实还可以配一部分打自己」共90个episode，得到03.pt，在pt之间对抗较优，但会因为送地雷而反而成为唯一输给随机模型的模型，需要观察行为。（学习率、step、epoch等参数均被魔改，需要修复屎山）
- 改掉03.pt训练时的bug（其实没混合训练），参数变更：加重了错误行为的惩罚，单次训练得到**04.pt目前全面最强**，打其他ckpt胜率明显高，打随机是4胜6和

## TODO
- 更新replay和render，从而利于分析agent的行为
- 重整奖励规则，让训练更有效
- 优化输入，**加上自回归部分**
- 网络层的大小要优化，现在矩阵是六百多乘32 很畸形
- 优化屎山
- 有些部分复杂度可以优化、持续检查bug
- 看deepnash




## 使用说明

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行对局

1. **AI vs AI**

   ```bash
   python scripts/ai_vs_ai.py --step --ckpt_blue ...
   ```

   支持实时输出和保存复盘文件。

2. **AI vs Human**

   ```bash
   python scripts/ai_vs_human.py 
   ```

   人类通过命令行输入棋子和坐标，AI 自动走子。

3. **AI vs Input**
 尚未实现


4. **复盘渲染**

   ```bash
   python scripts/render_replay.py --replay replays/ai_vs_human.json --step
   ```

   可逐步回放并渲染 PNG 图片。


---

## 训练说明

### 监督学习(暂时无需使用，也并没有调试)


### 强化学习（已跑通且有效）

通过自我博弈方式进行简易训练：

```bash
python -m minijunqi.ai.train_rl --epochs 10 --out checkpoints/rl.pt
```



