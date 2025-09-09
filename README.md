
# 军棋 AI 开发项目

## 目标
   搞出接近甚至超过人类水平的军棋AI

   已有[mini军棋项目](https://github.com/fuhongxue00/minijunqi_project)，初步验证方案在简化版规则下是有效的

## 最新进度
- 使用cursor将mini军棋的代码重构，并呕心沥血人工修复诸多bug，勉强可以正常跑通ai_vs_ai.py。
- 加上了铁路和工兵移动，规则应该是完善了，可能「无棋可走」的定义没有和平台对齐，先不管
- trainer.rl训练会有效（能够疯狂吃子了），但揭示了某个bug

## TODO
- 总体而言，完善规则并检查各处bug
- 目前bug rl到七八轮时会报错，发现是某一方选择移动棋子全部非法，说明可能是游戏进度结算有问题，或者是棋子移动仍有bug
- ai层面，检查输入对不对，然后简要测试能力，然后再检查、再测试，看训练情况。




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
python -m minijunqi.ai.train_rl --episodes 100 --out checkpoints/rl.pt
```



