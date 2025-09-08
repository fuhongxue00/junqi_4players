
# 军棋 AI 开发项目

## 目标
   搞出接近甚至超过人类水平的军棋AI

   已有[mini军棋项目](https://github.com/fuhongxue00/minijunqi_project)，初步验证方案在简化版规则下是有效的

## 最新进度
- 使用cursor将mini军棋的代码重构，并人工修复诸多bug，勉强可以正常跑通ai_vs_ai.py。


## TODO
- 总体而言，完善规则并检查各处bug
- 已预留了禁入未知和行营行营规则，对应地实现一下（比较简单，主要就是把坐标写对，并在render的图片里对应地显示一下）
- 移动规则完善，做法是维护一个铁路列表，棋子可以沿一条铁路移动任意长度位置，只要起点和终点之间没有棋子。另外工兵还需要额外写移动规则。（比较复杂，但其实也是一个独立的模块，比较好维护）
- 地雷和工兵的作战强度检查一下（比较简单）
- 跑通aivshuman以及强化学习脚本。
- ai层面，初步检查输入对不对，然后简要测试能力，然后再检查、再测试，看训练情况。




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



