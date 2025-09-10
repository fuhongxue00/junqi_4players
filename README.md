
# 军棋 AI 开发项目

## 目标
   搞出接近甚至超过人类水平的军棋AI

   已有[mini军棋项目](https://github.com/fuhongxue00/minijunqi_project)，初步验证方案在简化版规则下是有效的

## 最新进度
- 规则层面的bug目前已修复
- rl也已跑通，可以看出吃子数但至今仍是始终和棋(有少数时候可以淘汰一个玩家)

## TODO
- 重整奖励规则，让训练有效
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
python -m minijunqi.ai.train_rl --episodes 100 --out checkpoints/rl.pt
```



