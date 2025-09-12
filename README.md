
# 军棋 AI 开发项目

## 目标
   搞出接近甚至超过人类水平的军棋AI

   已有[mini军棋项目](https://github.com/fuhongxue00/minijunqi_project)，初步验证方案在简化版规则下是有效的

## 最新进度
- 规则层面的bug目前已修复，规则目前只差「40死后暴露军棋位置」这个规则实现比较烦、要改game和policy的encode方法
- 可视化之类的优化了一下
- 修改了网络输入的bug，多次调参。从01_new.pt开始（测试基本很强），保留了之前的04 05 06作为陪练
- 02_new.pt从训练日志看应该是无敌的，还测试结果也几乎是没有败且平局并不太多

## TODO
- 跑通并优化一下ai_vs_human.py，当aivsai看上去不太蠢时就可以细节测试了
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



