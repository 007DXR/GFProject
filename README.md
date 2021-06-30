# 写在前面

1：这**不是**最终报告，最终报告请查看pdf文件；

2：助教可能看到只有一个人上传代码，因为我们在线下一块完成的项目，所以在我们看来用微信互传压缩包比在GitHub上共享代码更方便。其实双方对代码的贡献都很大，**请不要根据上传次数判断代码量和工作量**。

3.这里，为了便于阅读核验，我们没有给出其他对比算法的使用和训练方法。我们把其他算法的使用说明放在了报告中以备查阅，如助教有兴趣，请移步pdf报告中PERFORMANCE EVALUATION章节下的Experimental Setup模块。

4.我们的Project有以下部分：

（1）**代码**。包括D3QN，DQN，PPO，RuleBased 和RuleAssisted 算法在Google Football 上的实现（其中PPO算法有必要的注释）。

（2）**模型**。包括RuleAssisted和PPO算法的模型。

（3）**报告**。最终报告包括项目介绍，算法理论分析，具体实现，复现方法，性能评估展示，分工和引用参考文献。期中报告也已经附上。

（4）**视频**。见压缩包。

5.为了方便助教核验，这里我们指出项目所要求的的内容可以在如下位置找到：

（1）**zip压缩包，包含项目报告，全部代码和训练模型**：教学网入口。

（2）**代码运行的环境，使用方法**：GFPPO请看下面的README，其他算法请看*最终报告.pdf*中的PERFORMANCE EVALUATION章节下的Experimental Setup模块。

（3）**任务选择的难度**：RL 3 vs 1 with Keeper.

（4）**具体算法**：详见代码。适应性改变请看*最终报告.pdf*中的PERFORMANCE EVALUATION章节下的Implementation and Results模块。

（5）**最终视频**：详见压缩包。胜率和获胜时长等分析请看*最终报告.pdf*中的PERFORMANCE EVALUATION章节下的Implementation and Results模块。

（6）**必要的逻辑和数据分析**：请看*最终报告.pdf*中的ALGORITHM章节。

（7）**GitHub开源地址**：https://github.com/382308889/GFProject。

（8）**注释**：见代码。

# PPO算法训练和测试方法

该部分由董欣然完成。

## 代码文件的功能

- train.py: 使用PPO方法进行训练
- GFPPO.py: 加载模型进行测试，并进行图像渲染

## 环境配置
- linux
- python-3.6.8
- openai-baselines
- pytorch-1.1.0

## 准备工作
- 从github/google-research下载[football](https://github.com/google-research/football)压缩包，并在linux环境下解压，将文件夹命名为football
- 从github下载[GFProject](https://github.com/382308889/GFProject)压缩包，并放在football/gfootball/examples/下解压，将文件夹命名为GFProject-master

## 运行代码进行训练

- 在arguments.py中将“--env-name”行设置default='academy_3_vs_1_with_keeper'，则训练3vs1模型；设置default='academy_empty_goal_close'，则训练踢空门模型
- 在GFProject-master文件夹下打开终端
- 在命令行键入python3 train.py，便可开始训练
- 训练的模型储存在saved_models文件夹中

## 加载模型进行测试
- 在GFProject-master文件夹下打开终端
- 在命令行键入python3 GFPPO.py，便可观摩比赛
- env_name可设置为"academy_3_vs_1_with_keeper"或"academy_empty_goal_close"，对应着3vs1和踢空门两种模式
- render设置为"True"则开启渲染，设置为"False"则关闭渲染
- rewards可设置为'scoring', 'scoring,checkpoints'，分表代表只有进球得分和除了带有附加得分
