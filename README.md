这是算法设计与分析GoogleResearchFootball的Project项目。



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
