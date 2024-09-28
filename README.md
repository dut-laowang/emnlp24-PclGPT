# PclGPT
PclGPT是一款基于 ChatGLM-3 和 LLaMA-2 的双语言大型语言模型组 (LLM)，根据训练语言分为PclGPT-CN 和PclGPT-EN。在基座的基础上，PclGPT综合进行了预训练和监督式微调 (SFT), 用于进行居高临下言论（Patronizing and Condescending Language, PCL）和其他攻击性言论的检测。模型支持的最大上下文为4096。

# 训练过程
![我们通过构建Pcl-PT预训练数据集, Pcl-SFT监督微调数据集以应用于预训练/监督微调过程。具体的构建和训练流程如下图所示。](https://github.com/dut-laowang/PclGPT/blob/main/figures/framework.PNG)
# 引用
如果你计划应用或扩展我们的工作，请引用以下论文
