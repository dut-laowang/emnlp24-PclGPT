# PclGPT
PclGPT是一款基于 ChatGLM-3 和 LLaMA-2 的双语言大型语言模型组 (LLM)，根据训练语言分为PclGPT-CN（基于ChatGLM） 和PclGPT-EN（基于LLaMA）。在基座的基础上，PclGPT综合进行了预训练和监督式微调 (SFT), 用于进行居高临下言论（Patronizing and Condescending Language, PCL）和其他攻击性言论的检测。模型支持的最大上下文为4096。

# 训练过程
![我们通过构建Pcl-PT预训练数据集, Pcl-SFT监督微调数据集以应用于预训练/监督微调过程。具体的构建和训练流程如下图所示。](https://github.com/dut-laowang/PclGPT/blob/main/figures/framework.PNG)

# 测试结果
<table>
  <thead>
    <tr>
      <th rowspan="2">LM</th>
      <th rowspan="2">Model</th>
      <th colspan="3">DPM</th>
      <th colspan="3">TD</th>
      <th colspan="3">CCPC (CN)</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>F1</th>
      <th>P</th><th>R</th><th>F1</th>
      <th>P</th><th>R</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">PLMs</td>
      <td>RoBERTa</td><td>76.3</td><td>78.7</td><td>77.4</td><td>88.4</td><td>86.7</td><td>86.5</td><td>61.2</td><td>61.3</td><td>61.3</td>
    </tr>
    <tr>
      <td>RoBERTa-L</td><td>80.2</td><td>74.9</td><td>77.2</td><td>88.1</td><td>86.0</td><td>85.9</td><td>62.5</td><td>61.6</td><td>62.0</td>
    </tr>
    <tr>
      <td>Chinese-BERT</td><td>71.2</td><td>63.5</td><td>66.2</td><td>76.7</td><td>74.7</td><td>74.2</td><td>66.6</td><td>71.0</td><td>67.3</td>
    </tr>
    <tr>
      <td>M-BERT</td><td>69.2</td><td>76.0</td><td>71.8</td><td>87.6</td><td>87.4</td><td>87.4</td><td>65.8</td><td>67.8</td><td>66.6</td>
    </tr>
    <tr>
      <td rowspan="3">Base-LLMs</td>
      <td>ChatGPT</td><td>50.8</td><td>52.3</td><td>46.9</td><td>59.2</td><td>58.1</td><td>56.7</td><td>53.1</td><td>54.2</td><td>53.6</td>
    </tr>
    <tr>
      <td>GPT-4.0</td><td>51.5</td><td>57.5</td><td>54.3</td><td>60.8</td><td>60.3</td><td>60.5</td><td>55.4</td><td>56.3</td><td>55.7</td>
    </tr>
    <tr>
      <td>Claude-3</td><td>52.3</td><td>52.5</td><td>52.3</td><td>61.6</td><td>64.1</td><td>63.2</td><td>57.2</td><td>57.7</td><td>57.3</td>
    </tr>
    <tr>
      <td rowspan="2">Base-LLMs</td>
      <td>LLama-2-7B</td><td>50.9</td><td>52.6</td><td>51.4</td><td>49.9</td><td>49.7</td><td>49.7</td><td>45.2</td><td>47.5</td><td>46.3</td>
    </tr>
    <tr>
      <td>ChatGLM-3-6B</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>51.9</td><td>50.2</td><td>51.0</td>
    </tr>
    <tr>
      <td rowspan="2">LLMs (Ours)</td>
      <td><strong>PclGPT-EN</strong></td><td><strong>80.4</strong></td><td><strong>81.8</strong></td><td><strong>81.1</strong></td><td><strong>89.9</strong></td><td><strong>89.0</strong></td><td>88.9</td><td>N/A</td><td>N/A</td><td>N/A</td>
    </tr>
    <tr>
      <td><strong>PclGPT-CN</strong></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td><strong>69.1</strong></td><td><strong>72.0</strong></td><td><strong>70.2</strong></td>
    </tr>
  </tbody>
</table>


# 权重和下载
我们在Hugging Face上发布了我们的 PclGPT-CN 和 PclGPT-EN 的 1.0版本权重 

# 引用
如果你计划应用或扩展我们的工作，请引用以下论文
