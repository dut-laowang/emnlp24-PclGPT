# PclGPT: A Large Language Model for Patronizing and Condescending Language Detection
## 🎓 This paper has been accepted in EMNLP 2024 ！

# PCL
**Patronizing and Condescending Language** (PCL) is a type of micro-aggression against vulnerable groups on the Internet. It is a subcategory of toxic speech and is an emerging field since 2022.

# PclGPT
PclGPT is a bilingual large language model group (LLM) based on ChatGLM-3 and LLaMA-2, divided into two versions according to the training language: PclGPT-CN (based on ChatGLM) and PclGPT-EN (based on LLaMA). Built upon these foundational models, PclGPT has undergone both pre-training and supervised fine-tuning (SFT) to detect patronizing and condescending language (PCL) and other offensive speech. The maximum supported context length for the model is 4096 tokens.

# Training Process
![We constructed the Pcl-PT pre-training dataset and the Pcl-SFT supervised fine-tuning dataset to be used in the pre-training/supervised fine-tuning process. The specific construction and training process are illustrated in the diagram below.](https://github.com/dut-laowang/PclGPT/blob/main/figures/framework.PNG)

# Test Results
We evaluated the detection performance of the model suite on two public English datasets for condescension detection (Talkdown, Don't Patronize Me) and one Chinese dataset (CPCL).

The performance metrics are calculated using Macro Precision, Recall, and F1-score.

## English Task
The results for the detection tasks in the PclGPT-EN English model group are as follows:
<table>
  <thead>
    <tr>
      <th rowspan="2">LM</th>
      <th rowspan="2">Model</th>
      <th colspan="3">DPM</th>
      <th colspan="3">TD</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>F1</th>
      <th>P</th><th>R</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">PLMs</td>
      <td>RoBERTa</td><td>76.3</td><td>78.7</td><td>77.4</td><td>88.4</td><td>86.7</td><td>86.5</td>
    </tr>
    <tr>
      <td>RoBERTa-L</td><td>80.2</td><td>74.9</td><td>77.2</td><td>88.1</td><td>86.0</td><td>85.9</td>
    </tr>
    <tr>
      <td>M-BERT</td><td>69.2</td><td>76.0</td><td>71.8</td><td>87.6</td><td>87.4</td><td>87.4</td>
    </tr>
    <tr>
      <td rowspan="3">Base-LLMs</td>
      <td>ChatGPT</td><td>50.8</td><td>52.3</td><td>46.9</td><td>59.2</td><td>58.1</td><td>56.7</td>
    </tr>
    <tr>
      <td>GPT-4.0</td><td>51.5</td><td>57.5</td><td>54.3</td><td>60.8</td><td>60.3</td><td>60.5</td>
    </tr>
    <tr>
      <td>Claude-3</td><td>52.3</td><td>52.5</td><td>52.3</td><td>61.6</td><td>64.1</td><td>63.2</td>
    </tr>
    <tr>
      <td rowspan="2">Base-LLMs</td>
      <td>LLama-2-7B</td><td>50.9</td><td>52.6</td><td>51.4</td><td>49.9</td><td>49.7</td><td>49.7</td>
    </tr>
    <tr>
      <td>ChatGLM-3-6B</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
    </tr>
    <tr>
      <td rowspan="1">LLMs (Ours)</td>
      <td><strong>PclGPT-EN</strong></td><td><strong>80.4</strong></td><td><strong>81.8</strong></td><td><strong>81.1</strong></td><td><strong>89.9</strong></td><td><strong>89.0</strong></td><td><strong>88.9</strong></td>
    </tr>
  </tbody>
</table>


## Chinese Task
The results for the detection tasks in the PclGPT-CN Chinese model group are as follows:
<table>
  <thead>
    <tr>
      <th rowspan="2">LM</th>
      <th rowspan="2">Model</th>
      <th colspan="3">CPCL (CN)</th>
    </tr>
    <tr>
      <th>P</th><th>R</th><th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">PLMs</td>
      <td>RoBERTa</td><td>61.2</td><td>61.3</td><td>61.3</td>
    </tr>
    <tr>
      <td>RoBERTa-L</td><td>62.5</td><td>61.6</td><td>62.0</td>
    </tr>
    <tr>
      <td>Chinese-BERT</td><td>66.6</td><td>71.0</td><td>67.3</td>
    </tr>
    <tr>
      <td>M-BERT</td><td>65.8</td><td>67.8</td><td>66.6</td>
    </tr>
    <tr>
      <td rowspan="3">Base-LLMs</td>
      <td>ChatGPT</td><td>53.1</td><td>54.2</td><td>53.6</td>
    </tr>
    <tr>
      <td>GPT-4.0</td><td>55.4</td><td>56.3</td><td>55.7</td>
    </tr>
    <tr>
      <td>Claude-3</td><td>57.2</td><td>57.7</td><td>57.3</td>
    </tr>
    <tr>
      <td rowspan="2">Base-LLMs</td>
      <td>LLama-2-7B</td><td>45.2</td><td>47.5</td><td>46.3</td>
    </tr>
    <tr>
      <td>ChatGLM-3-6B</td><td>51.9</td><td>50.2</td><td>51.0</td>
    </tr>
    <tr>
      <td rowspan="1">LLMs (Ours)</td>
      <td><strong>PclGPT-CN</strong></td><td><strong>69.1</strong></td><td><strong>72.0</strong></td><td><strong>70.2</strong></td>
    </tr>
  </tbody>
</table>


# Weights and Downloads
We have released version 1.0 weights of our PclGPT-CN and PclGPT-EN models on Hugging Face.

PclGPT-EN: [https://huggingface.co/DUTIR-Wang/PclGPT-EN](https://huggingface.co/DUTIR-Wang/PclGPT-EN)

PclGPT-CN: [https://huggingface.co/DUTIR-Wang/PclGPT-CN](https://huggingface.co/DUTIR-Wang/PclGPT-CN)

# Inference
After downloading the weights, use the following code for single-sample inference with PclGPT-EN.
```python
from transformers import LlamaTokenizer, LlamaForCausalLM

# LLaMA and Tokenizer
tokenizer = LlamaTokenizer.from_pretrained("DUTIR-Wang/PclGPT-EN")
model = LlamaForCausalLM.from_pretrained("DUTIR-Wang/PclGPT-EN").half().cuda()

def generate_response():
    # Sample
    sample_text = "For someone who's just a mere street sweeper, you sure think highly of yourself."
    
    instruction = (
        "Suppose you are a linguist and you are asked to judge whether a text is patronizing and condescending. "
        "Patronizing and condescending language expresses a sense of superiority or belittles others, making them feel inferior or incapable. "
        "# Your return: Based on the following conversation, make a decision and return your choice. "
        "FALSE means the text is not a condescending language, and TRUE means the text is a condescending language. "
        "-> here is the text: （{}）"
    ).format(sample_text)

    inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_output = extract_option(output_text)
    return classify_output(model_output)

def extract_option(output_text):
    options = ['TRUE', 'FALSE']
    for option in options:
        if option in output_text:
            return option
    return "Could not detect."

def classify_output(model_output):
    # return
    return "The text is {} a condescending language.".format(
        "not" if model_output == "FALSE" else ""
    )

response = generate_response()
print(response)
```
The output will be
```
"The text is a condescending language."
```

After downloading the weights, use the following code for single-sample inference with PclGPT-CN.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DUTIR-Wang/PclGPT-CN", trust_remote_code=True)
model = AutoModel.from_pretrained("DUTIR-Wang/PclGPT-CN", trust_remote_code=True).half().cuda()

def generate_response():
    # 样例文本
    sample_text = "区区扫地的，你可真是厉害坏了"
    
    instruction = (
        "假定你是一名语言学家，检测居高临下言论。居高临下言论是优势地位群体针对弱势群体的优越言论，"
        "语言攻击性较弱，但往往夹杂着讽刺，刻板印象，会对弱势群体造成伤害。"
        "居高临下根据语气程度分为 A级：非居高临下（正常非歧视、非冷暴力言论），"
        "B级：轻微居高临下（语气略带鼓励或怜悯），C级：中等居高临下（说话人较为客观陈述，但语气带有歧视），"
        "D级：严重居高临下（说话人语气轻蔑，严重歧视弱势群体）。"
        "接下来将给你一段文本，根据上述规则，你负责判断该文本属于（A/B/C/D级）的哪一级，并只回答选项。"
        "-> 文本：({})"
    ).format(sample_text)

    # Tokenizer 和模型推理
    inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_output = extract_option(output_text)
    return classify_output(model_output)

def extract_option(output_text):
    options = ['A', 'B', 'C', 'D']
    for char in reversed(output_text.strip()):
        if char in options:
            return char
    return "无法识别的输出" 

def classify_output(model_output):
    # 根据模型输出的选项返回相应的解释
    if model_output == "A":
        return "判断为A级：非居高临下"
    elif model_output == "B":
        return "判断为B级：轻微居高临下"
    elif model_output == "C":
        return "判断为C级：中等居高临下"
    elif model_output == "D":
        return "判断为D级：严重居高临下"
    else:
        return "无法识别的输出，请检查输入或模型输出"

response = generate_response()
print(response)
```
得到的输出为
```
"判断为D级：严重居高临下"
```
# Cite
Our paper can be accessed here. Paper link: [https://arxiv.org/abs/2410.00361](https://arxiv.org/abs/2410.00361)

If you plan to apply or extend our work, please cite the following paper.
```bibtex
@misc{wang2024pclgptlargelanguagemodel,
      title={PclGPT: A Large Language Model for Patronizing and Condescending Language Detection}, 
      author={Hongbo Wang and Mingda Li and Junyu Lu and Hebin Xia and Liang Yang and Bo Xu and Ruizhu Liu and Hongfei Lin},
      year={2024},
      eprint={2410.00361},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.00361}, 
}
```

# Statement
**The work studied in this paper falls within the subcategory of Toxic Speech. Condescending speech is a form of microaggression, and therefore, part of this research may cause discomfort and sensitivity among users. This research is solely intended for the protection of vulnerable groups and for identifying and managing online verbal attacks. Please do not use the model weights to generate any harmful content.**
