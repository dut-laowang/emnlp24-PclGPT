from transformers import AutoTokenizer, AutoModel
import unicodedata

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("DUTIR-Wang/PclGPT-CN", trust_remote_code=True)
model = AutoModel.from_pretrained("DUTIR-Wang/PclGPT-CN", trust_remote_code=True).half().cuda()

# 清洗函数：统一全角字符、移除不可见字符
def clean_input(text):
    normalized = unicodedata.normalize('NFKC', text)
    return ''.join(ch for ch in normalized if ch.isprintable())

# 推理主函数
def generate_response(user_input):
    instruction = (
        "假定你是一名语言学家，检测居高临下言论。居高临下言论是优势地位群体针对弱势群体的优越言论，"
        "语言攻击性较弱，但往往夹杂着讽刺，刻板印象，会对弱势群体造成伤害。"
        "居高临下根据语气程度分为 A级：非居高临下（正常非歧视、非冷暴力言论），"
        "B级：轻微居高临下（语气略带鼓励或怜悯），C级：中等居高临下（说话人较为客观陈述，但语气带有歧视），"
        "D级：严重居高临下（说话人语气轻蔑，严重歧视弱势群体）。"
        "接下来将给你一段文本，根据上述规则，你负责判断该文本属于（A/B/C/D级）的哪一级，并只回答选项。"
        "-> 文本：({})"
    ).format(str(user_input).strip())

    inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_output = extract_option(output_text)
    return classify_output(model_output)

# 提取模型返回的 A/B/C/D
def extract_option(output_text):
    options = ['A', 'B', 'C', 'D']
    for char in reversed(output_text.strip()):
        if char in options:
            return char
    return "无法识别的输出"

# 显示判断等级
def classify_output(model_output):
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

# 主交互循环
if __name__ == "__main__":
    print("欢迎使用居高临下言论识别模型，输入 exit 可退出。\n")

    while True:
        user_input = input("请输入文本，进行居高临下言论判别：\n")
        if user_input.strip().lower() == "exit":
            print("已退出。")
            break
        if not user_input.strip():
            print("⚠️ 输入不能为空，请重新输入。\n")
            continue
        try:
            cleaned = clean_input(user_input)
            response = generate_response(cleaned)
            print(response + "\n")
        except Exception as e:
            print("❌ 模型处理出错：", e, "\n")
