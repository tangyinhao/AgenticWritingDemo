import json
import math

def build_io_pairs(input_file: str, output_file: str):
    # 读取 json 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    history = ""  # 用来存放之前的元素拼接

    for i, elem in enumerate(data):
        if not isinstance(elem, str):
            continue  # 跳过非字符串
        if len(elem) < 8:
            history += elem
            continue

        prefix_len = math.ceil(len(elem) * 0.6)
        prefix = elem[:prefix_len]

        # 构造 input
        input_text = history + prefix

        # 构造 (input, output)
        result.append({
            "input": input_text,
            "output": elem
        })

        # 更新历史
        history += elem

    # 保存结果为 json 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# 使用示例
build_io_pairs("实体密集型文档/0/split_sentence.json", "实体密集型文档/0/split_sentence_io_60.json")
