import re
import json

def split_markdown_to_json(md_file: str, json_file_sent: str, json_file_comma: str):
    """
    读取 markdown 文件内容，分别按句号/分号和逗号切片，
    并保存为两个 JSON 文件（列表，每个元素为一个切片）。

    :param md_file: 输入的 markdown 文件路径
    :param json_file_sent: 按句号/分号切片的 json 文件路径
    :param json_file_comma: 按逗号切片的 json 文件路径
    """
    # 读取文件
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    # ===== 按句号/分号切片（保留符号） =====
    # 匹配：中文句号、英文句号、中文分号、英文分号
    slices_sent = re.split(r'(?<=[。\.；;])', text)
    slices_sent = [s.strip() for s in slices_sent if s.strip()]

    with open(json_file_sent, "w", encoding="utf-8") as f:
        json.dump(slices_sent, f, ensure_ascii=False, indent=2)

    print(f"已完成句号/分号切片，结果保存到 {json_file_sent}")

    # ===== 按逗号切片（保留逗号） =====
    slices_comma = re.split(r'(?<=[，,])', text)
    slices_comma = [s.strip() for s in slices_comma if s.strip()]

    with open(json_file_comma, "w", encoding="utf-8") as f:
        json.dump(slices_comma, f, ensure_ascii=False, indent=2)

    print(f"已完成逗号切片，结果保存到 {json_file_comma}")


# 调用示例
split_markdown_to_json(
    "引用密集型文档/0/full_content.md",
    "引用密集型文档/0/split_sentence.json",
    "引用密集型文档/0/split_comma.json"
)
