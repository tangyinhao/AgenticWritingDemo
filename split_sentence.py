#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
from typing import List, Tuple

SENT_PUNCT = r'(?<=[。？！；])'          # 句子级：仅中文句末标点（保留分隔符）
clause_PUNCT = r'(?<=[，。？！；])'       # 逗号级：中文逗号 + 句末标点（保留分隔符）
HEADING_RE = re.compile(r'^\s{0,3}(#{1,3})\s+.*?$', flags=re.M)  # 只分离 #/##/### 标题行

def _cut_before_reference(text: str) -> str:
    """
    忽略 '# Reference' 及其之后内容。按原要求，精确匹配 '# Reference'。
    """
    idx = text.find("# Reference")
    return text if idx == -1 else text[:idx]

def _split_into_blocks_with_headings(text: str) -> List[str]:
    """
    将文本按标题与非标题块分离。标题行作为独立块返回。
    """
    # 统一换行
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 用捕获组 split，保留标题行
    parts = re.split(r'(^\s{0,3}#{1,3}\s.*?$)', text, flags=re.M)
    # 过滤空串
    return [p for p in parts if p is not None and p != ""]

def _wrap_heading(h: str) -> str:
    """
    标题前后强制加入换行符，避免重复换行。
    """
    h = h.strip()  # 去掉首尾空白
    return f"\n{h}\n"

def split_markdown_to_lists(text: str) -> Tuple[List[str], List[str]]:
    """
    返回（句子级列表，逗号级列表）。
    规则：
      - 先把 #/##/### 标题行单独分离，并以 '\n... \n' 形式作为独立片段加入两个结果；
      - 非标题块再按原有标点规则切分；
      - 仍然会在 '# Reference' 之后截断。
    """
    text = _cut_before_reference(text)
    blocks = _split_into_blocks_with_headings(text)

    sent_list: List[str] = []
    clause_list: List[str] = []

    for blk in blocks:
        if HEADING_RE.match(blk):
            # 标题：作为独立片段加入，前后带换行
            heading_item = _wrap_heading(blk)
            sent_list.append(heading_item)
            clause_list.append(heading_item)
        else:
            # 非标题：按标点切分
            # 句子级
            s_parts = re.split(SENT_PUNCT, blk)
            s_parts = [s.strip() for s in s_parts if s and s.strip()]
            sent_list.extend(s_parts)

            # 逗号级
            c_parts = re.split(clause_PUNCT, blk)
            c_parts = [s.strip() for s in c_parts if s and s.strip()]
            clause_list.extend(c_parts)

    return sent_list, clause_list

def process_one_case_dir(case_dir: Path,
                         md_name: str = "full_content.md",
                         sent_json_name: str = "split_sentence.json",
                         clause_json_name: str = "split_clause.json") -> None:
    """
    在单个 case 目录中执行分片，并写入两个 JSON 文件。
    """
    md_path = case_dir / md_name
    if not md_path.exists():
        print(f"[SKIP] {case_dir} 下未找到 {md_name}")
        return

    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] 读取失败：{md_path} ({e})")
        return

    slices_sent, slices_clause = split_markdown_to_lists(text)

    (case_dir / sent_json_name).write_text(
        json.dumps(slices_sent, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[OK] 句子级切片写入：{case_dir / sent_json_name}")

    (case_dir / clause_json_name).write_text(
        json.dumps(slices_clause, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[OK] 逗号级切片写入：{case_dir / clause_json_name}")

def process_root(root_dir: str,
                 case_pattern: str = r"^case\d+$",
                 md_name: str = "full_content.md",
                 sent_json_name: str = "split_sentence.json",
                 clause_json_name: str = "split_clause.json") -> None:
    """
    批量处理根目录下所有符合 case_pattern 的子目录。
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERR] 根目录不存在：{root}")
        return

    regex = re.compile(case_pattern)
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and regex.match(p.name)])

    if not case_dirs:
        print(f"[INFO] 在 {root} 下未找到匹配 {case_pattern} 的子目录")
        return

    print(f"[INFO] 将处理 {len(case_dirs)} 个目录：{', '.join(p.name for p in case_dirs)}")
    for d in case_dirs:
        process_one_case_dir(d, md_name, sent_json_name, clause_json_name)

# ===== 示例调用 =====
if __name__ == "__main__":
    process_root("./")
