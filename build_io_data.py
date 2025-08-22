#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

CASE_DIR_RE = re.compile(r"^case\d+$")

def _read_text_file(path: Path) -> str:
    """安全读取文本文件，不存在则返回空字符串。"""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] 读取失败: {path} ({e})")
    return ""

def _gather_case_dirs(root_dir: Path) -> List[Tuple[str, Path]]:
    """遍历根目录下形如 case0 / case1 / ... 的子目录。"""
    if not root_dir.exists():
        print(f"[WARN] 根目录不存在: {root_dir}")
        return []
    case_dirs: List[Tuple[str, Path]] = []
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and CASE_DIR_RE.match(p.name):
            case_dirs.append((p.name, p))
    if not case_dirs:
        print(f"[WARN] 未发现任何 case* 子目录于: {root_dir}")
    else:
        print(f"[INFO] 共找到 {len(case_dirs)} 个用例目录")
    return case_dirs

def process_one_file(file_path: Path, file_label: str, ratios: List[float]) -> List[Dict]:
    """
    读取单个 split_xxx.json，按给定比例生成 (context, hint, output) 对。
    - 同一文件内 history 逐条累加
    - 对每条 elem，会按多个 ratio 生成多条样本（不改变 history）
    - 读取同级目录下的 user_intent.md 与 outline.md，填入每条样本的字段
    - 新增字段 "file"=file_label（如 "case0"）
    - 过滤规则：len(elem) < 8 或 elem 以 "\\n#" 开头时跳过（但仍累加到 history）
    """
    dir_path = file_path.parent
    user_intent = _read_text_file(dir_path / "user_intent.md")
    outline = _read_text_file(dir_path / "outline.md")
    if not user_intent:
        print(f"[WARN] 未找到或读取失败: {dir_path/'user_intent.md'}")
    if not outline:
        print(f"[WARN] 未找到或读取失败: {dir_path/'outline.md'}")

    # 读取 JSON 数据
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取失败，已跳过: {file_path} ({e})")
        return []

    if not isinstance(data, list):
        print(f"[WARN] JSON 非列表，已跳过: {file_path}")
        return []

    results: List[Dict] = []
    history = ""

    for elem in data:
        if not isinstance(elem, str):
            continue

        is_heading_fragment = elem.startswith("\n#")
        if len(elem) < 8 or is_heading_fragment:
            # 保持你当前策略：短元素/标题片段不产样本，但纳入 history
            history += elem
            continue

        # 对每个切割比例生成一条样本
        items_for_elem = []
        for r in ratios:
            prefix_len = math.ceil(len(elem) * r)
            prefix = elem[:prefix_len]
            input_text = history  # context 不包含本 elem

            items_for_elem.append({
                "context": input_text,
                "hint": prefix,
                "output": elem,
                "ratio": r,
                "user_intent": user_intent,
                "outline": outline,
                "file": file_label,  # 新增字段
            })

        results.extend(items_for_elem)

        # 在本 elem 处理完所有 ratio 之后再更新历史
        history += elem

    return results

def _build_for_filename(
    root_dir: Path,
    output_file: str,
    ratios: List[float],
    filename: str
):
    """
    针对指定 filename（如 split_sentence.json 或 split_clause.json）
    遍历所有 case* 目录，生成样本并保存。
    """
    all_results: List[Dict] = []
    case_dirs = _gather_case_dirs(root_dir)

    for case_name, case_path in case_dirs:
        fp = case_path / filename
        if not fp.exists():
            print(f"[WARN] 缺少目标文件: {fp}")
            continue

        print(f"[INFO] 处理 {case_name} -> {filename}")
        results = process_one_file(fp, case_name, ratios)
        all_results.extend(results)

    # 保存合并结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] ({filename}) 共生成样本 {len(all_results)} 条，已保存到: {output_file}")

def main():
    # ===== 配置根目录（按需修改） =====
    root_dir = Path("./")  # 👉 改成你的根目录路径

    # ===== 比例配置 =====
    # ratios = [0.1, 0.3, 0.5]
    ratios = [0.0, 0.3]

    # —— 1) 处理按句号/分号切片的文件 —— #
    sentence_output = "all_cases_io_sentence.json"
    _build_for_filename(
        root_dir=root_dir,
        output_file=sentence_output,
        ratios=ratios,
        filename="split_sentence.json"
    )

    # —— 2) 处理按逗号/从句切片的文件 —— #
    clause_output = "all_cases_io_clause.json"
    _build_for_filename(
        root_dir=root_dir,
        output_file=clause_output,
        ratios=ratios,
        filename="split_clause.json"
    )

if __name__ == "__main__":
    main()
