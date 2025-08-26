#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

def _find_history_from_markdown(md_text: str, snippet: str) -> Optional[str]:
    """
    在 md_text 中查找 snippet 的首次出现。
    命中则返回其前面的内容（作为 history/context）。
    若未命中，返回 None。
    - 先做精确匹配；若失败，再做“空白宽松”的正则匹配（将 snippet 中连续空白折叠为 \s+）。
    """
    if not snippet:
        return None

    # 优先精确匹配
    idx = md_text.find(snippet)
    if idx != -1:
        return md_text[:idx]

    # 宽松匹配：忽略空白差异
    # 将 snippet 中的连续空白折叠为 \s+，其余字符转义
    snippet_norm = re.sub(r"\s+", r"\\s+", re.escape(snippet.strip()))
    try:
        m = re.search(snippet_norm, md_text, flags=re.DOTALL)
        if m:
            return md_text[:m.start()]
    except re.error as e:
        print(f"[WARN] 正则匹配失败（将退回放弃该样本）: {e}")

    return None

def process_one_file(file_path: Path, file_label: str, ratios: List[float]) -> List[Dict]:
    """
    读取单个 split_snippet.json，按给定比例生成 (context, hint, output) 对。
    - 不再使用逐条累加的 history；改为：对每个元素到 full_content.md 中首次匹配，
      取匹配到的起始位置之前文本作为 history（context）。
    - 对每条 elem，会按多个 ratio 生成多条样本。
    - 读取同级目录下的 user_intent.md 与 outline.md，填入每条样本的字段。
    - 新增字段 "file"=file_label（如 "case0"）。
    - 过滤规则：len(elem) < 8 或 elem 以 "\\n#" 开头时跳过（但仍会去尝试匹配以便日志定位）。
    """
    dir_path = file_path.parent
    user_intent = _read_text_file(dir_path / "user_intent.md")
    outline = _read_text_file(dir_path / "outline.md")
    md_text = _read_text_file(dir_path / "full_content.md")

    if not user_intent:
        print(f"[WARN] 未找到或读取失败: {dir_path/'user_intent.md'}")
    if not outline:
        print(f"[WARN] 未找到或读取失败: {dir_path/'outline.md'}")
    if not md_text:
        print(f"[WARN] 未找到或读取失败: {dir_path/'full_content.md'}（该目录将无法生成样本）")

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

    for elem in data:
        if not isinstance(elem, str):
            continue

        is_heading_fragment = elem.startswith("\n#")
        if len(elem) < 8 or is_heading_fragment:
            # 与之前逻辑一致：这类元素不产样本。
            # 这里 history 不再累加，由 markdown 定位，仍尝试匹配仅用于日志定位/调试。
            if md_text:
                hist = _find_history_from_markdown(md_text, elem)
                if hist is None:
                    print(f"[INFO] 跳过（未找到或过短/标题片段）且未匹配到：{file_path} -> 片段开头: {repr(elem[:20])}")
            continue

        if not md_text:
            # 没有 full_content.md，无法生成该元素的样本
            print(f"[WARN] 缺少 full_content.md，跳过样本：{file_path} -> {repr(elem[:20])}")
            continue

        history = _find_history_from_markdown(md_text, elem)
        if history is None:
            print(f"[WARN] 在 markdown 中未匹配到该片段（将跳过）：{file_path} -> 片段开头: {repr(elem[:50])}")
            continue

        # 对每个切割比例生成样本
        for r in ratios:
            prefix_len = math.ceil(len(elem) * r)
            prefix = elem[:prefix_len]

            results.append({
                "context": history,      # 由 markdown 首次匹配位置之前的内容构成
                "hint": prefix,
                "output": elem,
                "ratio": r,
                "user_intent": user_intent,
                "outline": outline,
                "file": file_label,
            })

    return results

def _build_for_filename(
    root_dir: Path,
    output_file: str,
    ratios: List[float],
    filename: str
):
    """
    针对指定 filename（此处应为 split_snippet.json）
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
    ratios = [0.0, 0.3]

    # —— 处理按 snippet 切片的文件 —— #
    snippet_output = "all_cases_io_snippet.json"
    _build_for_filename(
        root_dir=root_dir,
        output_file=snippet_output,
        ratios=ratios,
        filename="split_snippet.json"
    )

if __name__ == "__main__":
    main()
