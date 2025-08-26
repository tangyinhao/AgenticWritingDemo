#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将大纲 Markdown + 原文 Markdown 组装为嵌套 JSON，并支持批量遍历目录（扁平 case* 结构版）：
- 根目录下直接放置若干个 case 目录（如：case0 / case1 / case2 / ...）
- 每个 case 目录内包含：
    - outline.md（含 #/##/### 与可选 <tag>...</tag>）
    - full_content.md（原文）
    - 输出默认写为 section_content.json

用法示例：
    python build_outline_json.py --root 数据集根目录路径
可选（覆盖默认文件名）：
    python build_outline_json.py --root 数据集根目录路径 \
        --outline-name outline.md --original-name full_content.md --output-name section_content.json
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

HEADING_RE = re.compile(r'^(#{1,6})\s*(.*?)\s*#*\s*$', re.M)

def normalize_title(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r'^[\s\.\-–—＊*•·\(\)【】\[\]\{\}]+', '', s)
    s = re.sub(r'^\d+[\.\-、：:\)]\s*', '', s)
    s = re.sub(r'^[一二三四五六七八九十百千]+[、.．：:\)]\s*', '', s)
    s = s.replace('`', '').replace('*', '').replace('_', '')
    s = s.lower()
    s = ''.join(ch for ch in s if (unicodedata.category(ch).startswith('L')
                                   or unicodedata.category(ch).startswith('N')
                                   or '\u4e00' <= ch <= '\u9fff'))
    return s

class DocNode:
    def __init__(self, title: str, level: int, start_idx: int):
        self.title = title
        self.level = level
        self.start_idx = start_idx  # 标题所在行号（0-based）
        self.end_idx: Optional[int] = None
        self.children: List['DocNode'] = []
        self.parent: Optional['DocNode'] = None

    def path_titles(self) -> List[str]:
        node, res = self, []
        while node:
            res.append(node.title)
            node = node.parent
        return list(reversed(res))

def parse_markdown_headings(md_text: str) -> Tuple[List[str], List[DocNode]]:
    lines = md_text.splitlines()
    matches = list(HEADING_RE.finditer(md_text))
    nodes: List[DocNode] = []
    for m in matches:
        hashes, title = m.group(1), m.group(2)
        start_pos = m.start()
        start_idx = md_text.count('\n', 0, start_pos)
        level = len(hashes)
        node = DocNode(title.strip(), level, start_idx)
        nodes.append(node)

    stack: List[DocNode] = []
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        stack.append(node)

    total_lines = len(lines)
    for i, node in enumerate(nodes):
        end_line = total_lines
        for j in range(i + 1, len(nodes)):
            if nodes[j].level <= node.level:
                end_line = nodes[j].start_idx
                break
        node.end_idx = end_line

    return lines, nodes

def build_original_path_map(md_text: str) -> Dict[Tuple[str, ...], str]:
    """
    为每个章节构造 content（不包含标题行），满足：
      - 若正文非空：仅正文，末尾补一个换行；
      - 若正文为空：content 为空字符串 ""。
    """
    lines, nodes = parse_markdown_headings(md_text)
    path_map: Dict[Tuple[str, ...], str] = {}
    for node in nodes:
        node.children.sort(key=lambda c: c.start_idx)

    total_lines = len(lines)

    def slice_lines(a: int, b: int) -> str:
        if a >= b:
            return ""
        return "\n".join(lines[a:b]).strip("\n")

    for node in nodes:
        # 标题所在行不再参与 content
        start = node.start_idx + 1
        end = node.end_idx if node.end_idx is not None else total_lines

        # 正文 = 本节到各子节之间的“父级正文”片段拼接（用空行分隔）
        segments: List[Tuple[int, int]] = []
        cursor = start
        for child in node.children:
            if child.start_idx > cursor:
                segments.append((cursor, child.start_idx))
            cursor = max(cursor, child.end_idx if child.end_idx is not None else cursor)
        if end > cursor:
            segments.append((cursor, end))

        pieces = [slice_lines(a, b) for (a, b) in segments if b > a]
        body = "\n\n".join(p for p in pieces if p.strip() != "").rstrip()

        # 只保留正文：非空则补一个结尾换行，空则返回 ""
        content = f"{body}\n" if body else ""

        norm_path = tuple(normalize_title(t) for t in node.path_titles())
        path_map[norm_path] = content

    return path_map

class OutlineNode:
    def __init__(self, title: str, level: int, tag: str = ""):
        self.title = title
        self.level = level
        self.tag = tag
        self.children: List['OutlineNode'] = []

    def to_json_like(self) -> Dict[str, Any]:
        return {"title": self.title, "tag": self.tag, "children": [c.to_json_like() for c in self.children]}

def parse_outline(outline_md: str) -> List[OutlineNode]:
    matches = list(HEADING_RE.finditer(outline_md))
    if not matches:
        return []

    nodes: List[Tuple[int, str, str]] = []
    for idx, m in enumerate(matches):
        hashes, title = m.group(1), m.group(2)
        level = min(len(hashes), 3)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(outline_md)
        block = outline_md[start:end]

        if level == 1:
            tag_text = ""
        else:
            tag_m = re.search(r'<tag>(.*?)</tag>', block, flags=re.DOTALL)
            tag_text = tag_m.group(1).strip() if tag_m else ""

        nodes.append((level, title.strip(), tag_text))

    root_list: List[OutlineNode] = []
    stack: List[OutlineNode] = []
    for level, title, tag_text in nodes:
        node = OutlineNode(title=title, level=level, tag=tag_text)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            root_list.append(node)
        stack.append(node)

    return root_list

def attach_content_from_original(outline_roots: List[OutlineNode],
                                 original_path_map: Dict[Tuple[str, ...], str]) -> Dict[str, Any]:
    def match_content(path_titles: List[str]) -> str:
        full_key = tuple(normalize_title(p) for p in path_titles)
        content = original_path_map.get(full_key)
        if content is not None:
            return content
        tail = normalize_title(path_titles[-1]) if path_titles else ""
        for k, v in original_path_map.items():
            if k and k[-1] == tail:
                return v
        print(f"[WARN] 未在原文中匹配到路径：{' > '.join(path_titles)}", file=sys.stderr)
        return ""

    def node_to_dict(node: OutlineNode, path: List[str]) -> Dict[str, Any]:
        current_path = path + [node.title]
        js = {"title": node.title, "tag": node.tag, "content": match_content(current_path), "children": []}
        if node.children:
            js["children"] = [node_to_dict(c, current_path) for c in node.children]
        return js

    if not outline_roots:
        return {"title": "ROOT", "tag": "", "children": [], "content": ""}

    if len(outline_roots) == 1:
        return node_to_dict(outline_roots[0], [])
    else:
        return {"title": "ROOT", "tag": "", "content": "", "children": [node_to_dict(r, []) for r in outline_roots]}

def build_structure(outline_path: str, original_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    with open(outline_path, 'r', encoding='utf-8') as f:
        outline_md = f.read()
    with open(original_path, 'r', encoding='utf-8') as f:
        original_md = f.read()

    outline_roots = parse_outline(outline_md)
    original_map = build_original_path_map(original_md)
    result = attach_content_from_original(outline_roots, original_map)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result

# -----------------------------
# 批量遍历（扁平 case* 结构）
# -----------------------------

CASE_DIR_PAT = re.compile(r'^case(\d+)$', re.IGNORECASE)

def is_case_dir(name: str) -> Optional[int]:
    """
    返回数字序号（int），若不是形如 case<number> 的目录名则返回 None。
    """
    m = CASE_DIR_PAT.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def process_root(root_dir: str,
                 outline_name: str = "outline.md",
                 original_name: str = "full_content.md",
                 output_name: str = "section_content.json") -> None:
    """
    遍历 root_dir：
      root_dir/
        ├─ case0/
        │   ├─ outline.md
        │   ├─ full_content.md
        │   └─ section_content.json (输出)
        ├─ case1/
        └─ case2/ ...
    """
    if not os.path.isdir(root_dir):
        print(f"[ERROR] 根目录不存在或不是目录：{root_dir}", file=sys.stderr)
        return

    # 收集所有形如 case<number> 的目录，并按数字排序
    case_entries = []
    for name in os.listdir(root_dir):
        case_idx = is_case_dir(name)
        path = os.path.join(root_dir, name)
        if case_idx is not None and os.path.isdir(path):
            case_entries.append((case_idx, name))

    if not case_entries:
        print(f"[WARN] 根目录下未发现任何 case* 目录：{root_dir}", file=sys.stderr)
        return

    case_entries.sort(key=lambda x: x[0])

    for idx, name in case_entries:
        case_dir = os.path.join(root_dir, name)
        outline_path = os.path.join(case_dir, outline_name)
        original_path = os.path.join(case_dir, original_name)
        output_path = os.path.join(case_dir, output_name)

        if not os.path.isfile(outline_path):
            print(f"[WARN] 缺少大纲：{outline_path}，已跳过。", file=sys.stderr)
            continue
        if not os.path.isfile(original_path):
            print(f"[WARN] 缺少原文：{original_path}，已跳过。", file=sys.stderr)
            continue

        try:
            build_structure(outline_path, original_path, output_path)
            print(f"[OK] 已生成：{output_path}")
        except Exception as e:
            print(f"[ERROR] 处理失败：{case_dir}（{e}）", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="将大纲 Markdown 与原文 Markdown 组装为嵌套 JSON（扁平 case* 遍历版）")
    parser.add_argument("--root", default="./", help="根目录路径，内部为若干 case* 目录")
    parser.add_argument("--outline-name", default="outline.md", help="大纲文件名（默认：outline.md）")
    parser.add_argument("--original-name", default="full_content.md", help="原文文件名（默认：full_content.md）")
    parser.add_argument("--output-name", default="section_content.json", help="输出 JSON 文件名（默认：section_content.json）")

    args = parser.parse_args()
    process_root(args.root, args.outline_name, args.original_name, args.output_name)

if __name__ == "__main__":
    main()
