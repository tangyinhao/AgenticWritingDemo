#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将大纲 Markdown + 原文 Markdown 组装为嵌套 JSON。
- 大纲最多到三级标题（# / ## / ###），每个节内可包含 <tag>...</tag> 作为章节写作摘要
- 叶子节点（无子标题）需从原文中抽取对应完整内容（该标题以下直到下一个同级或更高标题之前的所有原文）
- 输出结构为嵌套字典：
    {
      "title": "...",
      "tag": "...",
      "children": [
        { ... },
        ...
      ]
      # 对于叶子节点，还会有：
      "content": "原文对应内容",
      "children": []
    }

用法：
    python build_outline_json.py outline.md original.md -o structure.json
不加 -o 则打印到 stdout
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
    """
    归一化标题用于匹配：
    - NFKC 规范化
    - 全部小写
    - 去掉 Markdown/行内代码/空白
    - 去掉前缀编号（如 "1. "、"一、" 等常见形式）
    - 去除标点与空格，仅保留字母数字与 CJK
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()

    # 去掉常见编号前缀（宽松处理）
    s = re.sub(r'^[\s\.\-–—＊*•·\(\)【】\[\]\{\}]+', '', s)
    s = re.sub(r'^\d+[\.\-、：:\)]\s*', '', s)
    s = re.sub(r'^[一二三四五六七八九十百千]+[、.．：:\)]\s*', '', s)

    # 去掉行内代码反引号与加粗/斜体星号等
    s = s.replace('`', '').replace('*', '').replace('_', '')

    # 小写
    s = s.lower()

    # 仅保留字母数字与 CJK（把其他符号/空白去掉）
    s = ''.join(ch for ch in s if (unicodedata.category(ch).startswith('L')  # Letter
                                   or unicodedata.category(ch).startswith('N')  # Number
                                   or '\u4e00' <= ch <= '\u9fff'))  # CJK
    return s

# -----------------------------
# 原文解析：构建标题树并切片内容
# -----------------------------

class DocNode:
    def __init__(self, title: str, level: int, start_idx: int):
        self.title = title
        self.level = level
        self.start_idx = start_idx   # 在整篇文本中的行号（标题行）
        self.end_idx: Optional[int] = None  # 结束行号（不含）
        self.children: List['DocNode'] = []
        self.parent: Optional['DocNode'] = None

    def path_titles(self) -> List[str]:
        node, res = self, []
        while node:
            res.append(node.title)
            node = node.parent
        return list(reversed(res))

def parse_markdown_headings(md_text: str) -> Tuple[List[str], List[DocNode]]:
    """
    解析原文 Markdown 标题，返回 (行列表, 节点列表)
    节点包含层级、起止行号，build_end_indices 后可切片内容
    """
    lines = md_text.splitlines()
    matches = list(HEADING_RE.finditer(md_text))

    nodes: List[DocNode] = []
    for m in matches:
        hashes, title = m.group(1), m.group(2)
        # 计算其行号（通过起始位置反推）
        start_pos = m.start()
        start_idx = md_text.count('\n', 0, start_pos)
        level = len(hashes)
        node = DocNode(title.strip(), level, start_idx)
        nodes.append(node)

    # 构建父子关系（按层级用栈）
    stack: List[DocNode] = []
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        stack.append(node)

    # 计算 end_idx：下一个（同级或更高）标题的起始行号；最后一个到文末
    total_lines = len(lines)
    for i, node in enumerate(nodes):
        # 找下一个 index j > i，满足 level <= node.level
        end_line = total_lines
        for j in range(i + 1, len(nodes)):
            if nodes[j].level <= node.level:
                end_line = nodes[j].start_idx
                break
        node.end_idx = end_line

    return lines, nodes

def build_original_path_map(md_text: str) -> Dict[Tuple[str, ...], str]:
    """
    为原文每个标题生成 '仅自身直辖内容' 的映射：
    - 节点区间： [node.start_idx+1, node.end_idx)
    - 子区间：   [child.start_idx, child.end_idx)
    - 自身内容 = 节点区间 - 所有子区间 的并集（保持原顺序拼接）
    """
    lines, nodes = parse_markdown_headings(md_text)
    path_map: Dict[Tuple[str, ...], str] = {}

    # 方便按开始行排序子节点
    for node in nodes:
        node.children.sort(key=lambda c: c.start_idx)

    total_lines = len(lines)

    def slice_lines(a: int, b: int) -> str:
        if a >= b:
            return ""
        # 去掉收尾多余空行，但保留段落内部换行
        return "\n".join(lines[a:b]).strip("\n")

    for node in nodes:
        start = node.start_idx + 1
        end = node.end_idx if node.end_idx is not None else total_lines

        # 计算“自身”片段：父区间去除所有子区间
        segments: List[Tuple[int, int]] = []
        cursor = start
        for child in node.children:
            if child.start_idx > cursor:
                segments.append((cursor, child.start_idx))
            cursor = max(cursor, child.end_idx if child.end_idx is not None else cursor)
        if end > cursor:
            segments.append((cursor, end))

        # 拼接自身片段（用空行分隔，避免硬粘连）
        pieces = [slice_lines(a, b) for (a, b) in segments if b > a]
        content = "\n\n".join(p for p in pieces if p.strip() != "")

        norm_path = tuple(normalize_title(t) for t in node.path_titles())
        path_map[norm_path] = content

    return path_map

# -----------------------------
# 大纲解析：构建仅到 ### 的树，并提取 <tag> 内容
# -----------------------------

class OutlineNode:
    def __init__(self, title: str, level: int, tag: str = ""):
        self.title = title
        self.level = level
        self.tag = tag
        self.children: List['OutlineNode'] = []

    def to_json_like(self) -> Dict[str, Any]:
        # content 字段只在叶子阶段补上，这里不放
        return {
            "title": self.title,
            "tag": self.tag,
            "children": [c.to_json_like() for c in self.children]
        }

def parse_outline(outline_md: str) -> List[OutlineNode]:
    matches = list(HEADING_RE.finditer(outline_md))
    if not matches:
        return []

    nodes: List[Tuple[int, str, str]] = []
    for idx, m in enumerate(matches):
        hashes, title = m.group(1), m.group(2)
        level = min(len(hashes), 3)  # 只保留到 ###

        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(outline_md)
        block = outline_md[start:end]

        # 一级标题强制忽略 <tag>
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

# -----------------------------
# 将大纲叶子与原文映射，填充 content
# -----------------------------

def attach_content_from_original(outline_roots: List[OutlineNode],
                                 original_path_map: Dict[Tuple[str, ...], str]) -> Dict[str, Any]:
    """
    所有节点均尝试填充 content：
      1) 先按完整路径（规范化）匹配
      2) 失败则回退到仅末尾标题匹配（取首个命中）
      3) 仍失败则 content = ""
    """
    def match_content(path_titles: List[str]) -> str:
        full_key = tuple(normalize_title(p) for p in path_titles)
        content = original_path_map.get(full_key)
        if content is not None:
            return content

        # 末尾标题回退
        tail = normalize_title(path_titles[-1]) if path_titles else ""
        for k, v in original_path_map.items():
            if k and k[-1] == tail:
                return v
        # 未匹配
        print(f"[WARN] 未在原文中匹配到路径：{' > '.join(path_titles)}", file=sys.stderr)
        return ""

    def node_to_dict(node: OutlineNode, path: List[str]) -> Dict[str, Any]:
        current_path = path + [node.title]
        js = {
            "title": node.title,
            "tag": node.tag,
            "content": match_content(current_path),
            "children": []
        }
        if node.children:
            js["children"] = [node_to_dict(c, current_path) for c in node.children]
        return js

    if not outline_roots:
        return {"title": "ROOT", "tag": "", "children": [], "content": ""}

    if len(outline_roots) == 1:
        return node_to_dict(outline_roots[0], [])
    else:
        return {
            "title": "ROOT",
            "tag": "",
            "content": "",
            "children": [node_to_dict(r, []) for r in outline_roots]
        }

# -----------------------------
# 主流程
# -----------------------------

def build_structure(outline_path: str, original_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    with open(outline_path, 'r', encoding='utf-8') as f:
        outline_md = f.read()
    with open(original_path, 'r', encoding='utf-8') as f:
        original_md = f.read()

    outline_roots = parse_outline(outline_md)
    original_map = build_original_path_map(original_md)
    result = attach_content_from_original(outline_roots, original_map)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def main():
    parser = argparse.ArgumentParser(description="将大纲 Markdown 与原文 Markdown 组装为嵌套 JSON")
    parser.add_argument("--outline", help="大纲 Markdown 文件路径（含 #/##/### 和 <tag>...）", default="文字叙述型文档/0/outline.md")
    parser.add_argument("--original", help="原文 Markdown 文件路径", default="文字叙述型文档/0/full_content.md")
    parser.add_argument("--output", help="输出 JSON 文件路径（缺省则打印到 stdout）", default="文字叙述型文档/0/section_content.json")
    args = parser.parse_args()

    result = build_structure(args.outline, args.original, args.output)
    if not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
