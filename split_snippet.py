#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI, APIError, RateLimitError

# ========= 可按需修改的默认文件名 =========
INPUT_JSON_NAME = "section_content.json"
OUTPUT_JSON_NAME = "split_snippet.json"

# ========= 代理（如不需要可注释掉）=========
# os.environ.setdefault("http_proxy", "http://172.17.0.1:7890")
# os.environ.setdefault("https_proxy", "http://172.17.0.1:7890")
# os.environ.setdefault("HTTP_PROXY", "http://172.17.0.1:7890")
# os.environ.setdefault("HTTPS_PROXY", "http://172.17.0.1:7890")

# ========= OpenAI 客户端 =========
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ========= 工具函数 =========
def extract_contents(node: Dict[str, Any]) -> List[str]:
    """递归提取所有 'content' 字段（仅收集非空字符串）"""
    contents = []
    if isinstance(node, dict):
        if "content" in node and isinstance(node["content"], str) and node["content"].strip():
            contents.append(node["content"])
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                contents.extend(extract_contents(child))
    elif isinstance(node, list):
        # 万一根就是一个列表
        for item in node:
            contents.extend(extract_contents(item))
    return contents


def _best_effort_json_loads(text: str):
    """
    尝试从返回文本中提取 JSON 数组：
    - 去除 ```json ... ``` 或 ``` ... ``` 代码块包裹
    - 尝试用正则抽取第一个以 [ 开始、以 ] 结束的片段
    """
    if not isinstance(text, str):
        raise json.JSONDecodeError("not a string", doc=str(text), pos=0)

    # 去除三引号代码块
    fence = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    m = fence.search(text)
    if m:
        text = m.group(1).strip()

    # 若仍非纯数组，尝试抓取第一个 JSON 数组片段
    if not text.strip().startswith("["):
        arr = re.search(r"\[\s*[\s\S]*?\s*\]", text)
        if arr:
            text = arr.group(0)

    return json.loads(text)


def split_with_gpt(content: str, model: str = "gpt-4o", temperature: float = 0.0,
                   max_retries: int = 3, retry_base_sleep: float = 2.0) -> List[str]:
    """
    调用 GPT 将一段内容按语义切片为字符串数组。
    - 解析失败或接口异常时做有限次数重试
    - 最终仍失败则回退为 [content]
    """
    prompt = f"""
你是一个文档分片助手。请根据语义将下面的内容分割为若干语义完整的片段，片段的最小分隔不应该低于两个个完整的句子。
你分割时候的内容拼接起来之后应该是完整的contetn内容
要求：
1) 输出必须是 JSON 数组，数组元素均为分隔出来的字符串。
2) 数组元素必须和原文中对应内容完全一致。

### Content：
{content}
""".strip()

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个严格的助手，只输出符合要求的 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content_out = resp.choices[0].message.content
            try:
                slices = _best_effort_json_loads(content_out)
                # 只保留字符串条目
                slices = [s for s in slices if isinstance(s, str) and s.strip()]
                if not slices:
                    raise ValueError("Empty slices produced.")
                return slices
            except Exception:
                if attempt >= max_retries:
                    return [content]
                time.sleep(retry_base_sleep * attempt)
        except (RateLimitError, APIError) as e:
            # 简单指数退避
            if attempt >= max_retries:
                print(f"[ERROR] OpenAI 调用失败（已达最大重试次数）: {e}")
                return [content]
            sleep_s = retry_base_sleep * attempt
            print(f"[WARN] OpenAI 调用异常，{sleep_s:.1f}s 后重试（第 {attempt}/{max_retries} 次）: {e}")
            time.sleep(sleep_s)
        except Exception as e:
            # 其他未知异常：不再无限重试，按上限处理
            if attempt >= max_retries:
                print(f"[ERROR] 调用异常（已达最大重试次数）: {e}")
                return [content]
            time.sleep(retry_base_sleep * attempt)

    # 理论不达
    return [content]


def process_number_dir(number_dir: Path, model: str):
    """
    处理一个编号目录：
    - 读取 section_content.json
    - 提取所有 content，逐段调用分片
    - 汇总写入 split_snippet.json
    """
    in_path = number_dir / INPUT_JSON_NAME
    out_path = number_dir / OUTPUT_JSON_NAME

    if not in_path.exists():
        print(f"[SKIP] 找不到输入文件：{in_path}")
        return

    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 JSON 失败，跳过：{in_path} ({e})")
        return

    all_contents = extract_contents(data)
    #print(all_contents[0:5])
    if not all_contents:
        print(f"[WARN] 未提取到任何 content，跳过：{in_path}")
        return

    all_slices: List[str] = []
    for idx, content in enumerate(all_contents, 1):
        print(f"  - 处理段落 {idx}/{len(all_contents)} ...")
        slices = split_with_gpt(content, model=model)
        all_slices.extend(slices)

    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_slices, f, ensure_ascii=False, indent=2)
        print(f"[OK] 已写出：{out_path}")
    except Exception as e:
        print(f"[ERROR] 写文件失败：{out_path} ({e})")


def is_integer_dir(p: Path) -> bool:
    """判断目录名是否为非负整数（0/1/2/...）"""
    return p.is_dir() and p.name.isdigit()


def process_root(root: Path, model: str):
    """
    遍历根目录：
    - 第一层：类型子目录（示例：公式密集型文档 / 引用密集型文档 / 实体密集型文档 / 文字叙述型文档）
      —— 实际上不做名称强约束，凡是目录都进入
    - 第二层：编号子目录（0/1/2/...）
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"根目录不存在或不是目录：{root}")

    type_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not type_dirs:
        print(f"[WARN] 根目录下未发现类型子目录：{root}")
        return

    for type_dir in type_dirs:
        print(f"\n[TYPE] {type_dir.name}")
        number_dirs = [p for p in type_dir.iterdir() if is_integer_dir(p)]
        if not number_dirs:
            print(f"  [WARN] 未找到编号子目录（0/1/2/...）：{type_dir}")
            continue
        number_dirs.sort(key=lambda p: int(p.name))
        for nd in number_dirs:
            print(f"[DIR] {type_dir.name}/{nd.name}")
            process_root.current_dir = f"{type_dir.name}/{nd.name}"
            process_number_dir(nd, model=model)


def main():
    parser = argparse.ArgumentParser(description="批量遍历根目录下的类型与编号子目录，调用 OpenAI 进行内容分片。")
    parser.add_argument("--root", type=str, default="./", help="数据集根目录，例如：/path/to/dataset_root")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI 模型名（默认：gpt-4o）")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"[START] 根目录：{root}")
    process_root(root, model=args.model)
    print("[DONE] 全部处理完成。")


if __name__ == "__main__":
    main()
