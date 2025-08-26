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
OUTPUT_JSON_NAME = "split_snippet_test.json"

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
你是一个文档分片助手。请根据语义将### Content中的内容分割为若干语义完整的段落，每个分割后的片段不应该低于两个完整的句子(需要以句号结束才叫做句子，逗号不算)。
如果分隔的某个片段只有一个句子则可以考虑将其合并到其他的片段。
你的分隔应该在语义相近的情况下分隔的片段尽量的长(若都是描述的同一个主题则无需分隔)。
比如像下面的内容应该都是一个完整片段而不应该分隔：
Example
-------
1. PBST、PBS、Tween-20。  
2. SDS-PAGE 电泳缓冲液。  
3. Western blot 转膜缓冲液。  
4. 10×TBS 缓冲液。  
5. 1×TBST 缓冲液。  
6. 封闭缓冲液：5% 的脱脂奶粉溶液。  
7. 一抗/二抗稀释缓冲液。  
-------
8. 转膜用的夹子、两块海绵垫、一支滴管、滤纸、一张 PVDF 膜、转膜槽、转移电泳仪、摇床、计时器、磁力搅拌器、转子、Western blot 盒、SDS-PAGE 胶、脱脂奶粉。  
你分割后的各片段拼接起来之后应该完整还原原 Content,包括标点符号和标题的符号以及所有换行符。不要漏掉任何一个字符。

要求：
1) 输出必须是 JSON 数组，数组元素均为分隔出来的字符串。
2) 数组元素必须和原文中对应内容完全一致（不要改写、不要增删字符）。
3) 不要包含任何解释性文字。

### Content:
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


def process_case_dir(case_dir: Path, model: str):
    """
    处理一个 case* 目录：
    - 读取 section_content.json
    - 提取所有 content，逐段调用分片
    - 汇总写入 split_snippet.json
    """
    in_path = case_dir / INPUT_JSON_NAME
    out_path = case_dir / OUTPUT_JSON_NAME

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


def is_case_dir(p: Path, prefix: str = "case") -> bool:
    """
    判断目录名是否为形如 '<prefix><非负整数>' 的目录，例如 'case0' / 'case1' / ...
    """
    if not p.is_dir():
        return False
    return re.fullmatch(fr'{re.escape(prefix)}\d+', p.name) is not None


def process_root(root: Path, model: str, case_prefix: str = "case"):
    """
    遍历根目录：
    - 仅遍历第一层中所有形如 '<prefix><数字>' 的子目录（默认 'case0/ case1/ ...'）
    - 每个子目录中直接寻找并处理 section_content.json
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"根目录不存在或不是目录：{root}")

    case_dirs = [p for p in root.iterdir() if is_case_dir(p, case_prefix)]
    if not case_dirs:
        print(f"[WARN] 根目录下未发现 '{case_prefix}<数字>' 形式的子目录：{root}")
        return

    # 按数字序排序（case10 > case2）
    def case_index(path: Path) -> int:
        m = re.search(r'(\d+)$', path.name)
        return int(m.group(1)) if m else 0

    case_dirs.sort(key=case_index)

    for d in case_dirs:
        print(f"[DIR] {d.name}")
        process_root.current_dir = d.name
        process_case_dir(d, model=model)


def main():
    parser = argparse.ArgumentParser(
        description="遍历根目录下的 case* 子目录，调用 OpenAI 进行内容分片。"
    )
    parser.add_argument("--root", type=str, default="./", help="数据集根目录，例如：/path/to/dataset_root")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI 模型名（默认：gpt-4o）")
    parser.add_argument("--case-prefix", type=str, default="case", help="子目录前缀（默认：case）")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"[START] 根目录：{root}")
    process_root(root, model=args.model, case_prefix=args.case_prefix)
    print("[DONE] 全部处理完成。")


if __name__ == "__main__":
    main()
