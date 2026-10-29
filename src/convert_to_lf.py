"""
    @Author:MaYuhang
    @Time:2026/10/29
    @DESC:
"""
"""把 data/*.jsonl 转成 LLaMA-Factory 的 sharegpt 格式。

任意位置、任意方式运行均可：
    python src/convert_to_lf.py               # 仓库根目录，输出到云端默认路径
    python -m src.convert_to_lf               # 模块方式
    python src/convert_to_lf.py D:/out/dir    # 自定义输出目录
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 仓库根目录
sys.path.insert(0, str(ROOT))                # 位置无关：保证能找到 src 包

from src.message import SYSTEM_PROMPT  # 复用手写的 prompt，保证逐字一致

DATA = ROOT / 'data'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/sj-tmp/LLaMA-Factory/data')


def convert(name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'nlu_{name}.jsonl'
    with open(DATA / f'{name}.jsonl', encoding='utf-8') as fin, \
         open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            ex = json.loads(line)
            rec = {
                'conversations': [
                    {'from': 'human', 'value': f"用户文本：{ex['text']}"},   # 对齐 to_message 的 user 模板
                    {'from': 'gpt', 'value': json.dumps(
                        {'intent': ex['intent'], 'slots': ex['slots']},
                        ensure_ascii=False)},                                  # 对齐 assistant 的 json.dumps
                ],
                'system': SYSTEM_PROMPT,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'{out}: done')


if __name__ == '__main__':
    convert('train')
    convert('validation')