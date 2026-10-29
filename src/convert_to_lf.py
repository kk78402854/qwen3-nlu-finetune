"""
    @Author:MaYuhang
    @Time:2026/10/29
    @DESC:
"""
"""把 data/*.jsonl 转成 LLaMA-Factory 的 sharegpt 格式。在 qwen3-nlu-finetune 仓库根目录运行。"""
import json
from pathlib import Path

from src.message import SYSTEM_PROMPT  # 复用手写的 prompt，保证逐字一致

DATA = Path('data')
OUT = Path('/root/sj-tmp/LLaMA-Factory/data')


def convert(name: str):
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