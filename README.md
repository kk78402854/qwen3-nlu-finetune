# Qwen3-NLU-Finetune · 意图识别与槽位抽取微调实战

> 基于 Qwen3-0.6B 的中文联合 NLU 任务微调：全参 / LoRA / QLoRA 三方案对比

Python 3.10+ · Transformers · TRL · PEFT · bitsandbytes

## 项目简介

这是我的第一个大模型微调实战项目。任务为**联合自然语言理解（Joint NLU）**：输入一条用户话语，模型同时输出**意图分类**（intent）与**槽位抽取**（slots），以严格 JSON 格式返回。

将 NLU 任务建模为"生成式结构化输出"问题，而非传统的分类 + 序列标注双模型 pipeline，在 Qwen3-0.6B 小基座上完成了三种主流微调方案的完整对比与评估。

## 实验结果

| 方案 | 可训练参数占比 | 显存需求 | 测试准确率 |
|------|--------------|---------|-----------|
| 零样本基座（zero-shot） | - | - | 2.38% |
| 全参微调（Full SFT） | 100% | 高 | **61.66%** |
| LoRA | <1% | 中 | 见评估脚本 |
| QLoRA（4bit NF4） | <1% | 低（消费级显卡可跑） | 见评估脚本 |

> 准确率定义：生成的 JSON 与标注完全一致（intent 正确且 slots 的 name/value/顺序完全匹配），全对才算对——比单一指标严格得多。

## 任务定义

**输入**：`用户文本：明天早上八点叫我起床`

**输出**（严格单行 JSON）：

```json
{"intent":"alarm_set","slots":[{"name":"time","value":"明天早上八点"}]}
```

- 60 个候选意图（闹钟/日历/音乐/IoT/出行/天气等任务型对话领域）
- 56 个候选槽位（时间/地点/人名/歌曲名等）
- System Prompt 中约束：value 必须从原文逐字复制、槽位按出现顺序排列、同名槽位不合并

## 三种微调方案

| 脚本 | 方案 | 关键技术 |
|------|------|---------|
| `src/train_full.py` | 全参微调 | lr=2e-5，直接更新全部权重 |
| `src/train_lora.py` | LoRA | r=8 / alpha=16，注入全部 7 类线性投影（q/k/v/o/gate/up/down） |
| `src/train_qlora.py` | QLoRA | NF4 量化 + 双重量化 + bf16 计算精度，基座 4bit 加载 |

三者共享的训练配置：
- **`assistant_only_loss=True`**：只对 assistant 回复计算损失，System Prompt 的 60 意图 + 56 槽位大清单不参与梯度
- 断点续训（`resume_from_checkpoint`）、`load_best_model_at_end` 按 eval_loss 回捞最优 checkpoint
- bf16 混合精度、fused AdamW、warmup 100 步、TensorBoard 记录

## 评估方式

`src/evaluate.py`：批量贪心解码（`do_sample=False`）→ 解析 JSON → 与标注严格等值比较。

三处值得注意的工程细节：
- `padding_side='left'`：生成任务必须左侧 padding，否则右侧 padding 会污染解码
- `enable_thinking=False`：关闭 Qwen3 思考模式，直接输出答案
- 解析失败的输出直接判错（模拟真实生产约束）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据已附带在 data/ 目录（SMP 风格中文任务型对话数据集）
#    train.jsonl 11,514 条 / validation.jsonl 2,033 条 / test.jsonl 2,974 条

# 3. 修改模型路径（默认为 Linux 训练机路径 /root/models/Qwen/Qwen3-0.6B）
#    train_*.py 中的 model_name 换成本地路径或 HF Hub 名称

# 4. 训练（三选一，从项目根目录以模块方式运行）
python -m src.train_full
python -m src.train_lora
python -m src.train_qlora

# 5. 评估（取消 evaluate.py 底部对应模型的注释）
python -m src.evaluate
```

**数据格式**（每行一个 JSON）：

```json
{"text": "明天早上八点叫我起床", "intent": "alarm_set", "slots": [{"name": "time", "value": "明天早上八点"}]}
```

## 目录结构

```
qwen3-nlu-finetune/
├── src/
│   ├── message.py      # System Prompt 与数据 → 对话格式转换
│   ├── train_full.py   # 全参微调
│   ├── train_lora.py   # LoRA 微调
│   ├── train_qlora.py  # QLoRA 微调（4bit 量化）
│   └── evaluate.py     # 严格 JSON 匹配评估
├── data/               # 训练/验证/测试数据（jsonl）
├── requirements.txt
└── README.md
```

## 踩坑记录

- **LoRA 学习率要比全参高一个量级**：全参 2e-5，LoRA/QLoRA 用 2e-4（低秩适配器需要更快的更新速度）
- **量化模型保存后加载要用 `AutoPeftModelForCausalLM`**：LoRA checkpoint 不能用 `AutoModelForCausalLM` 直接读
- **Qwen3 默认开启思考模式**：评估/推理时必须 `enable_thinking=False`，否则输出夹杂推理过程导致 JSON 解析失败

## License

MIT
