import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.message import to_message

ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = ROOT_PATH / 'data'


def evaluate(model, tokenizer):
    # 加载测试集
    dataset = load_dataset('json', data_files=str(DATA_PATH / 'test.jsonl'))['train']

    # 处理数据
    dataset = dataset.map(to_message, batched=False, fn_kwargs={"include_assistant": False})

    # 评估模型
    batch_size = 32
    correct_num = 0
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        batch_text = tokenizer.apply_chat_template(
            batch['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        model_inputs = tokenizer(batch_text, padding=True, padding_side='left', return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1
        )
        # generated_ids.shape: [batch_size, seq_len]
        input_length = model_inputs['input_ids'].shape[1]
        output_ids = generated_ids[:, input_length:]

        batch_output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # 计算指标
        for output_text, intent, slots in zip(batch_output_text, batch['intent'], batch['slots']):

            try:
                output_dict = json.loads(output_text)
            except Exception:
                output_dict = None

            label_dict = {
                "intent": intent,
                "slots": slots
            }

            if output_dict == label_dict:
                correct_num += 1

    return correct_num / len(dataset)


if __name__ == '__main__':
    # 基线测试 0.0238
    # model_name = "/root/models/Qwen/Qwen3-0.6B"
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto" )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print(evaluate(model, tokenizer))

    # 全参微调 0.6166
    # model_name = "/root/models/Qwen/Qwen3-0.6B-sft-full"
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto" )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    # print(evaluate(model, tokenizer))

    # LoRA微调
    # model_name = "/root/models/Qwen/Qwen3-0.6B-sft-lora"
    # model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto" )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    # print(evaluate(model, tokenizer))

    # QLoRA微调
    model_name = "/root/models/Qwen/Qwen3-0.6B-sft-qlora"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto",
                                                     quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(evaluate(model, tokenizer))
