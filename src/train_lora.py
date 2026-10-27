from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.message import to_message

ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = ROOT_PATH / 'data'


def train():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05
    )

    # 模型分词器
    model_name = "/root/models/Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')

    peft_model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 数据
    dataset_dict = load_dataset('json', data_files={
        'train': str(DATA_PATH / 'train.jsonl'),
        'validation': str(DATA_PATH / 'validation.jsonl')
    })

    dataset_dict = dataset_dict.map(to_message, batched=False, fn_kwargs={'include_assistant': True},
                                    remove_columns=['text', 'intent', 'slots'])

    config = SFTConfig(
        max_length=10240,
        assistant_only_loss=True,
        output_dir=str(ROOT_PATH / 'outputs'),
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=100,
        optim='adamw_torch_fused',
        weight_decay=0.01,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=False,
        report_to='tensorboard',
        logging_strategy='steps',
        logging_steps=20,
        eval_strategy='steps',
        eval_steps=20,
        per_device_eval_batch_size=16,
        save_strategy='steps',
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=config,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model('/root/models/Qwen/Qwen3-0.6B-sft-lora')


if __name__ == '__main__':
    train()
