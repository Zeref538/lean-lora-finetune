"""
LoRA SFT on compressed (prompt, completion) pairs. One config = one run
(configs/qwen_lora.yaml). Uses Unsloth + TRL — no custom trainer.

Usage:
    python train.py --config configs/qwen_lora.yaml
"""
import argparse

import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["base_model"], max_seq_length=cfg["train"]["max_seq_len"], load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        random_state=cfg["seed"],
    )

    ds = load_dataset("json", data_files=cfg["train"]["data_path"], split="train")
    ds = ds.map(lambda r: {"text": f"{r['prompt']}\n\n{r['completion']}{tokenizer.eos_token}"})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=cfg["train"]["output_dir"],
            num_train_epochs=cfg["train"]["epochs"],
            per_device_train_batch_size=cfg["train"]["batch_size"],
            gradient_accumulation_steps=cfg["train"]["grad_accum"],
            learning_rate=cfg["train"]["lr"],
            max_seq_length=cfg["train"]["max_seq_len"],
            seed=cfg["seed"],
            logging_steps=10,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(cfg["train"]["output_dir"] + "/adapter")


if __name__ == "__main__":
    main()
