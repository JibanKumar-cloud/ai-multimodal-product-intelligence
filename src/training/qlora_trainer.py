"""QLoRA fine-tuning pipeline — supports text-only and multimodal modes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

os.environ.setdefault("WANDB_MODE", "disabled")


def setup_qlora_model(config: dict) -> tuple:
    from transformers import (
        AutoTokenizer, LlavaForConditionalGeneration,
        AutoProcessor, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    logger.info(f"Setting up QLoRA for {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config["model"]["quantization_config"].get(
            "bnb_4bit_use_double_quant", True
        ),
        bnb_4bit_quant_type=config["model"]["quantization_config"].get(
            "bnb_4bit_quant_type", "nf4"
        ),
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(model_name)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")

    return model, tokenizer, processor, peft_config


def train(config: dict) -> Path:
    from transformers import TrainingArguments, Trainer
    from src.data.dataset import AttributeExtractionDataset, MultimodalCollateFunction

    # Detect mode from config
    train_file = config["data"]["train_file"]
    mode = "multimodal" if "multimodal" in train_file else "text"

    logger.info("=" * 60)
    logger.info(f"STARTING QLoRA FINE-TUNING (mode={mode})")
    logger.info("=" * 60)

    model, tokenizer, processor, peft_config = setup_qlora_model(config)

    data_path = Path(config["data"]["dataset_path"])
    max_len = config["data"].get("max_seq_length", 512)
    img_size = config["data"].get("max_image_size", 336)

    train_ds = AttributeExtractionDataset(
        data_path=data_path / config["data"]["train_file"],
        tokenizer=tokenizer, processor=processor,
        max_length=max_len, image_size=img_size, mode=mode,
    )
    val_ds = AttributeExtractionDataset(
        data_path=data_path / config["data"]["val_file"],
        tokenizer=tokenizer, processor=processor,
        max_length=max_len, image_size=img_size, mode=mode,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    tc = config["training"]
    output_dir = Path(tc["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        warmup_ratio=tc["warmup_ratio"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        max_grad_norm=tc["max_grad_norm"],
        fp16=tc.get("fp16", True),
        bf16=tc.get("bf16", False),
        logging_steps=tc["logging_steps"],
        eval_strategy="steps",
        eval_steps=tc["eval_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=tc.get("greater_is_better", False),
        dataloader_num_workers=0,
        seed=tc.get("seed", 42),
        report_to="none",
        remove_unused_columns=False,
    )

    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=MultimodalCollateFunction(),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5,    # stop if no improvement for 3 evals
            early_stopping_threshold=0.01, # minimum improvement to count
        )],
    )

    logger.info("Starting training...")
    result = trainer.train()
    logger.info(f"Training loss: {result.training_loss:.4f}")

    best_path = output_dir / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    config["training_mode"] = mode
    with open(best_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Best model saved to {best_path}")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    return best_path
