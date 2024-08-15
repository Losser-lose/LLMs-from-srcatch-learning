"""
完成instruction任务的微调

实现流程：
    1. 数据准备与预处理
        - 导入原始数据
        - 构建Dataset和Dataloader
    2. 导入模型
    3. 微调模型
    4. 保存训练结果
    待保存的结果有
        - 回答结果
        - 模型参数
        
"""

import os
import sys
import urllib.request
import json
import re

sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from pathlib import Path
import tiktoken
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import urllib
from functools import partial

from utils.gpt_download import download_and_load_gpt2, load_weights_into_gpt
from utils.text_generation import (
    generate_text_simple,
    generate_and_print_sample,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from model.gpt_model import GPTModel
from utils.plot import plot_values


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        # 使用url提取数据
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")

        # 将数据存储到file_path中
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # 从file_path中导出数据
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text
    
    
def format_result(entry):
    instruction_text = f"### Instruction: \n{entry["instruction"]}"
    input_text = f"\n\n### Input: \n{entry["input"]}" if entry['input'] else ""
    output_text = f"\n\n### Input: \n{entry["output"]}" if entry["output"] else ""
    response_text = f"\n\n### Input: \n{entry["model_response"]}" if entry["model_response"] else ""
    
    return instruction_text + input_text + output_text + response_text
 

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    # 如果num_batches不为None，新生成一个随机抽取的data_loader
    if num_batches is not None and num_batches <= len(data_loader):
        data_loader = get_random_batches(data_loader, num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / len(data_loader)


def get_random_batches(data_loader, num_batches):
    """
    基于num_batches生成一个新的随机的data_loader

    """
    # 获取 DataLoader 的 batch_size
    batch_size = data_loader.batch_size

    # 获取原loader的collect_fn
    collate_fn = data_loader.collate_fn

    # 获取数据集的所有索引
    dataset = data_loader.dataset
    total_samples = len(dataset)

    # 随机选择一些索引
    random_indices = torch.randperm(total_samples).tolist()

    # 选择前 num_batches * batch_size 个索引
    selected_indices = random_indices[: num_batches * batch_size]

    # 创建子集
    subset = Subset(dataset, selected_indices)

    # 创建新的 DataLoader
    random_data_loader = DataLoader(
        subset, batch_size=batch_size, collate_fn=collate_fn
    )

    return random_data_loader


def evaluate_model(model, train_loader, val_loader, device, num_batches=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 修改原始数据格式并转化为token
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.data)


def custom_collect_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    batch_max_length = max([len(item) + 1 for item in batch])

    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        # 添加一个<|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])  # 相当于将inputs右移一个单位

        # 除了第一个pad_token_id外，将其他的pad_token_id全部转换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 如果allowed_max_length不为None的话，对input和target进行截断
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_num_batches,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1

    p_bar = tqdm(range(num_epochs), total=num_epochs, desc="Num of epoch")
    for epoch in p_bar:
        model.train()

        loader_p_bar = tqdm(train_loader)
        for input_batch, target_batch in loader_p_bar:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    num_batches=eval_num_batches,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)

        # 每训练完一个epoch展示结果
        train_loss, val_loss = evaluate_model(
            model,
            train_loader,
            val_loader,
            device,
            num_batches=eval_num_batches,
        )
        print(f"Epoch {epoch + 1}:", end="")
        print(
            f"Training loss {train_loss:.2f} | ",
            end="",
        )
        print(f"Validation loss {val_loss:.2f}")
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_token_seen


def main():
    # 基本参数设置
    batch_size = 8
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据准备与预处理
    # 导入原始数据
    file_path = os.path.abspath(
        os.path.join(os.getcwd(), "./data/instruction_data.json")
    )
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    # 创建Dataset和Dataloader
    # 数据集划分
    train_portion = int(len(data) * 0.8)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50 * "-")

    # 构建customized_collect_fn
    customized_collect_fn = partial(
        custom_collect_fn, device=device, allowed_max_length=1024
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collect_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collect_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collect_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # 2. 导入模型
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=os.path.abspath(os.path.join(os.getcwd(), "./data/models/gpt2")),
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)
    print("Loaded model:", CHOOSE_MODEL)
    print(50 * "-")

    # 3. 微调模型
    # 计算初始的loss
    print("Initial Loss")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"    Training loss: {train_loss: .4f}")
    print(f"    Validation loss: {val_loss: .4f}")

    # 开始训练
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_num_batches=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    print(50 * "-")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_tensor, tokens_seen, train_losses, val_losses, label="loss")

    # 4. 保存训练结果
    # 保存问答结果
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=100,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        generate_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generate_text[len(input_text) :]
            .replace("### Response:", "")
            .replace("\n", " ")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    test_data_path = os.path.abspath(
        os.path.join(
            os.getcwd(), "./data/instruction-data-with-response.json"
        )
    )
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # 保存模型参数
    file_name = os.path.abspath(
        os.path.join(
            os.getcwd(),
            "./data/models/"
            + f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth",
        )
    )
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    # Load model via
    # model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
    
    # 最终结果展示
    for i, entry in enumerate(test_data):
        print(50 * "-")
        print(f"Question {i+1}")
        result = format_result(entry)
        print(result)


if __name__ == "__main__":
    main()
