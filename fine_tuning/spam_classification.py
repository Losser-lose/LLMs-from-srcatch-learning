import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import tiktoken
from tqdm import tqdm
import time

from utils.data_download import download_and_unzip_spam_data  # 准备数据模块
from utils.datasets import SpamDataset
from utils.gpt_download import (
    load_gpt2_params_from_tf_ckpt,
    download_and_load_gpt2,
    load_weights_into_gpt,
)
from model.gpt_model import GPTModel
from utils.text_generation import (
    token_ids_to_text,
    text_to_token_ids,
    generate_text_simple,
)
from utils.evaluation import calc_accuracy_loader, calc_loss_batch, calc_loss_loader


# 设置随机种子
torch.manual_seed(123)

# 数据配置
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "/Users/kingsun/Desktop/LLM/projects/my-projects/LLMs-from-srcatch-learning/data/sms_spam_collection/sms_spam_collection.zip"
extracted_path = "/Users/kingsun/Desktop/LLM/projects/my-projects/LLMs-from-srcatch-learning/data/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def create_balanced_dataset(df):

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq):
    # Initialize lists to track losses and examples seen
    model.to(device)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    p_bar = tqdm(range(num_epochs), total=num_epochs, desc="Num of epoch")
    for epoch in p_bar:

        epoch_p_bar = tqdm(train_loader)
        for input_batch, target_batch in epoch_p_bar:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            # if global_step % eval_freq == 0:
            #     train_loss, val_loss = evaluate_model(
            #         model, train_loader, val_loader, device
            #     )
            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)

        # 每训练完一个epoch记录结果
        train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device
                )
        train_accuracy = calc_accuracy_loader(train_loader, model, device)
        val_accuracy = calc_accuracy_loader(val_loader, model, device)
        print(f"Epoch {epoch + 1}:", end="")
        print(f"Training accuracy: {train_accuracy*100:.2f}% with training loss {train_loss:.2f} | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}% with validation loss {val_loss:.2f}")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# def train(
#     model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
# ):
#     # Initialize lists to track losses and examples seen
#     model.to(device)
#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     examples_seen, global_step = 0, -1

#     # Main training loop
#     p_bar = tqdm(range(num_epochs), total=num_epochs, desc="Num of epoch")
#     for epoch in p_bar:
#         model.train()  # Set model to training mode

#         epoch_p_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
#         for input_batch, target_batch in epoch_p_bar:
#             optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()  # Calculate loss gradients
#             optimizer.step()  # Update model weights using loss gradients
#             examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
#             global_step += 1

#             # Optional evaluation step
#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)

#         # Calculate accuracy after each epoch
#         train_accuracy = calc_accuracy_loader(
#             train_loader, model, device
#         )
#         val_accuracy = calc_accuracy_loader(
#             val_loader, model, device
#         )
#         print(f"Epoch {epoch+1}:", end="")
#         print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
#         print(f"Validation accuracy: {val_accuracy*100:.2f}%")
#         train_accs.append(train_accuracy)
#         val_accs.append(val_accuracy)

#     return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def main():
    # 原始数据准备
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    balanced_df = create_balanced_dataset(df)
    print("=" * 50)
    print(f"Label information for balanced df")
    print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # 构建Dataset以及Dataloader
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(df=train_df, tokenizer=tokenizer)
    val_dataset = SpamDataset(df=val_df, tokenizer=tokenizer)
    test_dataset = SpamDataset(df=test_df, tokenizer=tokenizer)

    batch_size = 8
    num_workers = 0
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    print("=" * 50)
    print(f"Basic information for dataloaders:")
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    # 导入模型
    # 模型基本参数
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"  # 用于测试的输入

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

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    # 导入模型参数
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="/Users/kingsun/Desktop/LLM/projects/my-projects/LLMs-from-srcatch-learning/data/models/gpt2",
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # 检查预训练模型性能
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(INPUT_PROMPT, tokenizer),
        max_new_tokens=20,
        context_size=BASE_CONFIG["context_length"],
    )
    print("=" * 50)
    print(f"To test the pre-trained model, the answer for `{INPUT_PROMPT}` is:")
    print(token_ids_to_text(token_ids, tokenizer))

    # 进行fine-tuning
    # 冻结参数x
    for param in model.parameters():
        param.requires_grad = False

    # 添加分类输出头
    num_classes = 2
    # 将原本的out_head改为新的全链接层，输出维度为2维
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    # 解冻最后一层trf_block
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 进行微调
    start_time = time.time()

    # torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    eval_freq = 50
    device = torch.device("cpu")

    train_losses, val_losses, train_accs, val_accs, examples_seen = train(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()
