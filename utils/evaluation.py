import torch
from tqdm import tqdm
from torch.utils.data import Subset, RandomSampler, DataLoader


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if len(data_loader) == 0:
        return float("nan")

    # 如果num_batches不为None，新生成一个随机抽取的data_loader
    if num_batches is not None and num_batches <= len(data_loader):
        data_loader = get_random_batches(data_loader, num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # Logits of last output token
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """_summary_

    Args:
        data_loader: Dataloader
        model: 训练模型
        device: 训练设备
        num_batches: 需要计算的batch数量，默认为None，将会对dataloader中的所有数据进行计算；如果存在的话将会从dataloader中随机抽取num_batches个batch进行计算

    Returns:
        _type_: _description_
    """
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
    random_data_loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)

    return random_data_loader


def evaluate_model(model, train_loader, val_loader, device, num_batches=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss
