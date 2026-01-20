import os
import random
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AnimalSubset(Dataset):
    """
    从 CIFAR-10 中抽取指定动物类别，限制每类样本数量，并将原始标签映射到 [0, C-1]。
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: List[int],
        indices: List[int],
        class_indices: List[int],
        transform=None,
    ):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.transform = transform
        self.class_indices = class_indices
        # 例如 {2:0, 3:1, 4:2, 5:3, 6:4, 7:5}
        self.label_map = {orig: i for i, orig in enumerate(class_indices)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img = self.data[real_idx]
        label = self.targets[real_idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_map[label]
        return img, label


def build_animal_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    max_train_per_class: int = 500,
    max_val_per_class: int = 100,
    max_test_per_class: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    animal_class_indices = [2, 3, 4, 5, 6, 7]
    animal_class_names = ["bird", "cat", "deer", "dog", "frog", "horse"]
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"
            ),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    base_train = CIFAR10(root=data_dir, train=True, download=True)
    base_test = CIFAR10(root=data_dir, train=False, download=True)

    train_data = base_train.data
    train_targets = base_train.targets
    test_data = base_test.data
    test_targets = base_test.targets

    # 按类别收集索引
    train_indices_per_class = {c: [] for c in animal_class_indices}
    for idx, label in enumerate(train_targets):
        if label in train_indices_per_class:
            if len(train_indices_per_class[label]) < max_train_per_class + max_val_per_class:
                train_indices_per_class[label].append(idx)

    # 拆分训练集和验证集索引
    train_indices = []
    val_indices = []
    for c in animal_class_indices:
        indices_c = train_indices_per_class[c]
        train_indices_c = indices_c[:max_train_per_class]
        val_indices_c = indices_c[max_train_per_class : max_train_per_class + max_val_per_class]
        train_indices.extend(train_indices_c)
        val_indices.extend(val_indices_c)

    # 按类别收集测试集索引，并限制每类样本数量
    test_indices_per_class = {c: [] for c in animal_class_indices}
    for idx, label in enumerate(test_targets):
        if label in test_indices_per_class:
            if len(test_indices_per_class[label]) < max_test_per_class:
                test_indices_per_class[label].append(idx)
    test_indices = []
    for c in animal_class_indices:
        test_indices.extend(test_indices_per_class[c])

    train_dataset = AnimalSubset(
        data=train_data,
        targets=train_targets,
        indices=train_indices,
        class_indices=animal_class_indices,
        transform=train_transform,
    )
    val_dataset = AnimalSubset(
        data=train_data,
        targets=train_targets,
        indices=val_indices,
        class_indices=animal_class_indices,
        transform=eval_transform,
    )
    test_dataset = AnimalSubset(
        data=test_data,
        targets=test_targets,
        indices=test_indices,
        class_indices=animal_class_indices,
        transform=eval_transform,
    )

    # 打印实际样本数量，方便在报告中引用
    print("Train samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))
    print("Test samples: ", len(test_dataset))

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, animal_class_names


def build_model(num_classes: int = 6, freeze_backbone: bool = True) -> nn.Module:
    """
    使用 ImageNet 预训练的 ResNet-18 作为特征提取 backbone，
    将最后的全连接层替换为适应 6 类动物的新分类头。
    """

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )

    return model


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_with_predictions(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    cm: np.ndarray, classes: List[str], title: str, save_path: str = None
):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def show_misclassified_examples(
    images: torch.Tensor,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    max_examples: int = 16,
    imagenet_mean=(0.485, 0.456, 0.406),
    imagenet_std=(0.229, 0.224, 0.225),
):

    mis_idx = np.where(true_labels != pred_labels)[0]
    if len(mis_idx) == 0:
        print("当前批次没有误分类样本。")
        return

    mis_idx = mis_idx[:max_examples]
    images = images.cpu().numpy()
    # 反归一化
    mean = np.array(imagenet_mean).reshape(1, 3, 1, 1)
    std = np.array(imagenet_std).reshape(1, 3, 1, 1)
    images = std * images + mean
    images = np.clip(images, 0, 1)
    images = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)

    cols = 4
    rows = int(np.ceil(len(mis_idx) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(mis_idx):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[idx])
        t = class_names[int(true_labels[idx])]
        p = class_names[int(pred_labels[idx])]
        plt.title(f"T:{t} / P:{p}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 32
    num_epochs = 15
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 3  # 早停容忍轮数

    # 构建数据加载器（小样本动物子集）
    train_loader, val_loader, test_loader, class_names = build_animal_loaders(
        data_dir="./data",
        batch_size=batch_size,
        max_train_per_class=500,
        max_val_per_class=100,
        max_test_per_class=100,
    )

    # 构建迁移学习模型
    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)

    # 使用带 label smoothing 的交叉熵损失，以提升泛化性能
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 使用 AdamW 优化器 + L2 权重衰减
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print("开始训练...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, criterion, optimizer
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        scheduler.step()

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"验证集精度连续 {patience} 轮未提升，触发早停机制。"
                )
                break

    print(f"Best Val Acc: {best_val_acc:.4f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 在测试集上进行最终评估
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    # 混淆矩阵与分类报告
    y_true, y_pred = evaluate_with_predictions(model, device, test_loader)
    cm = confusion_matrix(y_true, y_pred)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(
        cm,
        classes=class_names,
        title="Confusion Matrix (Animal Transfer Learning)",
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # 展示一批测试样本中的误分类案例
    test_iter = iter(test_loader)
    images_batch, labels_batch = next(test_iter)
    images_batch_device = images_batch.to(device)
    with torch.no_grad():
        outputs_batch = model(images_batch_device)
        _, preds_batch = outputs_batch.max(1)

    show_misclassified_examples(
        images_batch,
        labels_batch.numpy(),
        preds_batch.cpu().numpy(),
        class_names=class_names,
        max_examples=16,
    )

    print("实验结束，结果图已保存在 outputs 目录。")


if __name__ == "__main__":
    main()
