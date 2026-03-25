import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import YOLODataset
from loss import DetectionLoss, compute_iou
from model import SimpleDetector


def train(args):
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = YOLODataset(
        images_dir=os.path.join(args.data, "images", "train"),
        labels_dir=os.path.join(args.data, "labels", "train"),
        transform=train_transform,
    )
    val_dataset = YOLODataset(
        images_dir=os.path.join(args.data, "images", "val"),
        labels_dir=os.path.join(args.data, "labels", "val"),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")

    model = SimpleDetector(num_classes=args.num_classes).to(device)
    criterion = DetectionLoss(bbox_weight=args.bbox_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("weights", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ---- 训练 ----
        model.train()
        train_loss_sum = 0.0
        for images, class_ids, bboxes in train_loader:
            images = images.to(device)
            class_ids = class_ids.to(device)
            bboxes = bboxes.to(device)

            class_logits, bbox_pred = model(images)
            loss, _, _ = criterion(class_logits, bbox_pred, class_ids, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        train_avg = train_loss_sum / len(train_dataset)

        # ---- 验证 ----
        model.eval()
        val_loss_sum = 0.0
        val_cls_sum = 0.0
        val_bbox_sum = 0.0
        correct = 0
        iou_sum = 0.0

        with torch.no_grad():
            for images, class_ids, bboxes in val_loader:
                images = images.to(device)
                class_ids = class_ids.to(device)
                bboxes = bboxes.to(device)

                class_logits, bbox_pred = model(images)
                loss, cls_l, bbox_l = criterion(
                    class_logits, bbox_pred, class_ids, bboxes
                )

                val_loss_sum += loss.item() * images.size(0)
                val_cls_sum += cls_l.item() * images.size(0)
                val_bbox_sum += bbox_l.item() * images.size(0)

                preds = class_logits.argmax(dim=1)
                correct += (preds == class_ids).sum().item()
                iou_sum += compute_iou(bbox_pred, bboxes).sum().item()

        n = len(val_dataset)
        val_avg = val_loss_sum / n
        val_acc = correct / n * 100
        val_iou = iou_sum / n

        print(
            f"Epoch [{epoch + 1:>3}/{args.epochs}]  "
            f"Train: {train_avg:.4f}  "
            f"Val: {val_avg:.4f} (cls {val_cls_sum / n:.4f} | bbox {val_bbox_sum / n:.4f})  "
            f"Acc: {val_acc:.1f}%  IoU: {val_iou:.3f}"
        )

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), "weights/best.pt")
            print(f"  ✓ 保存最优模型 (val_loss={val_avg:.4f})")

        scheduler.step()

    torch.save(model.state_dict(), "weights/last.pt")
    print("训练完成！最优模型: weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练简单目标检测模型")
    parser.add_argument("--data", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--num-classes", type=int, default=3, help="类别数量")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--bbox-weight", type=float, default=5.0, help="边界框损失权重")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda/mps)")

    train(parser.parse_args())
