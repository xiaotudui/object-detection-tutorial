import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """目标检测损失 = 分类损失 + λ × 边界框回归损失"""

    def __init__(self, bbox_weight=5.0):
        super().__init__()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()
        self.bbox_weight = bbox_weight

    def forward(self, class_logits, bbox_pred, class_target, bbox_target):
        cls_loss = self.cls_loss_fn(class_logits, class_target)
        bbox_loss = self.bbox_loss_fn(bbox_pred, bbox_target)
        total_loss = cls_loss + self.bbox_weight * bbox_loss
        return total_loss, cls_loss, bbox_loss


def compute_iou(pred, target):
    """计算预测框与真实框的IoU (输入为归一化的 x_center, y_center, w, h)"""
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (
        inter_y2 - inter_y1
    ).clamp(min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
    union_area = pred_area + tgt_area - inter_area

    return inter_area / (union_area + 1e-6)


if __name__ == "__main__":
    # Demo: 构造一批假数据，演示损失与 IoU 的计算方式
    batch_size = 4
    num_classes = 3

    class_logits = torch.randn(batch_size, num_classes)
    bbox_pred = torch.rand(batch_size, 4)
    class_target = torch.randint(0, num_classes, (batch_size,))
    bbox_target = torch.rand(batch_size, 4)

    criterion = DetectionLoss(bbox_weight=5.0)
    total_loss, cls_loss, bbox_loss = criterion(
        class_logits, bbox_pred, class_target, bbox_target
    )
    iou = compute_iou(bbox_pred, bbox_target).mean()

    print(f"total_loss: {total_loss.item():.4f}")
    print(f"cls_loss:   {cls_loss.item():.4f}")
    print(f"bbox_loss:  {bbox_loss.item():.4f}")
    print(f"mean IoU:   {iou.item():.4f}")
