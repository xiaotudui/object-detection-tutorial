# 最简单的目标检测入门教程

从零搭建一个单目标检测模型 —— 不依赖任何预训练权重，用最少的代码理解目标检测的核心思路。

## 项目特点

- **自定义 CNN 模型**：5 层卷积 + 分类头 / 回归头，结构清晰易懂
- **YOLO 格式数据集**：兼容主流标注工具的输出
- **单目标检测**：每张图预测一个物体的**类别**和**边界框**，聚焦核心原理
- **附带合成数据生成脚本**：无需下载真实数据集即可跑通全流程

## 项目结构

```
├── model.py               # SimpleDetector 模型定义
├── dataset.py             # YOLO 格式数据集加载
├── loss.py                # 检测损失函数 (分类 + 回归)
├── train.py               # 训练脚本
├── predict.py             # 推理与可视化
├── create_sample_data.py  # 合成示例数据生成
├── pyproject.toml         # 项目依赖
└── dataset/               # 数据集 (YOLO 格式)
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── classes.txt
```

## 快速开始

### 1. 安装依赖

```bash
# 推荐使用 uv
uv sync

# 或 pip
pip install torch torchvision pillow
```

### 2. 生成示例数据

```bash
python create_sample_data.py
```

会在 `dataset/` 下生成包含圆形、矩形、三角形的合成图片及 YOLO 标注。

### 3. 训练模型

```bash
python train.py --data dataset --num-classes 3 --epochs 50
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `dataset` | 数据集根目录 |
| `--num-classes` | `3` | 类别数 |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `16` | 批量大小 |
| `--lr` | `0.001` | 学习率 |
| `--img-size` | `224` | 输入图像尺寸 |
| `--device` | `auto` | 训练设备 (auto/cpu/cuda/mps) |

### 4. 推理预测

```bash
python predict.py --image dataset/images/val/0000.jpg --num-classes 3 --class-names circle rectangle triangle
```

结果会保存为 `result.jpg`。

## 数据集格式 (YOLO)

每张图片对应一个同名 `.txt` 标注文件，每行格式：

```
class_id  x_center  y_center  width  height
```

所有坐标和尺寸均**归一化到 0~1**（除以图片宽高）。

## 模型结构

```
输入 (3×224×224)
  ↓
5 × [Conv3×3 → BatchNorm → ReLU → MaxPool2×2]
  ↓
AdaptiveAvgPool → 256 维特征
  ↓
┌─────────────────────┬──────────────────────┐
│   分类头 (FC→ReLU→FC)  │   回归头 (FC→ReLU→FC→Sigmoid) │
│   → num_classes logits │   → 4 个归一化坐标         │
└─────────────────────┴──────────────────────┘
```

## 使用自己的数据

1. 按上面的 YOLO 格式准备图片和标注
2. 把图片放到 `dataset/images/train/` 和 `dataset/images/val/`
3. 把标注放到 `dataset/labels/train/` 和 `dataset/labels/val/`
4. 修改 `--num-classes` 和 `--class-names` 参数
