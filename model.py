import torch.nn as nn


class SimpleDetector(nn.Module):
    """简单的单目标检测模型

    结构: 5层卷积提取特征 -> 全局平均池化 -> 两个并行的全连接头
      - 分类头: 输出 num_classes 个类别的 logits
      - 回归头: 输出 4 个归一化的边界框坐标 (x_center, y_center, w, h)
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            self._conv_block(3, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)

        class_logits = self.classifier(x)
        bbox_pred = self.regressor(x)

        return class_logits, bbox_pred


if __name__ == "__main__":
    model = SimpleDetector(num_classes=1)
    print(model)