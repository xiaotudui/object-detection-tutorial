import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.image_files = sorted(
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        class_id = 0
        bbox = [0.5, 0.5, 0.1, 0.1]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]

        class_id = torch.tensor(class_id, dtype=torch.long)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, class_id, bbox

if __name__ == "__main__":
    # Demo: 定义一套常见的数据增强 + 标准化流程
    demo_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images_dir = "/Users/xiaotudui/Downloads/Custom_Data/images/train"
    labels_dir = "/Users/xiaotudui/Downloads/Custom_Data/labels/train"
    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
        print("请先准备数据目录: dataset/images/train 和 dataset/labels/train")
    else:
        demo_dataset = YOLODataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            transform=demo_transform,
        )
        print(f"数据集大小: {len(demo_dataset)}")
        image, class_id, bbox = demo_dataset[0]
        print(f"样本 image shape: {tuple(image.shape)}")
        print(f"样本 class_id: {class_id.item()}")
        print(f"样本 bbox (xywh, norm): {bbox.tolist()}")