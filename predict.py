import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from model import SimpleDetector

COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def predict(args):
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    class_names = args.class_names or [str(i) for i in range(args.num_classes)]

    model = SimpleDetector(num_classes=args.num_classes).to(device)
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )
    model.eval()

    orig_image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = orig_image.size

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(orig_image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, bbox_pred = model(input_tensor)

    class_id = class_logits.argmax(dim=1).item()
    confidence = torch.softmax(class_logits, dim=1)[0, class_id].item()
    xc, yc, w, h = bbox_pred[0].cpu().tolist()

    x1 = int((xc - w / 2) * orig_w)
    y1 = int((yc - h / 2) * orig_h)
    x2 = int((xc + w / 2) * orig_w)
    y2 = int((yc + h / 2) * orig_h)

    class_name = class_names[class_id]
    color = COLORS[class_id % len(COLORS)]

    print(f"检测结果: {class_name} ({confidence:.2%})")
    print(f"边界框 (xyxy): [{x1}, {y1}, {x2}, {y2}]")

    draw = ImageDraw.Draw(orig_image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    label = f"{class_name} {confidence:.0%}"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
    draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)

    orig_image.save(args.output)
    print(f"结果已保存至 {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单目标检测推理")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="模型权重")
    parser.add_argument("--num-classes", type=int, default=3, help="类别数量")
    parser.add_argument("--class-names", type=str, nargs="+", default=None, help="类别名称")
    parser.add_argument("--img-size", type=int, default=224, help="输入尺寸")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--output", type=str, default="result.jpg", help="输出图片路径")

    predict(parser.parse_args())
