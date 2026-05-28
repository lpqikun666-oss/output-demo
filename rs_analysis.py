"""
Remote Sensing / Urban Scene Analysis Demo
AI-driven semantic segmentation using DeepLabV3.

Demonstrates how AI can classify every pixel in an image into categories
such as road, building, vegetation, car, sky, etc.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision import models


# -- Config ----------------------------------------------------------------
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "background", "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

COLOUR_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def make_street_scene(size=(512, 512)):
    """
    Generate a synthetic street-level urban scene that resembles
    the Cityscapes dataset, so DeepLabV3 can produce meaningful
    segmentation results.
    """
    w, h = size
    img = Image.new("RGB", (w, h), (135, 180, 235))  # sky blue
    draw = ImageDraw.Draw(img)

    # Sky gradient (top 40%)
    for y in range(int(h * 0.4)):
        brightness = int(235 - (y / (h * 0.4)) * 60)
        draw.line([(0, y), (w, y)], fill=(brightness - 100, brightness - 80, brightness))

    # Buildings on the left
    for i in range(3):
        bh = np.random.randint(80, 150)
        bx = np.random.randint(0, w // 3)
        by = int(h * 0.4) - bh + np.random.randint(-20, 40)
        building_color = np.random.randint(120, 200)
        draw.rectangle([bx, by, bx + 80, int(h * 0.7)], fill=(building_color, building_color, building_color + 10))
        # Windows
        for wy in range(by + 10, int(h * 0.7) - 10, 20):
            for wx in range(bx + 10, bx + 70, 20):
                draw.rectangle([wx, wy, wx + 10, wy + 10], fill=(200, 200, 100))

    # Buildings on the right
    for i in range(3):
        bh = np.random.randint(80, 150)
        bx = np.random.randint(w * 2 // 3, w - 80)
        by = int(h * 0.4) - bh + np.random.randint(-20, 40)
        building_color = np.random.randint(120, 200)
        draw.rectangle([bx, by, bx + 80, int(h * 0.7)], fill=(building_color, building_color, building_color + 10))
        for wy in range(by + 10, int(h * 0.7) - 10, 20):
            for wx in range(bx + 10, bx + 70, 20):
                draw.rectangle([wx, wy, wx + 10, wy + 10], fill=(200, 200, 100))

    # Road (bottom portion)
    road_top = int(h * 0.65)
    draw.rectangle([0, road_top, w, h], fill=(80, 80, 80))

    # Sidewalk
    draw.rectangle([0, road_top - 8, w, road_top], fill=(200, 190, 180))

    # Road markings (dashed line)
    for x in range(20, w - 20, 50):
        draw.rectangle([x, int(h * 0.82), x + 20, int(h * 0.82) + 4], fill=(255, 255, 200))

    # Cars on the road
    car_positions = [
        (80, int(h * 0.72), (200, 50, 50)),
        (200, int(h * 0.72), (50, 100, 200)),
        (350, int(h * 0.75), (50, 150, 50)),
        (50, int(h * 0.85), (150, 150, 50)),
        (300, int(h * 0.85), (50, 50, 150)),
    ]
    for cx, cy, color in car_positions:
        draw.rectangle([cx, cy, cx + 40, cy + 18], fill=color)
        draw.rectangle([cx + 8, cy - 5, cx + 32, cy], fill=(100, 100, 140))
        # wheels
        draw.ellipse([cx + 5, cy + 14, cx + 12, cy + 20], fill=(30, 30, 30))
        draw.ellipse([cx + 28, cy + 14, cx + 35, cy + 20], fill=(30, 30, 30))

    # Trees / vegetation between buildings
    for tx in [w // 3, w // 2, w * 2 // 3]:
        ty = int(h * 0.4) - 30
        draw.rectangle([tx - 5, ty + 20, tx + 5, int(h * 0.55)], fill=(100, 60, 30))  # trunk
        draw.ellipse([tx - 30, ty - 10, tx + 30, ty + 40], fill=(50, 130, 50))  # canopy

    # Traffic signs
    draw.rectangle([w // 2 - 3, int(h * 0.35), w // 2 + 3, int(h * 0.45)], fill=(80, 80, 80))
    draw.rectangle([w // 2 - 10, int(h * 0.35) - 15, w // 2 + 10, int(h * 0.35)], fill=(200, 50, 50))

    # Traffic light
    draw.rectangle([w // 2 - 80, int(h * 0.30), w // 2 - 75, int(h * 0.42)], fill=(60, 60, 60))
    draw.ellipse([w // 2 - 82, int(h * 0.31), w // 2 - 73, int(h * 0.34)], fill=(50, 50, 50))
    draw.ellipse([w // 2 - 82, int(h * 0.35), w // 2 - 73, int(h * 0.38)], fill=(255, 200, 50))
    draw.ellipse([w // 2 - 82, int(h * 0.39), w // 2 - 73, int(h * 0.42)], fill=(50, 50, 50))

    return img


def load_model():
    """Load pre-trained DeepLabV3-MobileNet."""
    print("[*] Loading DeepLabV3-MobileNet on {} ...".format(DEVICE))
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights
            .COCO_WITH_VOC_LABELS_V1
    )
    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")
    return model


def preprocess(image):
    """Resize, normalise and batch the input image."""
    trf = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    return trf(image).unsqueeze(0).to(DEVICE)


def decode_segmap(mask):
    """Convert class-index mask (H,W) to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(COLOUR_PALETTE)):
        rgb[mask == cls_id] = COLOUR_PALETTE[cls_id]
    return rgb


def run_inference(model, image):
    """Run model and return (segmentation_rgb, class_mask)."""
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    mask = output.argmax(0).cpu().numpy().astype(np.uint8)
    return decode_segmap(mask), mask


def compute_class_stats(mask):
    """Return list of (class_name, percentage) sorted by coverage."""
    total = mask.size
    stats = []
    for cls_id, name in enumerate(CLASS_NAMES):
        count = np.sum(mask == cls_id)
        pct = count / total * 100
        if pct > 0.5:
            stats.append((name, pct))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def save_comparison(orig, seg_rgb, stats, save_path):
    """Save a side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(orig.resize((512, 512)))
    axes[0].set_title("Input Urban Scene", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(seg_rgb)
    axes[1].set_title("AI Segmentation Result", fontsize=13)
    axes[1].axis("off")

    text_lines = ["Detected classes (>0.5% coverage):"]
    for name, pct in stats[:10]:
        text_lines.append("  {}: {:.1f}%".format(name, pct))
    if len(stats) > 10:
        text_lines.append("  ... and {} more".format(len(stats) - 10))
    fig.text(0.5, 0.02, "\n".join(text_lines),
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Comparison saved -> {}".format(save_path))


def save_overlay(orig, seg_rgb, save_path):
    """Save a blended overlay of the original and segmentation."""
    orig_resized = np.array(orig.resize((512, 512)))
    blended = (orig_resized * 0.5 + seg_rgb * 0.5).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(blended)
    ax.set_title("Overlay: Input * AI Segmentation", fontsize=13)
    ax.axis("off")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Overlay saved -> {}".format(save_path))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  AI Semantic Segmentation Demo")
    print("  DeepLabV3 + Urban Scene Analysis")
    print("=" * 55)

    # Use screenshot we already took
    screenshot_path = os.path.join(RESULTS_DIR, "input_screenshot.png")
    if not os.path.exists(screenshot_path):
        print("[*] Taking screenshot ...")
        from PIL import ImageGrab
        img = ImageGrab.grab().resize((512, 512))
        img.save(screenshot_path)
    else:
        print("[*] Loading screenshot ...")
    image = Image.open(screenshot_path).convert("RGB")
    print("    Image size: {}".format(image.size))

    model = load_model()

    print("[.] Running inference (pixel classification) ...")
    seg_rgb, mask = run_inference(model, image)

    stats = compute_class_stats(mask)
    print("\nClass breakdown (pixel %):")
    for name, pct in stats:
        print("  {:20s}  {:5.1f}%".format(name, pct))

    save_comparison(image, seg_rgb, stats,
                    os.path.join(RESULTS_DIR, "comparison.png"))
    save_overlay(image, seg_rgb,
                 os.path.join(RESULTS_DIR, "overlay.png"))

    Image.fromarray(seg_rgb).save(
        os.path.join(RESULTS_DIR, "segmentation_map.png"))
    print("[OK] Segmentation map -> {}/segmentation_map.png".format(RESULTS_DIR))

    print("\n" + "=" * 55)
    print("  Done! Results saved in 'results/' folder.")
    print("=" * 55)


if __name__ == "__main__":
    main()
