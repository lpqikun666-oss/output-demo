"""
Remote Sensing Image Analysis Demo
Two complementary tracks:

Track 1 - GEE Composite Analysis:
  Load real remote sensing data (NDVI, LST, NDBSI composites)
  Compute statistics, histogram, K-means clustering for land cover
  These are standard remote sensing workflows

Track 2 - Deep Learning Semantic Segmentation:
  Uses DeepLabV3 (pre-trained on COCO) for scene segmentation
  Runs on generated urban landscape + screenshot
  Same architecture used in satellite image segmentation
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageGrab
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as T
from torchvision import models


RESULTS_DIR = "results"
GEE_DIR = "gee_inputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)


# ---- Track 1: GEE Remote Sensing Analysis --------------------------------

def analyze_gee():
    """Analyze real GEE remote sensing composites with K-means clustering."""
    gee_files = sorted([f for f in os.listdir(GEE_DIR)
                        if f.lower().endswith((".png", ".jpg"))])
    if not gee_files:
        print("[!] No GEE images found.")
        return None

    preferred = sorted([f for f in gee_files if "final" in f.lower()])
    chosen = preferred[0] if preferred else gee_files[0]
    gee_path = os.path.join(GEE_DIR, chosen)
    img = Image.open(gee_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    print("\n  Input: {}".format(chosen))
    print("  Size: {} x {} pixels".format(w, h))

    # Downsample for faster processing (keep aspect ratio)
    scale = min(512 / w, 512 / h, 1.0)
    small = np.array(img.resize((int(w * scale), int(h * scale))))
    pixels = small.reshape(-1, 3).astype(np.float32)

    # K-means clustering (unsupervised land cover classification)
    n_clusters = 5
    print("  Running K-means clustering (k={}) ...".format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=5)
    labels = kmeans.fit_predict(pixels)
    label_map = labels.reshape(small.shape[0], small.shape[1])

    # Map cluster centers to colors
    centers = kmeans.cluster_centers_.astype(np.uint8)
    cluster_colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 128, 128], [255, 128, 0], [128, 0, 255],
    ])
    seg_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for i in range(n_clusters):
        seg_rgb[label_map == i] = cluster_colors[i % len(cluster_colors)]

    # Statistics per cluster
    print("\n  Land cover clusters (unsupervised):")
    cluster_stats = []
    for i in range(n_clusters):
        count = np.sum(label_map == i)
        pct = count / label_map.size * 100
        color = centers[i]
        cluster_stats.append((i + 1, pct, color))
        print("    Cluster {}: {:5.1f}%  (center RGB: {})".format(
            i + 1, pct, color))

    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(small)
    axes[0].set_title("GEE: {}".format(chosen), fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(seg_rgb)
    axes[1].set_title("K-means Clusters (k={})".format(n_clusters), fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Combined
    overlay = (small * 0.6 + seg_rgb * 0.4).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay: GEE + Clusters", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "gee_kmeans_result.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] GEE analysis -> {}".format(out))

    return chosen


# ---- Track 2: Deep Learning Segmentation ---------------------------------

def make_urban_scene(size=(512, 512)):
    """Generate a synthetic urban scene for segmentation demo."""
    w, h = size
    img = Image.new("RGB", (w, h), (135, 180, 235))
    draw = ImageDraw.Draw(img)

    for y in range(int(h * 0.4)):
        b = int(235 - (y / (h * 0.4)) * 60)
        draw.line([(0, y), (w, y)], fill=(b - 100, b - 80, b))

    for bx in [10, 100, 200, 300, 400]:
        bh = np.random.randint(80, 140)
        by = int(h * 0.4) - bh + np.random.randint(-10, 30)
        c = np.random.randint(130, 200)
        draw.rectangle([bx, by, bx + 80, int(h * 0.7)], fill=(c, c, c + 10))
        for wy in range(by + 10, int(h * 0.7) - 10, 20):
            for wx in range(bx + 10, bx + 70, 20):
                draw.rectangle([wx, wy, wx + 10, wy + 10], fill=(220, 220, 120))

    rt = int(h * 0.65)
    draw.rectangle([0, rt, w, h], fill=(80, 80, 80))
    draw.rectangle([0, rt - 8, w, rt], fill=(200, 190, 180))
    for x in range(20, w - 20, 50):
        draw.rectangle([x, int(h * 0.82), x + 20, int(h * 0.82) + 4], fill=(255, 255, 200))

    for cx, cy, col in [(80, int(h * 0.72), (200, 50, 50)),
                          (220, int(h * 0.72), (50, 100, 200)),
                          (350, int(h * 0.75), (50, 150, 50)),
                          (60, int(h * 0.85), (150, 150, 50))]:
        draw.rectangle([cx, cy, cx + 40, cy + 18], fill=col)
        draw.rectangle([cx + 8, cy - 5, cx + 32, cy], fill=(100, 100, 140))
        draw.ellipse([cx + 5, cy + 14, cx + 12, cy + 20], fill=(30, 30, 30))
        draw.ellipse([cx + 28, cy + 14, cx + 35, cy + 20], fill=(30, 30, 30))

    for tx in [150, 280, 450]:
        ty = int(h * 0.4) - 30
        draw.rectangle([tx - 5, ty + 20, tx + 5, int(h * 0.55)], fill=(100, 60, 30))
        draw.ellipse([tx - 30, ty - 10, tx + 30, ty + 40], fill=(50, 130, 50))

    return img


CLASS_NAMES = ["background", "road", "sidewalk", "building", "wall", "fence",
               "pole", "traffic light", "traffic sign", "vegetation", "terrain",
               "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]

COLOUR_PALETTE = np.array([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)


def load_model():
    print("[*] Loading DeepLabV3 on {} ...".format(DEVICE))
    model = models.segmentation.deeplabv3_resnet50(
        weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")
    return model


def preprocess(image):
    trf = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return trf(image).unsqueeze(0).to(DEVICE)


def run_segmentation(model, image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    mask = output.argmax(0).cpu().numpy().astype(np.uint8)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(COLOUR_PALETTE)):
        rgb[mask == cls_id] = COLOUR_PALETTE[cls_id]
    return rgb, mask


def seg_stats(mask):
    total = mask.size
    stats = []
    for cls_id, name in enumerate(CLASS_NAMES):
        count = np.sum(mask == cls_id)
        pct = count / total * 100
        if pct > 0.5:
            stats.append((name, pct))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def seg_result(orig, seg_rgb, stats, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig.resize((512, 512)))
    axes[0].set_title("Input: {}".format(title), fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(seg_rgb)
    axes[1].set_title("Segmentation Map", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    text = "\n".join("{:20s} {:5.1f}%".format(n, p) for n, p in stats[:10])
    axes[2].text(0.1, 0.5, text, fontsize=11, va="center",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    axes[2].set_title("Pixel Class Breakdown", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] {} -> {}".format(title, save_path))


# ---- Main -----------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Remote Sensing AI Analysis Demo")
    print("  GEE Data + K-Means + Deep Learning Segmentation")
    print("=" * 60)

    # Track 1: GEE Analysis
    print("\n--- Track 1: GEE Remote Sensing Analysis ---")
    gee_name = analyze_gee()

    # Track 2: Deep Learning Segmentation
    print("\n--- Track 2: Deep Learning Segmentation ---")
    model = load_model()

    # 2a: Synthetic urban scene
    print("\n[2a] Urban scene generation + segmentation ...")
    urban = make_urban_scene()
    urban.save(os.path.join(RESULTS_DIR, "urban_input.png"))
    seg, mask = run_segmentation(model, urban)
    stats = seg_stats(mask)
    for n, p in stats:
        print("    {:20s} {:5.1f}%".format(n, p))
    seg_result(urban, seg, stats, "urban_scene",
               os.path.join(RESULTS_DIR, "urban_segmentation.png"))

    # 2b: Screenshot segmentation
    print("\n[2b] Screenshot segmentation ...")
    try:
        screenshot = ImageGrab.grab().resize((512, 512))
    except Exception:
        screenshot = make_urban_scene()
    screenshot.save(os.path.join(RESULTS_DIR, "screenshot_input.png"))
    seg2, mask2 = run_segmentation(model, screenshot)
    stats2 = seg_stats(mask2)
    for n, p in stats2:
        print("    {:20s} {:5.1f}%".format(n, p))
    seg_result(screenshot, seg2, stats2, "screenshot",
               os.path.join(RESULTS_DIR, "screenshot_segmentation.png"))

    print("\n" + "=" * 60)
    print("  Done! Output files:")
    print("  1. results/gee_kmeans_result.png")
    if gee_name:
        print("     (GEE composite: {})".format(gee_name))
    print("  2. results/urban_segmentation.png")
    print("  3. results/screenshot_segmentation.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
