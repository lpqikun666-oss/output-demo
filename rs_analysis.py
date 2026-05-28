"""
Remote Sensing Image Analysis Demo
Self-contained — no external data needed.

Track 1 - Synthetic RS Image + K-Means Land Cover Classification:
  Generates a realistic satellite-view image (water, forest, agriculture,
  urban, bare soil), then applies unsupervised K-means clustering to
  separate land cover types. A standard remote sensing workflow.

Track 2 - DeepLabV3 Semantic Segmentation:
  Pre-trained on COCO (real-world scenes). Demonstrates the deep learning
  pipeline for pixel-wise classification. Note: for optimal RS results,
  fine-tune on remote sensing datasets like DeepGlobe or ISPRS.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as T
from torchvision import models


RESULTS_DIR = "results"
GEE_DIR = "gee_inputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

# Color table for K-means cluster visualization
CLUSTER_COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [128, 128, 128], [255, 128, 0], [128, 0, 255],
], dtype=np.uint8)


# ========== Track 1: Synthetic RS Image + K-Means ============================


def generate_synthetic_rs_image(size=(512, 512)):
    """Generate a realistic synthetic satellite image with varied land cover.

    Simulates a true-color satellite view containing:
      - Lake / river network  (deep blue)
      - Dense forest          (dark green)
      - Agricultural fields   (patterned green / tan)
      - Urban area            (gray, rectilinear)
      - Bare soil / fallow    (brown)
    """
    w, h = size
    base = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(base)

    # --- Background: bare soil / sparse vegetation ---
    draw.rectangle([0, 0, w, h], fill=(168, 145, 105))

    # --- Lake (top-left) ---
    draw.ellipse([25, 20, 195, 185], fill=(50, 86, 146))
    # Depth gradient
    for r in range(5):
        s = 5 - r
        c = (42 + s*2, 78 + s*2, 138 + s*2)
        draw.ellipse([45 + r*5, 40 + r*5, 175 - r*5, 165 - r*5], fill=c)

    # --- River from lake flowing southeast ---
    river = [(180, 155), (215, 178), (228, 200), (232, 222),
             (225, 245), (208, 265), (185, 282)]
    draw.line(river, fill=(48, 84, 144), width=18)

    # --- Forest (top-right) ---
    draw.rectangle([250, 15, 505, 185], fill=(40, 100, 40))
    for _ in range(50):
        x = np.random.randint(255, 500)
        y = np.random.randint(20, 180)
        r = np.random.randint(8, 20)
        s = np.random.randint(35, 115)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(s, s+8, s-12))

    # --- Forest (bottom-right) ---
    draw.rectangle([360, 350, 505, 495], fill=(36, 96, 36))
    for _ in range(25):
        x = np.random.randint(365, 500)
        y = np.random.randint(355, 490)
        r = np.random.randint(8, 16)
        s = np.random.randint(32, 100)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(s, s+8, s-12))

    # --- Agricultural fields (bottom-left, 4x3 grid) ---
    palettes = [(140, 166, 80), (160, 140, 70), (118, 153, 88), (168, 128, 52)]
    for row in range(4):
        for col in range(3):
            x = 15 + col * 83
            y = 255 + row * 55
            c = palettes[(row + col) % len(palettes)]
            v = np.random.randint(-8, 8)
            draw.rectangle([x, y, x+73, y+45], fill=(c[0]+v, c[1]+v, c[2]+v))

    # --- Urban area (center-right) ---
    for bx in [215, 262, 310, 358]:
        for by in [255, 300, 345]:
            c = np.random.randint(145, 210)
            draw.rectangle([bx, by, bx+40, by+35], fill=(c, c, c))
            roof = np.random.randint(60, 130)
            draw.rectangle([bx+3, by+3, bx+37, by+10], fill=(roof, roof-5, roof-8))

    # --- Road network ---
    draw.rectangle([235, 295, 258, 405], fill=(178, 173, 168))
    draw.rectangle([195, 335, 360, 358], fill=(182, 177, 172))

    # --- Small vegetation patches ---
    for _ in range(20):
        x = np.random.randint(10, w-10)
        y = np.random.randint(195, 300)
        px = base.getpixel((x, y))
        if px != (50, 86, 146) and px != (48, 84, 144):
            s = np.random.randint(4, 12)
            shade = np.random.randint(60, 130)
            draw.ellipse([x-s, y-s, x+s, y+s], fill=(shade, shade+20, shade-20))

    # --- Smooth transitions ---
    img = base.filter(ImageFilter.GaussianBlur(radius=2))

    # --- Sensor noise ---
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(SEED).randn(h, w, 3) * 5
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def analyze_kmeans(image, n_clusters=5, title="", save_path=None):
    """Run K-means clustering on an image for land-cover classification.

    Returns list of (cluster_id, percentage, center_rgb) sorted by size.
    """
    w, h = image.size
    arr = np.array(image)

    # Downsample for speed
    scale = min(512 / w, 512 / h, 1.0)
    small = np.array(image.resize((int(w * scale), int(h * scale))))
    pixels = small.reshape(-1, 3).astype(np.float32)

    print("  Running K-means clustering (k={}, {} pixels) ...".format(
        n_clusters, len(pixels)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=5)
    labels = kmeans.fit_predict(pixels)
    label_map = labels.reshape(small.shape[0], small.shape[1])

    # Build colorized segmentation map
    seg_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for i in range(n_clusters):
        seg_rgb[label_map == i] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

    # Per-cluster statistics
    centers = kmeans.cluster_centers_.astype(np.uint8)
    stats = []
    for i in range(n_clusters):
        count = np.sum(label_map == i)
        pct = count / label_map.size * 100
        stats.append((i + 1, pct, centers[i]))
        print("    Cluster {}: {:5.1f}%  center RGB ({:3d},{:3d},{:3d})".format(
            i + 1, pct, centers[i][0], centers[i][1], centers[i][2]))
    stats.sort(key=lambda x: x[1], reverse=True)

    # Overlay
    overlay = (small * 0.55 + seg_rgb * 0.45).astype(np.uint8)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(small)
    axes[0].set_title("Input: {}".format(title), fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(seg_rgb)
    axes[1].set_title("K-means Land Cover (k={})".format(n_clusters),
                      fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay: Image + Clusters", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight") if save_path else None
    plt.close(fig)
    print("[OK] K-means result -> {}".format(save_path))

    return stats


# ========== Track 2: DeepLabV3 Semantic Segmentation ========================

CLASS_NAMES = [
    "background", "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

COLOUR_PALETTE = np.array([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)


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


def load_model():
    """Load DeepLabV3-ResNet50 pre-trained on COCO."""
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
    """Run DeepLabV3 and return (color_mask, class_mask)."""
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
    """Return list of (class_name, percentage) for classes > 0.5% coverage."""
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
    """Save three-panel segmentation figure."""
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


# ========== Main =============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Remote Sensing AI Analysis Demo")
    print("  [Track 1] Synthetic RS Image + K-Means Land Cover")
    print("  [Track 2] DeepLabV3 Semantic Segmentation Pipeline")
    print("=" * 60)

    # ---- Track 1: K-Means Land Cover Classification ----
    print("\n--- Track 1: Land Cover Classification (K-Means) ---")

    rs_image = generate_synthetic_rs_image()
    rs_image.save(os.path.join(RESULTS_DIR, "synthetic_rs_input.png"))
    print("[OK] Synthetic RS image generated (512x512)")

    stats = analyze_kmeans(rs_image, n_clusters=6,
                           title="Synthetic RS Image",
                           save_path=os.path.join(RESULTS_DIR,
                                                  "kmeans_landcover.png"))
    print("\n  Land cover summary (by area):")
    for cid, pct, center in stats:
        label = "Cluster {}".format(cid)
        print("    {:12s}  {:5.1f}%".format(label, pct))

    # Optional: also analyze GEE data if available
    if os.path.isdir(GEE_DIR):
        gee_files = sorted([f for f in os.listdir(GEE_DIR)
                            if f.lower().endswith((".png", ".jpg"))])
        gee_preferred = sorted([f for f in gee_files if "final" in f.lower()])
        if gee_preferred or gee_files:
            chosen = gee_preferred[0] if gee_preferred else gee_files[0]
            print("\n[*] Also found GEE data: {} (bonus analysis)".format(chosen))
            gee_img = Image.open(os.path.join(GEE_DIR, chosen)).convert("RGB")
            analyze_kmeans(gee_img, n_clusters=5,
                           title="GEE: " + chosen,
                           save_path=os.path.join(RESULTS_DIR,
                                                  "gee_kmeans_result.png"))

    # ---- Track 2: DeepLabV3 Segmentation ----
    print("\n--- Track 2: Deep Learning Segmentation ---")
    print("  Note: DeepLabV3 is pre-trained on COCO (real photos).")
    print("  For production RS use, fine-tune on remote sensing datasets.\n")

    model = load_model()

    print("\n[2a] Synthetic urban scene segmentation ...")
    urban = make_urban_scene()
    urban.save(os.path.join(RESULTS_DIR, "urban_input.png"))
    seg, mask = run_segmentation(model, urban)
    seg_stats_list = seg_stats(mask)
    for n, p in seg_stats_list:
        print("    {:20s} {:5.1f}%".format(n, p))
    seg_result(urban, seg, seg_stats_list, "urban_scene",
               os.path.join(RESULTS_DIR, "urban_segmentation.png"))

    print("\n[2b] Synthetic RS image segmentation ...")
    seg_rs, mask_rs = run_segmentation(model, rs_image)
    seg_stats_rs = seg_stats(mask_rs)
    for n, p in seg_stats_rs:
        print("    {:20s} {:5.1f}%".format(n, p))
    seg_result(rs_image, seg_rs, seg_stats_rs, "rs_image",
               os.path.join(RESULTS_DIR, "rs_segmentation.png"))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Done! Output files in 'results/':")
    print("  1. synthetic_rs_input.png   (generated satellite-like image)")
    print("  2. kmeans_landcover.png     (K-means land cover classification)")
    print("  3. urban_segmentation.png   (DeepLabV3 on urban scene)")
    print("  4. rs_segmentation.png      (DeepLabV3 on RS image)")
    print("=" * 60)


if __name__ == "__main__":
    main()
