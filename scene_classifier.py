"""
AI-Driven Scene Classification Demo
Uses ResNet50 (pre-trained on ImageNet) to classify scenes/images.

Relevance to Remote Sensing:
- Scene classification is a fundamental task in remote sensing
- Used for land cover / land use classification
- Can identify urban, vegetation, water, agricultural areas
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models

from imagenet_classes import CLASS_MAP


RESULTS_DIR = "results"
GEE_INPUT_DIR = "gee_inputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BROAD_CATEGORIES = {
    "Urban": ["palace", "castle", "skyscraper", "church", "temple", "mosque", "stadium",
              "apartment", "house", "building", "shop", "store", "restaurant", "library",
              "museum", "theater", "arena", "highway", "street", "road", "bridge", "dam",
              "traffic", "parking", "lamp", "crosswalk", "fountain", "monument", "tower",
              "fort", "wall", "ruin", "amphitheater", "facade"],
    "Vegetation": ["tree", "forest", "jungle", "palm", "pine", "oak", "maple", "evergreen",
                   "shrub", "fern", "moss", "bamboo", "cactus", "grass", "lawn", "meadow",
                   "park", "leaf", "flower", "plant", "garden", "vine", "crop", "field",
                   "orchard", "vineyard", "rainforest", "woodland", "thicket"],
    "Water": ["lake", "ocean", "sea", "river", "stream", "pond", "waterfall", "swamp",
              "marsh", "reef", "coast", "shore", "beach", "harbor", "dock", "boat",
              "ship", "submarine", "water", "wave", "surf", "tide", "estuary", "bay",
              "gulf", "port", "marina", "pier", "wharf", "lighthouse", "buoy"],
    "Agriculture": ["farm", "barn", "haystack", "tractor", "harvester", "corn", "wheat",
                    "rice", "orchard", "vineyard", "greenhouse", "garden", "cattle",
                    "sheep", "goat", "pig", "chicken", "agriculture", "rural",
                    "pasture", "meadow", "field", "crop", "silo", "plow"],
    "Barren / Mountain": ["desert", "sand", "dune", "canyon", "cliff", "mountain",
                          "volcano", "rock", "stone", "boulder", "cave", "butte",
                          "mesa", "plateau", "badlands", "hill", "summit", "ridge",
                          "valley", "gorge", "ravine", "arid", "dry"],
    "Snow / Ice": ["snow", "ice", "glacier", "iceberg", "ski", "igloo", "polar",
                   "arctic", "frozen", "frost", "snowmobile", "snowboard", "winter",
                   "alpine", "avalanche", "ice floe"],
    "Sky / Atmosphere": ["sky", "cloud", "rainbow", "sunset", "sunrise", "storm",
                         "lightning", "tornado", "hurricane", "fog", "mist", "haze",
                         "smog", "airplane", "helicopter", "jet", "bird", "kite",
                         "balloon", "parachute", "satellite", "space", "moon", "star"],
}


def get_image():
    """Load a remote sensing image from GEE outputs for analysis."""
    gee_dir = GEE_INPUT_DIR
    if os.path.isdir(gee_dir):
        images = sorted([f for f in os.listdir(gee_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    else:
        images = []

    if not images:
        print("[!] No GEE images found. Falling back to screenshot ...")
        from PIL import ImageGrab
        img = ImageGrab.grab().resize((512, 512))
        img.save(os.path.join(RESULTS_DIR, "fallback_input.png"))
        return img, "fallback_screenshot"

    # Use the NDVI_final or first available image as demo input
    preferred = [f for f in images if "final" in f.lower()]
    chosen = preferred[0] if preferred else images[0]
    img_path = os.path.join(gee_dir, chosen)

    print("[*] Loading remote sensing image: {}".format(chosen))
    img = Image.open(img_path).convert("RGB")
    return img, chosen


def load_model():
    """Load pre-trained ResNet50."""
    print("[*] Loading ResNet50 (ImageNet pretrained) on {} ...".format(DEVICE))
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")
    return model


def classify(model, image):
    """Run classification and return top predictions."""
    trf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = trf(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    top10 = torch.topk(probs, 10)

    results = []
    for i in range(10):
        idx = top10.indices[i].item()
        prob = top10.values[i].item() * 100
        name = CLASS_MAP.get(idx, "unknown class {}".format(idx))
        results.append({"id": idx, "name": name, "confidence": prob})

    return results


def categorize(predictions):
    """Map predictions to broad remote sensing categories."""
    category_scores = {}
    for cat_name, keywords in BROAD_CATEGORIES.items():
        score = 0
        for pred in predictions:
            for kw in keywords:
                if kw.lower() in pred["name"].lower():
                    score += pred["confidence"]
        if score > 0:
            category_scores[cat_name] = round(score, 1)

    return sorted(category_scores.items(), key=lambda x: x[1], reverse=True)


def save_results(image, predictions, categories, img_name, save_path):
    """Save visual results with classification info."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: input image
    axes[0].imshow(image.resize((512, 512)))
    axes[0].set_title("Remote Sensing Input\n{}".format(img_name), fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Right: predictions bar chart
    names = [p["name"][:25] for p in predictions]
    confs = [p["confidence"] for p in predictions]

    bars = axes[1].barh(range(len(names)), confs, color="steelblue", height=0.6)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_xlabel("Confidence (%)", fontsize=11)
    axes[1].set_title("Top-10 Predictions (ResNet50)", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()

    # Add confidence labels on bars
    for i, (bar, conf) in enumerate(zip(bars, confs)):
        axes[1].text(conf + 0.5, bar.get_y() + bar.get_height()/2,
                     "{:.1f}%".format(conf), va="center", fontsize=8)

    # Category summary at the bottom
    cat_text = "   |   ".join("{}: {}%".format(cat, score) for cat, score in categories)
    fig.text(0.5, 0.02, "Scene Categories: " + cat_text,
             ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Result chart saved -> {}".format(save_path))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  AI + Remote Sensing Scene Classification Demo")
    print("  Model: ResNet50 | Task: Land Cover / Scene Understanding")
    print("=" * 60)

    # 1. Get input
    image, img_name = get_image()

    # 2. Load model
    model = load_model()

    # 3. Classify
    print("[*] Classifying remote sensing scene ...")
    predictions = classify(model, image)

    # 4. Categorize
    categories = categorize(predictions)

    # 5. Show results
    print("\n--- Top 5 Predictions ---")
    for p in predictions[:5]:
        print("  {:.1f}%  {}".format(p["confidence"], p["name"]))

    print("\n--- Scene Categories ---")
    for cat, score in categories:
        print("  {}: {}%".format(cat, score))

    # 6. Save visualization
    save_results(image, predictions, categories, img_name,
                 os.path.join(RESULTS_DIR, "classification_result.png"))

    print("\n" + "=" * 60)
    print("  Done! Results saved in 'results/' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
