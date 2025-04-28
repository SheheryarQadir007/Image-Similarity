"""
Near-duplicate detection with PDQ + YOLOv8 embeddings for two images
==================================================================

• Compares two images using PDQ, YOLO embeddings, SSIM, MSE, semantic overlap, edge density, and color histogram.
• Categorizes the pair as exact_similar, somewhat_similar, or completely_different.
• Visualizes the result and saves it to an output directory.
• Supports both local paths and image URLs.
"""

# ─────────────────── imports ───────────────────
import os
import logging
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import pdqhash
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import uuid
import shutil

# ────────────────── logging ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ───────────── user-defined images ─────────────
# Define your two images here (local paths or URLs)
IMAGE1 = r"https://plus.unsplash.com/premium_photo-1674815329488-c4fc6bf4ced8?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with your first image path or URL
IMAGE2 = r"https://plus.unsplash.com/premium_photo-1674815329488-c4fc6bf4ced8?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Replace with your second image path or URL
OUTPUT_DIR = "image_comparison"  # Directory to save the comparison visualization

# ───────────── configuration ─────────────
EXACT_SIMILAR = dict(pdq_max=30, cosine_min=0.93, ssim_min=0.80)
SOMEWHAT_SIMILAR = dict(pdq_max=150, cosine_min=0.80, ssim_min=0.20)
PDQ_THRESHOLD = 200
COSINE_THRESH = 0.80
SEMANTIC_OVERLAP_THRESHOLD = 0.2
EDGE_DENSITY_DIFF_THRESHOLD = 0.05
COLOR_HIST_DIFF_THRESHOLD = 0.5

WEIGHTS = dict(pdq=0.35, cosine=0.35, ssim=0.25, mse=0.05)
SCORE_THRESHOLDS = dict(exact_similar=0.85, somewhat_similar=0.51)
MAX_MSE = 10000

RESIZE_DIM = (256, 256)
YOLO_DIM = (640, 640)
SSIM_DIM = (224, 224)

SEMANTIC_CATEGORIES = {
    "residential": ["sofa", "couch", "bed", "tvmonitor", "coffee table", "lamp", "chandelier", "table", "oven", "refrigerator"],
    "commercial": ["desk", "chair", "monitor", "keyboard", "mouse", "noticeboard"]
}
COCO_TO_SEMANTIC = {
    "chair": "commercial",
    "sofa": "residential",
    "bed": "residential",
    "tvmonitor": "residential",
    "desk": "commercial",
    "monitor": "commercial",
    "keyboard": "commercial",
    "mouse": "commercial",
    "table": "residential",
    "oven": "residential",
    "refrigerator": "residential"
}

# ───────────── YOLO initialisation ─────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    yolo_model = YOLO("yolov8n.pt").to(device)
    SPPF = yolo_model.model.model[9]
except Exception as e:
    logger.error(f"Failed to initialize YOLO model: {str(e)}")
    raise RuntimeError("YOLO model initialization failed")

# ───────────── helper functions ─────────────
def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

def load_image_from_url(url: str) -> np.ndarray:
    try:
        logger.info(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {str(e)}")
        raise

def resize_image(image: np.ndarray, size: tuple, output_path: str = None) -> np.ndarray:
    try:
        logger.info(f"Resizing image to {size}")
        resized_img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if output_path:
            cv2.imwrite(output_path, resized_img)
            logger.info(f"Resized image saved to {output_path}")
        return resized_img
    except Exception as e:
        logger.error(f"Image resizing failed: {str(e)}")
        raise

def preprocess_image(image_path_or_url: str, temp_dir: str = "temp_resized") -> tuple:
    os.makedirs(temp_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    
    if is_url(image_path_or_url):
        img = load_image_from_url(image_path_or_url)
        temp_filename = f"resized_{unique_id}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        resize_image(img, RESIZE_DIM, output_path=temp_path)
        return temp_path, image_path_or_url
    else:
        if not os.path.exists(image_path_or_url):
            logger.error(f"Image path does not exist: {image_path_or_url}")
            raise FileNotFoundError(f"Image path does not exist: {image_path_or_url}")
        temp_filename = f"resized_{os.path.basename(image_path_or_url)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        img = cv2.imread(image_path_or_url)
        if img is None:
            logger.error(f"Failed to load image for resizing: {image_path_or_url}")
            raise ValueError("cv2.imread returned None")
        resize_image(img, RESIZE_DIM, output_path=temp_path)
        return temp_path, image_path_or_url

def compute_pdq(path: str) -> int:
    try:
        logger.info(f"Loading image for PDQ: {path}")
        img = cv2.imread(path)
        if img is None:
            logger.error(f"cv2.imread failed to load image: {path}")
            raise ValueError("cv2.imread returned None")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vec, _ = pdqhash.compute(img)
        return int("".join(map(str, vec)), 2)
    except Exception as e:
        logger.error(f"PDQ failed for {path}: {str(e)}")
        return -1

pdq_distance = lambda a, b: (a ^ b).bit_count()

def compute_yolo_embedding(paths: list) -> np.ndarray:
    try:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        batch = torch.stack(
            [torch.from_numpy(np.array(im.resize(YOLO_DIM))).permute(2, 0, 1).float().div_(255.0) for im in imgs]
        ).to(device)

        feats = []
        def hook(_, __, out): feats.append(out)
        h = SPPF.register_forward_hook(hook)
        with torch.no_grad():
            yolo_model.predict(batch, verbose=False)
        h.remove()

        if not feats:
            logger.error("Hook captured no features")
            raise RuntimeError("Hook captured no features")

        fmap = feats[0][0] if isinstance(feats[0], list) else feats[0]
        emb = torch.mean(fmap, dim=(2, 3)).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb /= np.where(norms == 0, 1, norms)

        for im in imgs:
            im.close()
        del batch
        if device == "cuda":
            torch.cuda.empty_cache()
        return emb
    except Exception as e:
        logger.error(f"YOLO embedding failed: {str(e)}")
        raise

def detect_objects(paths: list) -> list:
    try:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        batch = torch.stack(
            [torch.from_numpy(np.array(im.resize(YOLO_DIM))).permute(2, 0, 1).float().div_(255.0) for im in imgs]
        ).to(device)

        with torch.no_grad():
            results = yolo_model.predict(batch, verbose=False)

        object_categories = []
        for i, result in enumerate(results):
            detected_objects = []
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = yolo_model.names[class_id]
                    if class_name in COCO_TO_SEMANTIC:
                        detected_objects.append(COCO_TO_SEMANTIC[class_name])
            object_categories.append(set(detected_objects))
            logger.info(f"Detected objects in {paths[i]}: {detected_objects}")

        for im in imgs:
            im.close()
        del batch
        if device == "cuda":
            torch.cuda.empty_cache()
        return object_categories
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        raise

def compute_semantic_overlap(categories1: set, categories2: set) -> float:
    if not categories1 and not categories2:
        return 1.0
    if not categories1 or not categories2:
        return 0.0
    intersection = len(categories1.intersection(categories2))
    union = len(categories1.union(categories2))
    overlap = intersection / union if union > 0 else 0.0
    logger.info(f"Semantic categories: Image 1={categories1}, Image 2={categories2}, Overlap={overlap:.3f}")
    return overlap

def compute_edge_density(path: str) -> float:
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image for edge detection: {path}")
            raise ValueError("cv2.imread returned None")
        img = cv2.resize(img, SSIM_DIM)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(sobel, 50, 150)
        edge_density = np.sum(edges == 255) / (SSIM_DIM[0] * SSIM_DIM[1])
        return edge_density
    except Exception as e:
        logger.error(f"Edge detection failed for {path}: {str(e)}")
        raise

def compute_edge_density_difference(path1: str, path2: str) -> float:
    density1 = compute_edge_density(path1)
    density2 = compute_edge_density(path2)
    diff = abs(density1 - density2)
    logger.info(f"Edge density difference: Image 1={density1:.3f}, Image 2={density2:.3f}, Diff={diff:.3f}")
    return diff

def compute_color_histogram(path: str) -> np.ndarray:
    try:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Failed to load image for histogram: {path}")
            raise ValueError("cv2.imread returned None")
        img = cv2.resize(img, SSIM_DIM)
        hist = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        hist = np.concatenate(hist).flatten()
        hist = hist / hist.sum()
        return hist
    except Exception as e:
        logger.error(f"Color histogram computation failed for {path}: {str(e)}")
        raise

def compute_color_histogram_difference(path1: str, path2: str) -> float:
    hist1 = compute_color_histogram(path1)
    hist2 = compute_color_histogram(path2)
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    diff = min(diff / 200, 1.0)
    logger.info(f"Color histogram difference: {diff:.3f}")
    return diff

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def ssim_score(p1: str, p2: str) -> float:
    try:
        g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            logger.error(f"Failed to load images for SSIM: {p1}, {p2}")
            raise ValueError("cv2.imread returned None")
        g1 = cv2.resize(g1, SSIM_DIM)
        g2 = cv2.resize(g2, SSIM_DIM)
        s, _ = ssim(g1, g2, full=True)
        return float(s)
    except Exception as e:
        logger.error(f"SSIM failed {p1},{p2}: {str(e)}")
        raise

def mse_score(p1: str, p2: str) -> float:
    try:
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            logger.error(f"Failed to load images for MSE: {p1}, {p2}")
            raise ValueError("cv2.imread returned None")
        img1 = cv2.resize(img1, SSIM_DIM)
        img2 = cv2.resize(img2, SSIM_DIM)
        img1 = cv2.GaussianBlur(img1, (7, 7), 0)
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        mse = np.mean((img1 - img2) ** 2)
        mse_similarity = 1 - (mse / MAX_MSE)
        mse_similarity = max(0, min(1, mse_similarity))
        return mse_similarity
    except Exception as e:
        logger.error(f"MSE failed {p1},{p2}: {str(e)}")
        raise

def compute_weighted_score(dpdq: int, cs: float, ss: float, mse_sim: float) -> float:
    pdq_sim = 1 - (dpdq / 256)
    score = (
        WEIGHTS["pdq"] * pdq_sim +
        WEIGHTS["cosine"] * cs +
        WEIGHTS["ssim"] * ss +
        WEIGHTS["mse"] * mse_sim
    )
    return score

def categorize(dpdq: int, cs: float, ss: float, mse_sim: float, semantic_overlap: float, edge_density_diff: float, color_hist_diff: float) -> str:
    exact_conditions = (
        dpdq <= EXACT_SIMILAR["pdq_max"]
        and cs >= EXACT_SIMILAR["cosine_min"]
        and ss >= EXACT_SIMILAR["ssim_min"]
    )
    if exact_conditions:
        return "exact_similar"
    
    if semantic_overlap == 0.0:
        return "completely_different"
    
    criteria_met = 0
    if semantic_overlap < SEMANTIC_OVERLAP_THRESHOLD:
        criteria_met += 1
    if edge_density_diff > EDGE_DENSITY_DIFF_THRESHOLD:
        criteria_met += 1
    if color_hist_diff > COLOR_HIST_DIFF_THRESHOLD:
        criteria_met += 1
    
    if criteria_met >= 2:
        return "completely_different"
    
    score = compute_weighted_score(dpdq, cs, ss, mse_sim)
    if score >= SCORE_THRESHOLDS["exact_similar"]:
        return "exact_similar"
    elif score >= SCORE_THRESHOLDS["somewhat_similar"]:
        return "somewhat_similar"
    else:
        return "completely_different"

def plot_image_pair(p1: str, p2: str, dpdq: int, cs: float, ss: float, mse_sim: float, semantic_overlap: float, edge_density_diff: float, color_hist_diff: float, category: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.ravel()

    for col, path, lbl in [(0, p1, "Image A"), (1, p2, "Image B")]:
        ax = axes[col]
        try:
            if is_url(path):
                img = Image.open(BytesIO(requests.get(path).content))
            else:
                img = Image.open(path)
            ax.imshow(img)
            ax.set_title(lbl)
            ax.axis("off")
            img.close()
        except Exception as e:
            ax.set_title(f"Err: {e}")
            ax.axis("off")

    fig.suptitle(
        f"{category.replace('_',' ').title()}\n"
        f"PDQ Distance={dpdq} | Cosine={cs:.3f} | SSIM={ss:.3f} | MSE Similarity={mse_sim:.3f}\n"
        f"Semantic Overlap={semantic_overlap:.3f} | Edge Density Diff={edge_density_diff:.3f} | Color Hist Diff={color_hist_diff:.3f}",
        fontsize=12,
    )

    plt.tight_layout()
    output_path = os.path.join(out_dir, f"comparison_{uuid.uuid4()}.png")
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved comparison visualization to {output_path}")

def compare_two_images(image1_path_or_url: str, image2_path_or_url: str, output_dir: str) -> dict:
    logger.info(f"Comparing {image1_path_or_url} vs {image2_path_or_url}")

    temp_image1_path, original_image1 = preprocess_image(image1_path_or_url)
    temp_image2_path, original_image2 = preprocess_image(image2_path_or_url)
    temp_paths = [temp_image1_path, temp_image2_path]

    try:
        feats = []
        emb = compute_yolo_embedding(temp_paths)
        for i, (p, e) in enumerate(zip(temp_paths, emb)):
            h = compute_pdq(p)
            if h == -1 or not np.isfinite(e).all():
                logger.error(f"Failed to extract features for {p}")
                raise ValueError(f"Feature extraction failed for {p}")
            feats.append({"path": p, "pdq": h, "emb": e})

        object_categories = detect_objects(temp_paths)
        semantic_overlap = compute_semantic_overlap(object_categories[0], object_categories[1])
        edge_density_diff = compute_edge_density_difference(temp_image1_path, temp_image2_path)
        color_hist_diff = compute_color_histogram_difference(temp_image1_path, temp_image2_path)

        dpdq = pdq_distance(feats[0]["pdq"], feats[1]["pdq"])
        cs = cosine(feats[0]["emb"], feats[1]["emb"])

        if cs >= COSINE_THRESH and dpdq > PDQ_THRESHOLD:
            logger.info(f"Images skipped: High cosine ({cs:.3f}) but PDQ distance ({dpdq}) > {PDQ_THRESHOLD}")
            return {
                "category": "skipped",
                "reason": "High cosine similarity but PDQ distance exceeds threshold",
                "pdq_distance": dpdq,
                "cosine_similarity": cs
            }

        ss = ssim_score(temp_image1_path, temp_image2_path)
        mse_sim = mse_score(temp_image1_path, temp_image2_path)
        category = categorize(dpdq, cs, ss, mse_sim, semantic_overlap, edge_density_diff, color_hist_diff)

        logger.info(
            f"PDQ Distance: {dpdq} | Cosine Similarity: {cs:.3f} | SSIM: {ss:.3f} | MSE Similarity: {mse_sim:.3f} | "
            f"Semantic Overlap: {semantic_overlap:.3f} | Edge Density Diff: {edge_density_diff:.3f} | Color Hist Diff: {color_hist_diff:.3f} | Category: {category}"
        )

        plot_image_pair(original_image1, original_image2, dpdq, cs, ss, mse_sim, semantic_overlap, edge_density_diff, color_hist_diff, category, output_dir)

        return {
            "category": category,
            "pdq_distance": dpdq,
            "cosine_similarity": cs,
            "ssim": ss,
            "mse_similarity": mse_sim,
            "semantic_overlap": semantic_overlap,
            "edge_density_diff": edge_density_diff,
            "color_hist_diff": color_hist_diff
        }

    finally:
        logger.info("Cleaning up temporary files")
        temp_dir = "temp_resized"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    try:
        result = compare_two_images(IMAGE1, IMAGE2, OUTPUT_DIR)
        logger.info(f"Comparison result: {result}")
        print(f"Comparison result: {result}")
    except Exception as e:
        logger.error(f"Failed to compare images: {str(e)}")
        print(f"Error: Failed to compare images: {str(e)}")
        exit(1)