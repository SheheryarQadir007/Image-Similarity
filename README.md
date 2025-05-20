# 🧠 Near-Duplicate Image Detection with PDQ + YOLOv8 Embeddings

This project compares two images to determine if they are **exactly similar**, **somewhat similar**, or **completely different**, using a multi-metric approach that combines:

- ✅ **PDQ Hash Distance**
- ✅ **YOLOv8 Feature Embeddings**
- ✅ **SSIM and MSE Similarity**
- ✅ **Edge Density and Color Histogram Comparison**
- ✅ **Semantic Object Overlap**

---

## 🔍 Use Cases

- Detect near-duplicate images in datasets
- Identify redundant or similar images from surveillance, real estate, or catalog imagery
- Evaluate image tampering or visual changes using robust hash and feature analysis

---

## 🚀 Features

- 🔎 Detects visual similarity using **PDQ**, **YOLO**, and **SSIM**
- 📦 Accepts both **local image paths** and **image URLs**
- 📈 Outputs a categorized result: `exact_similar`, `somewhat_similar`, `completely_different`
- 📁 Saves side-by-side **comparison visualizations**
- ⚡ Supports **CUDA** acceleration (if available)
- 📑 Detailed logging and fail-safe cleanup of temporary files

---

## 🗂️ Output Example

📊 Comparison result:
{
"category": "exact_similar",
"pdq_distance": 14,
"cosine_similarity": 0.95,
"ssim": 0.87,
"mse_similarity": 0.93,
"semantic_overlap": 1.0,
"edge_density_diff": 0.012,
"color_hist_diff": 0.078
}



---

## 📁 Directory Structure

.
├── main.py # Entry point for image comparison
├── image_comparison/ # Visual comparison outputs
├── image_comparison.log # Logs for debug & analysis
└── requirements.txt # Dependency file


---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/image-duplicate-detector.git
cd image-duplicate-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
