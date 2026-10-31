from pathlib import Path
import csv
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
IMAGE_PATH = BASE_DIR / "input" / "object.jpg"

DETECTION_DIR = BASE_DIR / "output" / "detections"
PLOT_DIR = BASE_DIR / "output" / "plots"
LOG_DIR = BASE_DIR / "output" / "logs"

for directory in (DETECTION_DIR, PLOT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ANNOTATED_PATH = DETECTION_DIR / "detected_output.jpg"
CSV_PATH = DETECTION_DIR / "detection_results.csv"
COMPARISON_PATH = PLOT_DIR / "detection_comparison.png"
LOG_PATH = LOG_DIR / "detection_log.txt"


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

original_bgr = cv2.imread(str(IMAGE_PATH))

if original_bgr is None:
    raise ValueError(f"OpenCV could not read the image: {IMAGE_PATH}")

original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 3. LOAD PRETRAINED YOLO MODEL
# ============================================================

print("=" * 65)
print("PRACTICAL 14: YOLO OBJECT DETECTION")
print("=" * 65)

print("\nLoading pretrained YOLOv8 Nano model...")
model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Model task:", model.task)
print("Number of supported classes:", len(model.names))
print("Inference device: CPU")


# ============================================================
# 4. PERFORM OBJECT DETECTION
# ============================================================

print("\nRunning object detection...")

start_time = time.perf_counter()

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    device="cpu",
    verbose=False
)

elapsed_seconds = time.perf_counter() - start_time

result = results[0]


# ============================================================
# 5. SAVE ANNOTATED IMAGE
# ============================================================

annotated_bgr = result.plot()

if not cv2.imwrite(str(ANNOTATED_PATH), annotated_bgr):
    raise RuntimeError("Failed to save the annotated detection image.")

annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 6. EXTRACT DETECTION INFORMATION
# ============================================================

detections = []

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        detections.append({
            "object_id": len(detections) + 1,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
        })


# ============================================================
# 7. EXPORT DETECTION RESULTS TO CSV
# ============================================================

csv_fields = [
    "object_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]

with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(detections)


# ============================================================
# 8. GENERATE ORIGINAL VS DETECTED COMPARISON
# ============================================================

figure, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(original_rgb)
axes[0].set_title("Original Input Image", fontsize=13)
axes[0].axis("off")

axes[1].imshow(annotated_rgb)
axes[1].set_title("YOLOv8 Object Detection", fontsize=13)
axes[1].axis("off")

figure.suptitle(
    "Practical 14 - Object Detection using YOLOv8",
    fontsize=16,
    fontweight="bold"
)

figure.tight_layout()
figure.savefig(COMPARISON_PATH, dpi=200, bbox_inches="tight")
plt.close(figure)


# ============================================================
# 9. GENERATE EXECUTION LOG
# ============================================================

image_height, image_width = original_bgr.shape[:2]

class_counts = {}

for detection in detections:
    name = detection["class_name"]
    class_counts[name] = class_counts.get(name, 0) + 1

log_lines = [
    "PRACTICAL 14: OBJECT DETECTION USING YOLOv8",
    "=" * 55,
    f"Input image: {IMAGE_PATH.name}",
    f"Image dimensions: {image_width} x {image_height}",
    f"Model: {MODEL_PATH.name}",
    "Model architecture: YOLOv8 Nano",
    "Dataset classes: COCO pretrained classes",
    "Inference device: CPU",
    "Confidence threshold: 0.25",
    f"Inference time: {elapsed_seconds:.4f} seconds",
    f"Total detected objects: {len(detections)}",
    "",
    "DETECTION SUMMARY",
    "-" * 55,
]

if class_counts:
    for class_name, count in sorted(class_counts.items()):
        log_lines.append(f"{class_name}: {count}")
else:
    log_lines.append("No objects detected above the confidence threshold.")

log_lines.extend([
    "",
    "DETAILED DETECTIONS",
    "-" * 55,
])

for detection in detections:
    log_lines.append(
        f"ID {detection['object_id']}: "
        f"{detection['class_name']} | "
        f"Confidence: {detection['confidence']:.4f} | "
        f"Bounding box: "
        f"({detection['x1']}, {detection['y1']}, "
        f"{detection['x2']}, {detection['y2']})"
    )

log_lines.extend([
    "",
    "GENERATED OUTPUTS",
    "-" * 55,
    f"Annotated image: {ANNOTATED_PATH.relative_to(BASE_DIR)}",
    f"Detection CSV: {CSV_PATH.relative_to(BASE_DIR)}",
    f"Comparison plot: {COMPARISON_PATH.relative_to(BASE_DIR)}",
])

LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


# ============================================================
# 10. DISPLAY ACTUAL EXPERIMENTAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("OBJECT DETECTION RESULTS")
print("=" * 65)

print("Input image dimensions:", f"{image_width} x {image_height}")
print("Total detected objects:", len(detections))
print("Inference time:", f"{elapsed_seconds:.4f} seconds")

if detections:
    print("\nDetected objects:")

    for detection in detections:
        print(
            f"  {detection['object_id']}. "
            f"{detection['class_name']} | "
            f"Confidence: {detection['confidence']:.4f} | "
            f"Bounding box: "
            f"({detection['x1']}, {detection['y1']}, "
            f"{detection['x2']}, {detection['y2']})"
        )
else:
    print("\nNo objects detected at confidence threshold 0.25.")

print("\nGenerated files:")
print("  Annotated image:", ANNOTATED_PATH)
print("  Detection CSV:", CSV_PATH)
print("  Comparison plot:", COMPARISON_PATH)
print("  Execution log:", LOG_PATH)

print("\nPRACTICAL 14 DETECTION COMPLETED SUCCESSFULLY.")
