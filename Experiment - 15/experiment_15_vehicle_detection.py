"""
AML Lab Practical 15
Vehicle Detection in Road Traffic using YOLOv8.
"""

from pathlib import Path
import csv
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
IMAGE_PATH = BASE_DIR / "input" / "traffic.jpg"

OUTPUT_DIR = BASE_DIR / "output"
DETECTION_DIR = OUTPUT_DIR / "detections"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

ANNOTATED_PATH = DETECTION_DIR / "vehicle_detection_output.jpg"
CSV_PATH = DETECTION_DIR / "vehicle_detection_results.csv"
PLOT_PATH = PLOT_DIR / "vehicle_detection_comparison.png"
LOG_PATH = LOG_DIR / "vehicle_detection_log.txt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
}


# ---------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORIES
# ---------------------------------------------------------

for directory in [DETECTION_DIR, PLOT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3. VALIDATE INPUT FILES
# ---------------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Traffic image not found: {IMAGE_PATH}")


# ---------------------------------------------------------
# 4. LOAD IMAGE
# ---------------------------------------------------------

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise ValueError("OpenCV could not read the traffic image.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
output_image = image_rgb.copy()

image_height, image_width = image_rgb.shape[:2]


# ---------------------------------------------------------
# 5. LOAD PRETRAINED YOLO MODEL
# ---------------------------------------------------------

print("=" * 65)
print("AML LAB PRACTICAL 15: VEHICLE DETECTION USING YOLO")
print("=" * 65)

print("\nLoading pretrained YOLOv8 Nano model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Model task:", model.task)
print("Inference device: CPU")
print("CUDA available:", torch.cuda.is_available())


# ---------------------------------------------------------
# 6. RUN OBJECT DETECTION
# ---------------------------------------------------------

print("\nRunning vehicle detection...")

start_time = time.perf_counter()

results = model.predict(
    source=str(IMAGE_PATH),
    conf=CONFIDENCE_THRESHOLD,
    imgsz=IMAGE_SIZE,
    device="cpu",
    verbose=False,
)

inference_time = time.perf_counter() - start_time

result = results[0]


# ---------------------------------------------------------
# 7. FILTER VEHICLE CLASSES
# ---------------------------------------------------------

detections = []
vehicle_counts = {name: 0 for name in sorted(VEHICLE_CLASSES)}

print("\nDetected Vehicles:")
print("-" * 65)

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = model.names[class_id]

        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(
            int, [x1, y1, x2, y2]
        )

        vehicle_counts[class_name] += 1

        detection = {
            "object_id": len(detections) + 1,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        }

        detections.append(detection)

        # Draw bounding box.
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        # Draw class name and confidence.
        label = f"{class_name}: {confidence:.2f}"

        label_y = max(y1 - 10, 20)

        cv2.putText(
            output_image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        print(f"Object ID: {detection['object_id']}")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Bounding Box: {[x1, y1, x2, y2]}")
        print("-" * 65)


# ---------------------------------------------------------
# 8. CALCULATE TOTAL VEHICLE COUNT
# ---------------------------------------------------------

total_vehicles = len(detections)

print("\nVehicle Count Summary:")

for class_name, count in vehicle_counts.items():
    print(f"{class_name}: {count}")

print("Total Vehicles Detected:", total_vehicles)


# ---------------------------------------------------------
# 9. SAVE ANNOTATED IMAGE
# ---------------------------------------------------------

output_bgr = cv2.cvtColor(
    output_image,
    cv2.COLOR_RGB2BGR,
)

if not cv2.imwrite(str(ANNOTATED_PATH), output_bgr):
    raise RuntimeError("Failed to save annotated output image.")

print("\nAnnotated image saved:", ANNOTATED_PATH)


# ---------------------------------------------------------
# 10. SAVE DETECTION RESULTS TO CSV
# ---------------------------------------------------------

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

print("Detection CSV saved:", CSV_PATH)


# ---------------------------------------------------------
# 11. CREATE ORIGINAL VS DETECTED COMPARISON
# ---------------------------------------------------------

figure, axes = plt.subplots(1, 2, figsize=(15, 8))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Traffic Image")
axes[0].axis("off")

axes[1].imshow(output_image)
axes[1].set_title(
    f"Vehicle Detection (Total: {total_vehicles})"
)
axes[1].axis("off")

figure.suptitle(
    "Vehicle Detection using YOLOv8 Nano",
    fontsize=15,
)

figure.tight_layout()
figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
plt.close(figure)

print("Comparison plot saved:", PLOT_PATH)


# ---------------------------------------------------------
# 12. SAVE EXECUTION LOG
# ---------------------------------------------------------

log_lines = [
    "AML LAB PRACTICAL 15 - VEHICLE DETECTION",
    "=" * 55,
    f"Model: {MODEL_PATH.name}",
    f"Input image: {IMAGE_PATH.name}",
    f"Image dimensions: {image_width} x {image_height}",
    f"Confidence threshold: {CONFIDENCE_THRESHOLD}",
    "Inference device: CPU",
    f"Inference time: {inference_time:.4f} seconds",
    "",
    "VEHICLE COUNTS",
    "-" * 55,
]

for class_name, count in vehicle_counts.items():
    log_lines.append(f"{class_name}: {count}")

log_lines.extend([
    f"Total vehicles detected: {total_vehicles}",
    "",
    "INDIVIDUAL DETECTIONS",
    "-" * 55,
])

for detection in detections:
    log_lines.append(
        f"ID={detection['object_id']} | "
        f"Class={detection['class_name']} | "
        f"Confidence={detection['confidence']:.4f} | "
        f"Box={[detection['x1'], detection['y1'], detection['x2'], detection['y2']]}"
    )

log_lines.extend([
    "",
    "GENERATED FILES",
    str(ANNOTATED_PATH.relative_to(BASE_DIR)),
    str(CSV_PATH.relative_to(BASE_DIR)),
    str(PLOT_PATH.relative_to(BASE_DIR)),
])

LOG_PATH.write_text(
    "\n".join(log_lines) + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# 13. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Input image dimensions:", f"{image_width} x {image_height}")
print("Total vehicles detected:", total_vehicles)
print(f"Inference time: {inference_time:.4f} seconds")
print("Annotated output:", ANNOTATED_PATH)
print("CSV results:", CSV_PATH)
print("Comparison plot:", PLOT_PATH)
print("Execution log:", LOG_PATH)

print("\nNOTE:")
print(
    "Only the five configured vehicle categories are counted. "
    "Detection results depend on image content, confidence threshold, "
    "and model performance."
)
