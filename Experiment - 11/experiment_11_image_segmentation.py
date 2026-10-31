"""
PRACTICAL 11: SIMPLE IMAGE SEGMENTATION USING OPENCV
CO MAPPING: CO5

Method:
1. Load the input image.
2. Convert BGR to RGB and grayscale.
3. Apply Gaussian blur.
4. Apply Otsu binary thresholding.
5. Detect external contours.
6. Select the largest normal-threshold contour.
7. Create a mask and extract the ROI.
8. Save the output visualizations.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input" / "object.jpg"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image(filename, image):
    path = OUTPUT_DIR / filename
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save {path}")
    print(f"Saved: {path.relative_to(BASE_DIR)}")


def main():
    print("=" * 65)
    print("PRACTICAL 11: SIMPLE IMAGE SEGMENTATION USING OPENCV")
    print("=" * 65)

    # 1. Load image
    image = cv2.imread(str(INPUT_PATH))
    if image is None:
        raise FileNotFoundError(f"Cannot load input image: {INPUT_PATH}")

    height, width, channels = image.shape
    print("\nINPUT IMAGE DETAILS")
    print(f"File: {INPUT_PATH.name}")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Channels: {channels}")

    # 2. Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image("grayscale_image.png", gray)

    # 4. Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_image("blurred_image.png", blurred)

    # 5. Otsu binary thresholding
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    inverse = cv2.bitwise_not(binary)
    save_image("binary_threshold.png", binary)
    save_image("inverse_threshold.png", inverse)

    print("\nTHRESHOLD INFORMATION")
    print(f"Otsu threshold value: {threshold_value:.2f}")

    # 6. Find contours in the normal binary image
    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("\nCONTOUR ANALYSIS")
    print(f"Normal threshold contour count: {len(contours)}")

    if not contours:
        raise RuntimeError(
            "No contours detected. Try an image with better contrast."
        )

    # Select the largest contour from normal thresholding.
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)

    x, y, roi_width, roi_height = cv2.boundingRect(largest_contour)

    print(f"Largest contour area: {largest_area:.2f} pixels")
    print(
        f"ROI bounding box: x={x}, y={y}, "
        f"width={roi_width}, height={roi_height}"
    )

    # 7. Create a filled contour mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        255,
        cv2.FILLED
    )
    save_image("object_mask.png", mask)

    # 8. Extract the segmented ROI
    segmented = cv2.bitwise_and(image, image, mask=mask)
    save_image("segmented_roi.png", segmented)

    roi_crop = segmented[y:y + roi_height, x:x + roi_width]
    if roi_crop.size == 0:
        raise RuntimeError("ROI extraction produced an empty image.")
    save_image("roi_crop.png", roi_crop)

    # 9. Draw contour on original image
    contour_image = image.copy()
    cv2.drawContours(
        contour_image,
        [largest_contour],
        -1,
        (0, 0, 255),
        3
    )
    save_image("contour_detection.png", contour_image)

    # 10. Threshold diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    diagnostic_views = [
        (rgb, "Original Image", None),
        (gray, "Grayscale Image", "gray"),
        (binary, "Normal Otsu Threshold", "gray"),
        (inverse, "Inverse Otsu Threshold", "gray"),
    ]

    for ax, (view, title, cmap) in zip(
        axes.flat, diagnostic_views
    ):
        ax.imshow(view, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "threshold_diagnostic.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved: output/threshold_diagnostic.png")

    # 11. Final six-panel visualization
    panels = [
        (rgb, "Original Image", None),
        (gray, "Grayscale Image", "gray"),
        (binary, "Binary Threshold", "gray"),
        (mask, "Object Mask", "gray"),
        (cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), "Segmented ROI", None),
        (
            cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB),
            "Detected Contour",
            None
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for ax, (view, title, cmap) in zip(axes.flat, panels):
        ax.imshow(view, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        "Practical 11 - Simple Image Segmentation Using OpenCV",
        fontsize=16
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "image_segmentation_result.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved: output/image_segmentation_result.png")

    # 12. Save execution log
    log = [
        "PRACTICAL 11: SIMPLE IMAGE SEGMENTATION USING OPENCV",
        "=" * 55,
        f"Input image: {INPUT_PATH.name}",
        f"Image dimensions: {width} x {height}",
        f"Channels: {channels}",
        f"Otsu threshold: {threshold_value:.2f}",
        "Threshold direction: NORMAL",
        f"Contour count: {len(contours)}",
        f"Largest contour area: {largest_area:.2f}",
        f"ROI bounding box: ({x}, {y}, {roi_width}, {roi_height})",
        f"Mask foreground percentage: {np.count_nonzero(mask) / mask.size * 100:.2f}%",
        "",
        "Generated files:",
        "grayscale_image.png",
        "blurred_image.png",
        "binary_threshold.png",
        "inverse_threshold.png",
        "object_mask.png",
        "segmented_roi.png",
        "roi_crop.png",
        "contour_detection.png",
        "threshold_diagnostic.png",
        "image_segmentation_result.png",
    ]

    (OUTPUT_DIR / "experiment_log.txt").write_text(
        "\n".join(log) + "\n",
        encoding="utf-8"
    )
    print("Saved: output/experiment_log.txt")

    print("\n" + "=" * 65)
    print("EXPERIMENT EXECUTION COMPLETED")
    print("=" * 65)
    print(f"Selected threshold: NORMAL")
    print(f"Largest contour area: {largest_area:.2f}")
    print(f"ROI bounding box: ({x}, {y}, {roi_width}, {roi_height})")
    print(
        f"Mask foreground percentage: "
        f"{np.count_nonzero(mask) / mask.size * 100:.2f}%"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print("STATUS: EXECUTION SUCCESSFUL")


if __name__ == "__main__":
    main()
