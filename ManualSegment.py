import cv2
import numpy as np
import os

# =========================
# Configuration
# =========================

IMAGE_PATH = "datasets/new_data/2025_02_24___12_28_15_rgb.jpg"     # path to your RGB image
OUTPUT_MASK_PATH = "RecSeg3D_data/permanent_data/Segmentations/mask.png"    # output binary mask

WINDOW_NAME = "Manual Segmentation (Click points, press 'q' to finish)"

# =========================
# Global state
# =========================
points = []
image = None
image_vis = None


def mouse_callback(event, x, y, flags, param):
    global points, image_vis

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Draw point
        cv2.circle(image_vis, (x, y), 4, (0, 0, 255), -1)

        # Draw lines between points
        if len(points) > 1:
            cv2.line(
                image_vis,
                points[-2],
                points[-1],
                (0, 255, 0),
                2
            )

        cv2.imshow(WINDOW_NAME, image_vis)


def create_mask(image_shape, polygon_points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    polygon = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def main():
    global image, image_vis

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise RuntimeError("Failed to load image")

    image_vis = image.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Instructions:")
    print(" - Left click to add polygon points")
    print(" - Minimum 3 points required")
    print(" - Press 'q' to finish and save mask")
    print(" - Press 'r' to reset")

    while True:
        cv2.imshow(WINDOW_NAME, image_vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points.clear()
            image_vis = image.copy()
            print("Reset points")

        elif key == ord('q'):
            if len(points) < 3:
                print("Need at least 3 points to create a mask")
                continue
            break

    cv2.destroyAllWindows()

    # Close polygon visually
    image_closed = image.copy()
    cv2.polylines(
        image_closed,
        [np.array(points, dtype=np.int32)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2
    )

    # Create and save mask
    mask = create_mask(image.shape, points)
    cv2.imwrite(OUTPUT_MASK_PATH, mask)

    print(f"Mask saved to: {OUTPUT_MASK_PATH}")

    # Optional: show final result
    overlay = image.copy()
    overlay[mask == 255] = (0.5 * overlay[mask == 255] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    cv2.imshow("Final Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
