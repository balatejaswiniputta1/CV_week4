import argparse
import os
import sys
import cv2
import numpy as np


# -----------------------------
# Utility: metrics
# -----------------------------
def iou_dice(mask_a_u1: np.ndarray, mask_b_u1: np.ndarray):
    a = mask_a_u1.astype(bool)
    b = mask_b_u1.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (a.sum() + b.sum()) if (a.sum() + b.sum()) > 0 else 0.0
    return float(iou), float(dice)


def largest_contour(binary_255: np.ndarray):
    cnts, _ = cv2.findContours(binary_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def overlay_contour(img_bgr: np.ndarray, contour, color=(0, 255, 0), thickness=3):
    out = img_bgr.copy()
    if contour is not None:
        cv2.drawContours(out, [contour], -1, color, thickness)
    return out


# -----------------------------
# Part A: OpenCV (NO ML/DL)
# -----------------------------
def opencv_segment_thermal(img_bgr: np.ndarray, min_area=1500):
    """
    Classical segmentation for thermal-like pseudo-color images:
      - grayscale intensity
      - CLAHE for contrast
      - bilateral filter (edge-preserving)
      - Otsu threshold
      - morphology close/open
      - keep largest contour
    Returns:
      mask_255 (uint8 0/255), contour, debug dict
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    cnt = largest_contour(clean)
    if cnt is None:
        raise RuntimeError("OpenCV: No contour found. (Thresholding failed)")

    if cv2.contourArea(cnt) < min_area:
        raise RuntimeError(
            f"OpenCV: Largest contour too small (area={cv2.contourArea(cnt):.1f}). "
            f"Try lowering --min_area."
        )

    mask = np.zeros_like(clean)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    debug = {
        "gray": gray,
        "enhanced": enhanced,
        "thresh": thresh,
        "clean": clean
    }
    return mask, cnt, debug


# -----------------------------
# Part B: SAM2 (Ultralytics) for comparison
# -----------------------------
def sam2_segment_ultralytics(image_path: str, img_bgr: np.ndarray, sam_model_name: str):
    from ultralytics import SAM
    import numpy as np
    import cv2

    model = SAM(sam_model_name)

    # Automatic masks
    results = model(image_path)
    r0 = results[0]
    if r0.masks is None or r0.masks.data is None:
        raise RuntimeError("SAM2: No masks produced.")

    masks = r0.masks.data.cpu().numpy().astype(np.uint8)  # (N,H,W) values 0/1

    # Use thermal intensity to choose the right mask (bird should be hottest)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    img_area = H * W

    best_score = -1e9
    best_mask = None

    for m in masks:
        area = int(m.sum())
        if area < 0.01 * img_area:   # too small -> skip
            continue
        if area > 0.70 * img_area:   # too big (background/whole image) -> skip
            continue

        # mean temperature (intensity) inside mask
        mean_hot = float(gray[m.astype(bool)].mean())

        # slight bonus for reasonable size so tiny hot specks don’t win
        size_bonus = 0.000001 * area

        score = mean_hot + size_bonus

        if score > best_score:
            best_score = score
            best_mask = m

    if best_mask is None:
        # fallback: pick mask with highest mean_hot regardless of size filters
        best_mask = max(masks, key=lambda m: float(gray[m.astype(bool)].mean()))

    return (best_mask * 255).astype(np.uint8)




def contour_from_mask(mask_255: np.ndarray):
    cnt = largest_contour(mask_255)
    return cnt


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to thermal image (jpg/png).")
    ap.add_argument("--outdir", default="out_final", help="Output directory.")
    ap.add_argument("--min_area", type=int, default=1500, help="Min area for OpenCV contour.")
    ap.add_argument("--save_steps", action="store_true", help="Save OpenCV intermediate steps.")
    ap.add_argument("--run_sam2", action="store_true", help="Also run SAM2 (DL) for comparison.")
    ap.add_argument("--sam_model", default="sam2_b.pt",
                    help="Ultralytics SAM2 weight name/path (e.g., sam2_b.pt, sam2_l.pt).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # ---- OpenCV segmentation ----
    cv_mask, cv_cnt, debug = opencv_segment_thermal(img_bgr, min_area=args.min_area)
    cv_overlay = overlay_contour(img_bgr, cv_cnt, color=(0, 255, 0), thickness=3)

    cv2.imwrite(os.path.join(args.outdir, "mask_cv.png"), cv_mask)
    cv2.imwrite(os.path.join(args.outdir, "overlay_cv.png"), cv_overlay)

    if args.save_steps:
        cv2.imwrite(os.path.join(args.outdir, "step_gray.png"), debug["gray"])
        cv2.imwrite(os.path.join(args.outdir, "step_enhanced.png"), debug["enhanced"])
        cv2.imwrite(os.path.join(args.outdir, "step_threshold.png"), debug["thresh"])
        cv2.imwrite(os.path.join(args.outdir, "step_clean.png"), debug["clean"])

    print(f"[OpenCV] Saved mask_cv.png and overlay_cv.png to {args.outdir}")

    # ---- SAM2 comparison (optional) ----
    if args.run_sam2:
        try:
            sam_mask = sam2_segment_ultralytics(args.image, img_bgr, args.sam_model)
        except Exception as e:
            print("\n[SAM2] FAILED to run.")
            print("Reason:", e)
            print("\nTry:")
            print("  pip install ultralytics")
            print("  or try a different weight name: --sam_model sam2_l.pt")
            sys.exit(1)

        # Resize SAM mask if needed (safety)
        if sam_mask.shape[:2] != cv_mask.shape[:2]:
            sam_mask = cv2.resize(sam_mask, (cv_mask.shape[1], cv_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        sam_cnt = contour_from_mask(sam_mask)
        sam_overlay = overlay_contour(img_bgr, sam_cnt, color=(0, 0, 255), thickness=3)

        # Compare overlay (OpenCV=green, SAM2=red)
        compare = img_bgr.copy()
        if cv_cnt is not None:
            cv2.drawContours(compare, [cv_cnt], -1, (0, 255, 0), 3)
        if sam_cnt is not None:
            cv2.drawContours(compare, [sam_cnt], -1, (0, 0, 255), 3)

        # Metrics
        cv_u1 = (cv_mask > 127).astype(np.uint8)
        sam_u1 = (sam_mask > 127).astype(np.uint8)
        iou, dice = iou_dice(cv_u1, sam_u1)

        cv2.imwrite(os.path.join(args.outdir, "mask_sam2.png"), sam_mask)
        cv2.imwrite(os.path.join(args.outdir, "overlay_sam2.png"), sam_overlay)
        cv2.imwrite(os.path.join(args.outdir, "overlay_compare.png"), compare)

        print(f"[SAM2] Saved mask_sam2.png, overlay_sam2.png, overlay_compare.png to {args.outdir}")
        print(f"[METRICS] IoU={iou:.4f}  Dice={dice:.4f}")
        print("Legend: OpenCV boundary=GREEN, SAM2 boundary=RED")

    else:
        print("[SAM2] Skipped (run with --run_sam2 to compare)")


if __name__ == "__main__":
    main()
