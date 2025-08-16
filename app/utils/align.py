import cv2, numpy as np
from typing import Tuple, Optional
from app.utils.ssim import ssim_metrics

def _pre_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE ayuda mucho en empaques brillantes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def _homography_align(src, tgt, max_w=2200, nfeatures=6000,
                      min_inliers=25, min_inlier_ratio=0.35
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    def resize_keep(img, w):
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    Ssrc, Stgt = src, tgt
    if tgt.shape[1] > max_w: Stgt = resize_keep(tgt, max_w)
    if src.shape[1] > max_w: Ssrc = resize_keep(src, max_w)

    g1 = _pre_gray(Ssrc)
    g2 = _pre_gray(Stgt)

    # AKAZE → Hamming
    try:
        det = cv2.AKAZE_create()
        k1, d1 = det.detectAndCompute(g1, None)
        k2, d2 = det.detectAndCompute(g2, None)
        norm = cv2.NORM_HAMMING
    except Exception:
        det = cv2.ORB_create(nfeatures)
        k1, d1 = det.detectAndCompute(g1, None)
        k2, d2 = det.detectAndCompute(g2, None)
        norm = cv2.NORM_HAMMING

    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, None

    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if n is not None and m.distance < 0.70 * n.distance]
    if len(good) < 12:
        return None, None

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, None

    inliers = int(mask.sum())
    inlier_ratio = inliers / max(len(good), 1)
    if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return None, None

    # evitar homografías degeneradas
    if not np.isfinite(H).all() or abs(np.linalg.det(H[:2,:2])) < 1e-6 or abs(H[2,2]) < 1e-6:
        return None, None

    # escalar a tamaño del master original
    scale_x = tgt.shape[1] / Stgt.shape[1]
    scale_y = tgt.shape[0] / Stgt.shape[0]
    S = np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]], dtype=np.float32)
    sx = Ssrc.shape[1] / src.shape[1]
    sy = Ssrc.shape[0] / src.shape[0]
    Sinv = np.array([[1/sx,0,0],[0,1/sy,0],[0,0,1]], dtype=np.float32)

    H_full = S @ H @ Sinv
    aligned = cv2.warpPerspective(src, H_full, (tgt.shape[1], tgt.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, H_full

def _ecc_refine(aligned, master, iters=80):
    try:
        g1 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
        warp = np.eye(2,3, dtype=np.float32)  # afin
        cc, warp = cv2.findTransformECC(g2, g1, warp, cv2.MOTION_AFFINE,
                                        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, iters, 1e-5))
        refined = cv2.warpAffine(aligned, warp, (master.shape[1], master.shape[0]),
                                 flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return refined
    except Exception:
        return aligned

def align_to_master(img_bgr, master_bgr, try_rotations=(0,90,180,270)):
    best = None; bestH = None; bestScore = -1.0
    for rot in try_rotations:
        src = img_bgr if rot == 0 else cv2.rotate(
            img_bgr, {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
        )
        aligned, H = _homography_align(src, master_bgr)
        if aligned is None:
            continue
        aligned = _ecc_refine(aligned, master_bgr, iters=80)
        score, _ = ssim_metrics(aligned, master_bgr)
        if score > bestScore:
            best, bestH, bestScore = aligned, H, score
    return best, bestH
