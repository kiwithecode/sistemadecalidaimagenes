import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def ssim_metrics(img_aligned_bgr, master_bgr):
    img_gray = cv2.cvtColor(img_aligned_bgr, cv2.COLOR_BGR2GRAY)
    master_gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(master_gray, img_gray, data_range=255, full=True,
                       gaussian_weights=True, use_sample_covariance=False)
    diff = (1.0 - diff).astype(np.float32)  # 0=igual, 1=difiere
    return float(score), diff

def make_ssim_overlay(diff_map):
    dm = np.clip(diff_map, 0, 1)
    dm = (dm * 255).astype(np.uint8)
    return cv2.applyColorMap(dm, cv2.COLORMAP_JET)
