import cv2, numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

def deltaE_stats_with_lab(img_aligned_bgr, master_bgr, mask=None):
    img_rgb = cv2.cvtColor(img_aligned_bgr, cv2.COLOR_BGR2RGB)
    mast_rgb = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2RGB)
    img_lab = rgb2lab(img_rgb)
    mast_lab = rgb2lab(mast_rgb)
    dE_map = deltaE_ciede2000(img_lab, mast_lab)

    if mask is not None:
        m = mask > 0
        vals_img = img_lab[m]; vals_mast = mast_lab[m]; vals_de = dE_map[m]
    else:
        vals_img = img_lab.reshape(-1,3)
        vals_mast = mast_lab.reshape(-1,3)
        vals_de = dE_map.reshape(-1)

    lab_avg_img = np.mean(vals_img, axis=0)
    lab_avg_master = np.mean(vals_mast, axis=0)
    lab_diff = lab_avg_img - lab_avg_master

    return {
        "dE_avg": float(np.mean(vals_de)),
        "dE_max": float(np.max(vals_de)),
        "lab_avg_img": lab_avg_img,
        "lab_avg_master": lab_avg_master,
        "lab_diff": lab_diff,
        "dE_map_full": dE_map
    }
