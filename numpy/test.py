import numpy as np
import cv2

def generate_fake_ct(size=512):
    img = np.zeros((size, size), np.uint8)

    # phổi
    cv2.ellipse(img, (size//3, size//2), (80,120), 0, 0, 360, 120, -1)
    cv2.ellipse(img, (2*size//3, size//2), (80,120), 0, 0, 360, 120, -1)

    img = cv2.GaussianBlur(img, (101,101), 0)

    # u
    cv2.circle(img, (size//3+30, size//2-20), 18, 200, -1)

    img = img + np.random.normal(0,5,img.shape).astype(np.uint8)

    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

from segment_anything import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

image = generate_fake_ct()

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85,
    min_mask_region_area=300
)

masks = mask_generator.generate(image)
print("Number of masks:", len(masks))

masks = mask_generator.generate(image)
print("Number of masks:", len(masks))