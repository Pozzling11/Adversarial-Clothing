import cv2
import numpy as np
import torch

def yolo_saliency_map(model, img_t, device):
    """
    Simple saliency: gradient of person confidence w.r.t. input image.
    img_t: (1,3,H,W) float32 [0,1] torch tensor
    Returns: (H,W) numpy array, normalized
    """
    img_t = img_t.clone().detach().to(device).requires_grad_(True)
    out = model(img_t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    # Person confidence channel (assume channel 4)
    person_scores = out[0, 4, :]  # (N_anchors,)
    score = person_scores.mean()
    score.backward()
    grad = img_t.grad[0].cpu().numpy()  # (3,H,W)
    sal = np.abs(grad).max(axis=0)  # (H,W)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal
