import numpy as np

def crop_template_to_visible_region(template_mask: np.ndarray, seg_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Crop the template mask vertically to match the visible region in the segmentation mask.
    Returns the cropped template and the number of rows cropped from the top.
    Both masks must be the same shape (H, W), binary (0/255 or 0/1).
    """
    # Find vertical extent of visible region in seg_mask
    rows = np.any(seg_mask > 0, axis=1)
    if not np.any(rows):
        # No visible region, return original
        return template_mask, 0
    top = np.argmax(rows)
    bottom = len(rows) - 1 - np.argmax(rows[::-1])
    # Crop template to this vertical extent
    cropped = template_mask[top:bottom+1, :]
    return cropped, top