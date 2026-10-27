import numpy as np
import torch
from processing import img_related
from torch.nn import functional as F
# import cv2

def highlight_masked_region(img, mask, boundary_width=5, alpha=0.15, edgecolor=(0.5,0.5,0.5)):
    boundary_width = boundary_width // 2 * 2 + 1    # boundary width must be an odd integer
    img_annotated = img_related.unnormalize2show(img)
    dilated = F.max_pool2d(mask.float()[None, None], boundary_width,
                    stride=1, padding=boundary_width // 2)[0, 0].bool()

    if edgecolor is not None:
        img_annotated[dilated & ~mask] = torch.tensor(edgecolor, dtype=img_annotated.dtype)
    alphas = alpha * (~dilated).float() + dilated.float()
    rgba = torch.cat([img_annotated, alphas.unsqueeze(-1)], dim=-1)
    return rgba

'''
==================================================================
|                Utility funcs for generating GIF                |
==================================================================
'''
class CocoAnnotation:
    def __init__(self):
        self.det_masks = dict()
        self.seg_masks = dict()

    def update_det(self, label, det_mask):
        self.det_masks[label] = det_mask

    def update_seg(self, label, seg_mask):
        self.seg_masks[label] = seg_mask

    def __len__(self):
        return len(self.det_masks)
    

def color_hex2rgb(s):
    s = s[1:] if s[0] == '#' else s
    assert len(s) == 6
    rgb = []
    for i in range(0, 6, 2):
        substr = s[i:i+2]
        rgb.append(int(substr, 16)/255)
    return rgb


class RoundMask:
    def __init__(self, max_width, radius):
        y, x = np.ogrid[:max_width, :max_width]
        self.shift = max_width//2
        self.mask = (x - self.shift)**2 + (y - self.shift)**2 <= radius**2
        self.mask = torch.tensor(self.mask)

    def generate_mask(self, y, x, size=224):
        mask = torch.zeros((size, size))
        mask[y-self.shift:y+self.shift+1, x-self.shift:x+self.shift+1] = self.mask
        return mask
    
def curved_trajectory( start=(31, 201), end=(161, 31), n_points=300, bend=80, integer=True,):
    """
    Generate a smooth downward-bending trajectory.
    Coordinates are (row, col), so increasing row means downward.
    """

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    t = np.linspace(0, 1, n_points)
    midpoint = (start + end) / 2

    # Move control point downward by increasing the row coordinate
    control = midpoint + np.array([bend, 0])
    points = (
        (1 - t)[:, None] ** 2 * start
        + 2 * (1 - t)[:, None] * t[:, None] * control
        + t[:, None] ** 2 * end
    )

    if integer:
        points = np.round(points).astype(int)
        # Remove duplicate consecutive matrix coordinates
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        points = points[keep]

    return points

def line_trajectory(start=(31, 201), end=(161, 31), n_points=300, integer=True,):
    """
    Generate a straight trajectory from start to end.
    Coordinates are (row, col), suitable for matrix indexing.
    """

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    t = np.linspace(0, 1, n_points)

    points = (1 - t)[:, None] * start + t[:, None] * end

    if integer:
        points = np.round(points).astype(int)

        # Remove duplicate consecutive matrix coordinates
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        points = points[keep]

    return points