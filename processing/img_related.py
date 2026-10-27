# Utility functions for image related processing and visualization

from PIL import Image
from typing import Tuple
from torchvision.transforms.functional import center_crop, resize
import torch

PIX_STAT = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
PIX_MEAN, PIX_STD = torch.tensor(PIX_STAT[0]), torch.tensor(PIX_STAT[1])
IMG_SIZE = (224, 224)

def crop(img: Image, resolution: int, bbox: Tuple[int] = None):
    '''
    Args:
        img:              [PIL.Image] The image to be cropped in original coordinates.
        resolution:       [int] Target resolution (e.g., 224).
        bbox:             [Tuple(int)] Optional bounding box (x_min, y_min, width, height) in the original image coordinates.
    -----
    Returns:
        cropped_img:      [PIL.Image] The cropped image of shape (resolution, resolution).
        transformed_bbox: [List(int)] If bbox is provided, the transformed bounding box coordinates of shape (x_min, y_min, width, height) in the cropped image.
        is_truncated:     [bool] If bbox is provided, indicates whether the bounding box was partially or fully truncated due to center cropping.
    '''
    # Resize the image such that its shortest edge matches the target resolution
    resized = resize(img, resolution, interpolation=Image.BICUBIC)
    # Calculate the uniform scaling factor
    f_x = resized.size[0] / img.size[0]
    f_y = resized.size[1] / img.size[1]
    # Perform a center crop on the resized image to achieve a square output (resolution x resolution)
    cropped = center_crop(resized, resolution)
    # Calculate the coordinate shift introduced by the center crop
    sx = (resized.size[0] - cropped.size[0]) // 2
    sy = (resized.size[1] - cropped.size[1]) // 2
    # If a bounding box is provided, transform its coordinates
    if bbox is not None:
        x, y, w, h = bbox
        x, w = x * f_x, w * f_x
        y, h = y * f_y, h * f_y
        x_max, y_max = round(x + w), round(y + h)
        x, y = round(x), round(y)
        w, h = x_max - x, y_max - y
        x_min = x - sx
        x_max = x_min + w
        y_min = y - sy
        y_max = y_min + h
        is_truncated = (x_min < 0) or (y_min < 0) or (x_max > resolution) or (y_max > resolution)
        x_min = max(min(x_min, resolution), 0)
        y_min = max(min(y_min, resolution), 0)
        x_max = max(min(x_max, resolution), 0)
        y_max = max(min(y_max, resolution), 0)
        w_new = x_max - x_min
        h_new = y_max - y_min
        transformed_bbox = [x_min, y_min, w_new, h_new]
        return cropped, transformed_bbox, is_truncated
    else:
        return cropped

def image_crop(img: Image, resolution: int):
    resized = resize(img, resolution, interpolation=Image.BICUBIC)
    f_x = resized.size[0] / img.size[0]
    f_y = resized.size[1] / img.size[1]
    # Perform a center crop on the resized image to achieve a square output (resolution x resolution)
    cropped = center_crop(resized, resolution)
    # Calculate the coordinate shift introduced by the center crop
    sx = (resized.size[0] - cropped.size[0]) // 2
    sy = (resized.size[1] - cropped.size[1]) // 2
    return cropped, (f_x, f_y), (sx, sy)

def tensor_crop(img: torch.Tensor, resolution: int):
    '''
    Args:
        img:              [PIL.Image] The image to be cropped in original coordinates.
        resolution:       [int] Target resolution (e.g., 224).
    -----
    Returns:
        cropped_img:      [PIL.Image] The cropped image of shape (resolution, resolution).
        transformed_bbox: [List(int)] If bbox is provided, the transformed bounding box coordinates of shape (x_min, y_min, width, height) in the cropped image.
        is_truncated:     [bool] If bbox is provided, indicates whether the bounding box was partially or fully truncated due to center cropping.
    '''
    # Resize the image such that its shortest edge matches the target resolution
    img = img.unsqueeze(0) if img.dim() == 2 else img
    resized = resize(img, resolution, interpolation=Image.BICUBIC)
    # Perform a center crop on the resized image to achieve a square output (resolution x resolution)
    cropped = center_crop(resized, resolution)
    return cropped.reshape(resolution, resolution)

def truncated_bbox(bbox, scales, shifts, resolution, MIN_BBOX_SIZE=32):
    x, y, w, h = bbox
    fx, fy = scales
    sx, sy = shifts
    x, w = x * fx, w * fx
    y, h = y * fy, h * fy
    x_max, y_max = round(x + w), round(y + h)
    x, y = round(x), round(y)
    w, h = x_max - x, y_max - y
    is_too_small = (w < MIN_BBOX_SIZE) or (h < MIN_BBOX_SIZE)
    x_min = x - sx
    x_max = x_min + w
    y_min = y - sy
    y_max = y_min + h
    is_truncated = (x_min < 0) or (y_min < 0) or (x_max > resolution) or (y_max > resolution)
    return is_truncated or is_too_small

def mask_bbox(mask, bbox):
    left, top, width, height = bbox
    mask[top:top+height, left:left+width] = 127
    return mask


def unnormalize2show(img):
    img = img[0] if img.dim() == 4 else img
    return (img.permute(1,2,0).cpu() * PIX_STD) + PIX_MEAN

