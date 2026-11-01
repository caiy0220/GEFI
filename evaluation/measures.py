import torch, math

def pga_hit(mask, mat, expl_name, resolution=224, patch_size=32):
    '''
    Check whether the highest attributed patch falls within the bounding box / segmentation.
    For GEFI (pixel-level explanation), this is equivalent to check the weighted center of the
    most interacting features of the same quantile (e.g., 1/49 for ViT-B/32).
    -----
    Args:
        mask:       [torch.Tensor] Mask representing the target object, can be bbox or seg.
        mat:        [torch.Tensor] Interaction/Attribution matrix in pixel level.
        expl_name:  [str] Explainer name.
        resolution: [int] Image resolution.
        patch_size: [int] Patch size for patch-based explainers.
    -----
    Returns:
        is_hit: [int] 0 => not hit, 1 => hit.
    '''
    if 'gefi' in expl_name or 'random' in expl_name:
        num_patches = (resolution // patch_size) ** 2
        patch_ratio = 1 - float(1/num_patches)
        pga_thresh = mat.to(dtype=torch.float).quantile(patch_ratio)
        ys, xs = torch.nonzero(mat > pga_thresh, as_tuple=True)
        weights = mat[ys, xs]
        weights = weights / weights.sum()
        max_y = round((ys * weights).float().sum().item())
        max_x = round((xs * weights).float().sum().item())
    else:
        # Other patch-based approaches
        max_val = mat.max()
        y_indices, x_indices = torch.nonzero(mat == max_val, as_tuple=True)
        max_y = round(y_indices.float().mean().item())
        max_x = round(x_indices.float().mean().item())
    
    return int(mask[max_y, max_x] == 1.0)   

def pge_compute(mask, mat, quantile=-1, positive_only=False):
    '''
    Compute Point Game Energy (PGE) given the mask and attribution matrix.
    For the combination of configuration: `quantile=0` and `positive_only=True`,
    the output of this function is equivalent to the original computation process
    defined as: `pos_heatmap = torch.clamp(mat, min=0)`
    -----
    Args:
        mask:          [torch.Tensor] Mask representing the target object, can be bbox or seg.
        mat:           [torch.Tensor] Interaction/Attribution matrix in pixel level.
        quantile:      [float] Specify the quantile of features to be considered.
                               Should be within [0, 1] or the default threshold of 0 is used.
        positive_only: [bool] Only consider positive features if set to be True, PGE becomes unbounded in this case.
    -----
    Returns:
        [float] Averaged Point Game Energy (PGE).
    '''
    thresh = 0.0
    if 0 <= quantile <= 1:
        # Determin threshold by quantile
        valid_values = mat.flatten()
        if positive_only:
            # Consider only positive values if specified
            valid_values = valid_values[torch.nonzero(valid_values>0, as_tuple=True)]
        thresh = valid_values.to(dtype=torch.float).quantile(quantile) 
    pos_heatmap = mat.clone()
    pos_heatmap[pos_heatmap < thresh] = 0.0
    energy_inside = (pos_heatmap * mask).mean().item()
    energy_total = pos_heatmap.mean().item()
    return energy_inside / (energy_total + math.copysign(1e-8, energy_total))

def iou(mask, mat, resolution=224, patch_size=32):
    '''
    Compute IoU by picking the highest attributed features of the same portion as the masked object.
    For a fair comparison with patch-level explainers, the number of features are rounded to fit patch size.
    -----
    Args:
        mask:       [torch.Tensor] Mask representing the target object, can be bbox or seg.
        mat:        [torch.Tensor] Interaction/Attribution matrix in pixel level.
        resolution: [int] Image resolution.
        patch_size: [int] Patch size for patch-based explainers.
    -----
    Returns:
        [float] IoU (Intersection over Union) between the masked object and the most attributed area.
    '''
    num_pixel_per_patch = patch_size ** 2
    num_pixel_total = resolution ** 2
    mask_bool = mask.bool()
    num_masked_pixel = mask_bool.sum().item()
    num_patches = max(1, round(num_masked_pixel / num_pixel_per_patch))
    iou_quantile = 1 - (num_patches * num_pixel_per_patch / num_pixel_total)
    thresh = mat.float().quantile(iou_quantile)
    masked_by_expl = mat >= thresh
    u = masked_by_expl | mask_bool
    i = masked_by_expl & mask_bool
    return (i.sum() / u.sum()).item()


class TrendRecord(object):
    """
    Class to manage changing trend pairs.
    Changing trends of a contrastive pair are recorded by `add_entry`,
    summaries of different kind can be extracted through the other wrapped interfaces.
    """
    def __init__(self):
        self.w_anchor = []  # A list of (num_samples) arrays of shape (num_queries,) 
        self.wo_anchor = []

    def add_entry(self, trd_with_anchor, trd_without_anchor):
        self.w_anchor.append(trd_with_anchor)
        self.wo_anchor.append(trd_without_anchor)
    
    def get_summary(self):
        """
        Average the changing trends across all samples.
        """
        avg_w_anchor = torch.vstack(self.w_anchor).mean(0)  # Shape: (num_queries,)
        avg_wo_anchor = torch.vstack(self.wo_anchor).mean(0)
        return avg_w_anchor, avg_wo_anchor
    
    def get_per_sample_summary(self, normalized=True):
        t_w_anchor = torch.vstack(self.w_anchor)
        t_wo_anchor = torch.vstack(self.wo_anchor)
        norm_factor = t_w_anchor[:, -1].abs() if normalized else 1
        avg_w_anchor = t_w_anchor.mean(1) / norm_factor # (num_samples, )
        avg_wo_anchor = t_wo_anchor.mean(1) / norm_factor
        return avg_w_anchor, avg_wo_anchor
    
    def get_diff(self):
        avg_w, avg_wo = self.get_per_sample_summary()
        # avg_w, avg_wo = self.get_summary()
        return avg_w - avg_wo
    
    def get_var(self):
        avg_w_anchor = torch.vstack(self.w_anchor)
        avg_wo_anchor = torch.vstack(self.wo_anchor)
        var = (avg_wo_anchor - avg_w_anchor).var(dim=0)
        return var
