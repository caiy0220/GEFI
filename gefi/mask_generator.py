from torchvision.transforms import GaussianBlur
import torch
from torchvision.transforms.functional import resize, InterpolationMode
from tqdm import tqdm
from utils import deprecated

''' Proxy Gradient Estimation masks '''
class PGE_mask(object):
    def __init__(self, masks, paths, weighted=True):
        """
        A structure that collects mask-relevant entities
        """
        self.masks = masks
        self.paths = paths
        self.weights = self._get_mask_weights() if weighted else self._uniform_weights()
        self.budget = len(masks)

    def _uniform_weights(self):
        masks_in_float, gammas = self.masks.to(torch.float), self.paths
        ws = (masks_in_float - gammas) 
        return ws

    def _get_mask_weights(self):
        masks_in_float, gammas = self.masks.to(torch.float), self.paths
        ws = (masks_in_float - gammas) / (gammas * (1-gammas))
        return ws

""" 
Abstract class for sampler that generates masks and paths, 
which collects observations for estimation. 
"""
class Sampler(object):
    def __init__(self, sample_num, size, grid_num=10, truncation=0.):
        self.sample_num = sample_num // 2  # Halve the sample number due to antithetic sampling
        self.size = size  
        self.grid_num = grid_num
        self.truncation = truncation    # Truncation for variance-bias trade-off, cut-off heavy estimation tails
        self.verbose = False

    def _get_masks(self, num, mask_shape):
        """ 
        Generate masks and paths 
        Returns: 
            masks: (sample_num, size) random noise as initial masks
            paths: (sample_num) location on proxy path [truncation, 1-truncation]
        """
        masks = torch.rand((num, *mask_shape))  
        paths = self._get_paths(num, mask_shape)
        return masks, paths

    def _get_paths(self, num, mask_shape):
        range_l, range_r = self.truncation, 1-self.truncation
        '''
        Approximatinng the path integral through a Riemann sum.
        Returns: 
            paths: (sample_num) location on proxy path [truncation, 1-truncation]
        ''' 
        if self.truncation == 0.:
            # Safely ignoring the full absence & presence as they do not distinguish different features
            grids = torch.linspace(range_l, range_r, self.grid_num+2)[1:-1]
        else:
            grids = torch.linspace(range_l, range_r, self.grid_num)
        
        repeats = num // self.grid_num
        residual = num % self.grid_num
        residual_idxs = torch.randperm(self.grid_num)[:residual]
        residual_grids = grids[residual_idxs]

        grids = torch.concat([grids.tile(repeats), residual_grids])
        appended_dims = (1,) * len(mask_shape)
        return grids.reshape(-1,*appended_dims)  # Aligning the shape with masks
    
    def _mask_path_overlay(self, masks, paths, num_features):
        """ 
        Integrating masks and path anchors to determine feature absence/presence 
        Return:
            masks: (sample_num, size) binarization based on path anchors
        """
        hits = masks < paths
        dims = torch.arange(1, masks.dim()).tolist()
        presence_counts = hits.sum(dim=dims)

        # Exclude full absence and presence as they contribute to all features the same values
        valid_idxs = torch.where((presence_counts > 0) & (presence_counts < num_features))[0][:self.sample_num]
        hits, paths = hits[valid_idxs], paths[valid_idxs]

        # Antithetic Sampling
        hits = torch.concat([hits, ~hits]) 
        paths = torch.concat([paths, 1-paths]) 
        return hits, paths 

    def _reassign_mask_values(self, masks, num_features):
        masks = masks.reshape(-1, num_features)
        order = masks.argsort(dim=1)

        # base = torch.linspace(0, 1, num_features)
        # base = base.unsqueeze(0).expand_as(masks)
        # pseudo_uniforms = base + torch.rand_like(masks) / num_features
        ranks = torch.arange(
            num_features, device=masks.device, dtype=masks.dtype
        )
        pseudo_uniforms = (ranks.unsqueeze(0) + torch.rand_like(masks)) / num_features

        new_masks = torch.empty_like(masks)
        new_masks.scatter_(1, order, pseudo_uniforms)
        return new_masks
    
    @deprecated
    def _deprecated_reassign_mask_values(self, masks, batch_size, verbose=False):
        buff = []
        with torch.no_grad():
            for ptr_l in tqdm(range(0, len(masks), batch_size), disable=not verbose):
                ptr_r = ptr_l + batch_size
                batch = masks[ptr_l:ptr_r]
                batch_ranking = torch.sort(batch.cuda()).indices.cpu()
                pseudo_uniforms = get_ranked_pseudo_uniform(len(batch))

                batch_uniform = torch.zeros_like(pseudo_uniforms)
                batch_uniform[batch_ranking] = pseudo_uniforms
                buff.append(batch_uniform)
        return torch.concat(buff)
    
def get_ranked_pseudo_uniform(num):
    grids = torch.linspace(0, 1, num)
    pseudo_uniforms = grids + torch.rand_like(grids) / num
    return pseudo_uniforms
    

class TextSampler(Sampler):
    def __init__(self, sample_num, size, grid_num=10, truncation=0.):
        super().__init__(sample_num, size, grid_num, truncation)

    def get_masks(self, weighted=True):
        """ Text mask generation """
        num_features = self.size    # Texts are 1-D
        num_masks_oversampling = int(self.sample_num + 2 * (self.sample_num / num_features + 1))
        masks, paths = self._get_masks(num_masks_oversampling, (num_features,))
        masks, paths = self._mask_path_overlay(masks, paths, num_features)
        return PGE_mask(masks, paths, weighted)

class ImageSampler(Sampler):
    def __init__(self, sample_num, target_size, patch_size, grid_num=10, truncation=0.,):
        super().__init__(sample_num, target_size, grid_num, truncation)
        self.patch_size = torch.tensor((patch_size, patch_size)) if isinstance(patch_size, int) else torch.tensor(patch_size)
        self.size = torch.tensor(target_size)[-2:]

    def get_masks(self, weighted=True):
        """ Image mask generation """
        num_features = self.size[-2:].prod()
        num_masks_oversampling = int(self.sample_num + 2 * (self.sample_num / num_features + 1))

        ''' Mask resizing from patch_size to target_size '''
        init_mask_size = self.size[-2:] // self.patch_size
        # Sampling one more patch for random cropping
        raw_masks, paths = self._get_masks(num_masks_oversampling, (1, *(init_mask_size+1)))
        masks = resize(raw_masks, self.size[-2:]+self.patch_size, InterpolationMode.BICUBIC)
        masks = self._per_mask_random_cropping(masks, *self.patch_size)

        masks = self._reassign_mask_values(masks, num_features).reshape(-1, 1, *self.size)
        masks, paths = self._mask_path_overlay(masks, paths, num_features)
        return PGE_mask(masks, paths, weighted)

    def _mask_path_overlay(self, masks, paths, num_features):
        # Binarization by ranking, as mask values are not uniformly distributed after resizing
        flat_masks = masks.reshape(-1, num_features)
        order = flat_masks.argsort(dim=1)

        # Equivalent distribution to thresholding (rank + jitter) / N.
        expected_counts = paths.reshape(-1, 1) * num_features
        counts = expected_counts.floor()
        counts += (torch.rand_like(expected_counts) < expected_counts - counts)
        counts = counts.clamp(min=1, max=num_features-1)

        ranks = torch.arange(num_features, device=masks.device)[None, :]
        sorted_hits = ranks < counts

        hits = torch.empty_like(flat_masks, dtype=torch.bool)
        hits.scatter_(1, order, sorted_hits)
        hits = hits.reshape_as(masks)

        # Antithetic sampling
        hits = torch.cat([hits, ~hits])
        paths = torch.cat([paths, 1 - paths])
        return hits, paths
    
    def _per_mask_random_cropping(self, masks, max_y, max_x):
        tops = torch.randint(0, max_y, (len(masks),), device=masks.device)
        lefts = torch.randint(0, max_x, (len(masks),), device=masks.device)
        new_masks = []
        for top, left, mask in zip(tops, lefts, masks):
            new_masks.append(mask[:, top:top+self.size[0], left:left+self.size[1]])
        return torch.vstack(new_masks).reshape(-1, 1, *self.size)

