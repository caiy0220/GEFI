import torch, math
import numpy as np
import utils
from utils import deprecated, log
from gefi.mask_generator import TextSampler, ImageSampler
from gefi.baseline_generator import TextBaseline, ImageBaseline
from tqdm import tqdm
import torch.nn.functional as F

SOT_ID, EOT_ID, PAD_ID = 49406, 49407, 0

class GEFI4Clip(object):
    def __init__(self, sample_num, img_size, txt_size, img_baseline=0, txt_baseline=0, 
                 patch_size=16, grid_num=10, truncation=0., img_ratio=0.5):
        """
        Parameters
        --------------------
        sample num:     [int] query budget
        img_size:       [tuple, tensor] specifying image shape in order (C, H, W)
        text_size:      [int] specifying length of text size
        img_baseline:   [str, tensor, int/float, ...] refer to get_baseline function for details
        txt_baseline:   [int] refer to get_baseline function for details, -1 represents removal
        patch_size:     [int] initial patch size for mask generation, will be resized to `img_size`
        grid_num:       [int] steps for the Riemann sum over proxy path to approximate the integrand
        truncation:     [float] **experimental feature** to cut off heavy tails of estimation as gamma 
                        approaches 0 or 1, disabled by default value 0. if not sure about its utility
        """
        self.grid_num, self.truncation = grid_num, truncation
        self.verbose = False
        self.batch_size = 64

        # Baseline configurations
        self.img_size, self.txt_size = torch.tensor(img_size), txt_size
        self.img_baseline, self.txt_baseline = ImageBaseline(img_baseline), TextBaseline(txt_baseline)
        self.last_valid_txt_length = -1

        log('GEFI initializing query masks, might take some time')
        self.img_masks = ImageSampler(
            int(sample_num*img_ratio), target_size=self.img_size, patch_size=patch_size, 
            grid_num=self.grid_num, truncation=self.truncation, 
        ).get_masks()
        self.txt_masks = TextSampler(
            int(sample_num*(1-img_ratio)), self.txt_size, self.grid_num, 
            self.truncation
        ).get_masks()

    # --> The main entrance for explaining 
    def collect_observations(self, m, img, txt, cv_mode=1):
        """
        Args: 
            m:      [CLIP] the to-be-explained model, CLIP in the context of the paper
            img:    [tensor] image part of shape (1, C, H, W) ready for the processing by `m`
            txt:    [tensor] text part of shape (1, L) ready for the processing by `m`
            cv_mode:    [int] indication of the computation of control variate
                        0 -> disable control variate
                        1 -> control variate based on image masks
                        2 -> control variate based on text masks
                        3 -> control variate considering all feature masks
        
        Return
        --------------------
        attr:       [tensor] the attribution map of shape (1, H, W) for the explicand
        enhanced:   [tensor] an visually enhanced version of the attribution in shape (3, H, W) 
                    only returned when the argument `visual_enhanced` is enabled, 
        """
        device = utils.get_device(m)

        self.img_baseline.reset_baseline_status()
        self.txt_baseline.reset_baseline_status()
        with torch.no_grad():
            img_embeddings = self._get_img_embeddings(m, img.to(device))    # Image queries to embeddings
            txt_embeddings = self._get_txt_embeddings(m, txt.to(device))    # Text queries to embeddings

            # Cosine similarities as observations
            observations = img_embeddings @ txt_embeddings.T

        return self._apply_intxn_control_variate(observations, cv_mode).float()

    # Pair-wise interaction with support of multiple anchors, interactions are returned 
    # for each anchor seperately organized in the first dimension of the return Tensor.
    def interaction_anchored_on_text(self, obs, token_idxs):
        """
        Args:
            obs:    [torch.Tensor] observations in shape of (image_budget, text_budget)
            token_idxs: [torch.Tensor] 1-D tensor containing the token indices
        -----
        Returns:
            estimated:  [torch.Tensor] interactions anchored on token_idxs
                        in shape of (N, 224, 224), where N is the number of anchors
        """
        device = utils.get_device(obs)
        num_anchors = len(token_idxs) if isinstance(token_idxs, (torch.Tensor, list)) else 1
        txt_weights = self.txt_masks.weights[:, token_idxs]
        txt_weights = txt_weights.reshape(-1, num_anchors).to(dtype=obs.dtype)
        aggregated_by_txt = torch.matmul(obs, txt_weights.to(device)) # Information aggregation from the text aspect

        img_weights = self.img_masks.weights.to(device)
        estimated = (aggregated_by_txt.T @ img_weights.flatten(1)) / (obs.shape[0] * obs.shape[1])
        estimated = estimated.reshape(num_anchors, *img_weights.shape[-2:]).cpu()
        return estimated

    # Image anchors are passed as a binary mask, with 1 indicating anchored features
    def interaction_anchored_on_image(self, obs, anchor_mask):
        """
        Args:
            obs:    [torch.Tensor] observations in shape of (image_budget, text_budget)
            anchor_mask:    [torch.Tensor] 2-D tensor having the same shape as the input image
                            specifying the anchoring pixels with 1s
        -----
        Returns:
            estimated:  [torch.Tensor] interactions anchored on pixels
                        in shape of (N), where N is the number of valid tokens 
        """
        device = utils.get_device(obs)
        anchors = anchor_mask.reshape(-1,anchor_mask.shape[-2]*anchor_mask.shape[-1]) # one anchor mask each channel
        anchors = anchors.to(device=device, dtype=obs.dtype)
        anchor_sizes = anchors.sum(dim=1)

        present_ratios = []
        for start in range(0, self.img_masks.budget, self.batch_size):
            masks = self.img_masks.masks[start:start + self.batch_size]
            masks = masks.flatten(1).to(device=device, dtype=obs.dtype)
            present_ratios.append((masks @ anchors.T) / anchor_sizes)
        weights = torch.cat(present_ratios, dim=0)
        weights -= self.img_masks.paths.flatten().to(device)[:, None]

        aggregated = obs.T @ weights
        txt_weights = self.txt_masks.weights[:, self.content_positions].to(device)
        estimated = (aggregated.T @ txt_weights) / (obs.shape[0] * obs.shape[1])
        estimated = estimated.cpu()

        return estimated

    def visual_enhancement(self, img, estimated):
        if isinstance(self.img_baseline.baseline, str) and self.img_baseline.baseline == 'gaussian':
            baseline = 0
        else:
            baseline = self.img_baseline.get_baseline(img).to(utils.get_device(img))

        diff = (img - baseline)[0].mean(0).cpu().abs()
        enhanced = estimated*diff
        return enhanced

    def _get_img_embeddings(self, m, img):
        """ Collect embeddings for visual perturbations """
        num = self.img_masks.budget
        device = utils.get_device(m)    # Get the device from the model
        img_embeddings = []
        for idx in tqdm(range(0, num, self.batch_size), disable=not self.verbose):
            batch_masks = self.img_masks.masks[idx:idx+self.batch_size].to(device)
            queries = self._get_img_queries(img, batch_masks)
            img_observations = m.encode_image(queries)
            img_observations = F.normalize(img_observations, dim=-1)
            img_embeddings.append(img_observations)
        return torch.cat(img_embeddings)
    
    def _get_txt_embeddings(self, m, txt):
        """ Collect embeddings for textual perturbations """
        self.text_handler = getattr(m, "text_handler", None)
        device = utils.get_device(m)    # Get the device from the model

        num = self.txt_masks.budget
        txt_embeddings = []
        for idx in tqdm(range(0, num, self.batch_size), disable=not self.verbose):
            batch_masks = self.txt_masks.masks[idx:idx+self.batch_size].to(device)
            queries = self._get_txt_queries(txt, batch_masks)
            txt_observations = m.encode_text(queries)
            txt_observations = F.normalize(txt_observations, dim=-1)
            txt_embeddings.append(txt_observations)
        return torch.cat(txt_embeddings)
        
    # ---> Returning queries as specified by the explicand, baseline, and masks
    def _get_img_queries(self, x, masks):
        baseline = self.img_baseline.get_baseline(x, batch_size=len(masks))
        return torch.add(masks * x, ~masks * baseline)
    
    # ---> Returning queries as specified by the explicand, baseline, and masks
    def _get_txt_queries(self, x, masks, sot_id=SOT_ID, eot_id=EOT_ID):
        if getattr(self, "text_handler", None) is not None:
            # For processing with SigLIP
            txt = x.squeeze(0) if x.dim() == 2 else x
            # self.last_valid_txt_length = len(self.text_handler.content_positions(txt))
            self.content_positions = self.text_handler.content_positions(txt).cpu()
            self.last_valid_txt_length = len(self.content_positions)
            return self.text_handler.build_queries(x, masks)

        # For processing with CLIP
        pad_id = self.txt_baseline.get_baseline(x)

        device = utils.get_device(x)
        if x.dim() == 2:
            x = x.squeeze(0)  # Remove batch dimension if present
        contents = get_contents(x, sot_id, eot_id)
        self.last_valid_txt_length = len(contents)
        self.content_positions = torch.arange(1, len(contents) + 1)

        batch_size = len(masks)
        padded_len = x.size(0)
        content_len = contents.size(0)
        valid_masks = masks[:, 1:content_len+1]

        queries = torch.full((batch_size, padded_len), pad_id,
                             dtype=x.dtype, device=device)
        queries[:, 0] = sot_id

        # Broadcast interior tokens to all masks
        contents_expanded = contents.unsqueeze(0).expand(batch_size, content_len)  # [N, K]

        # New positions of kept tokens after compaction
        row_idxs = torch.arange(batch_size)
        row_idxs = row_idxs.unsqueeze(1).expand(batch_size, content_len).to(device)
        col_idxs = valid_masks.cumsum(dim=1).to(device)
        queries[row_idxs[valid_masks], col_idxs[valid_masks]] = contents_expanded[valid_masks]

        # EOT goes right after the last kept interior token
        eot_new_pos = col_idxs[:, -1] + 1
        queries[torch.arange(batch_size), eot_new_pos] = eot_id
        return queries

    def _apply_intxn_control_variate(self, obs, cv_mode):
        device = utils.get_device(obs)
        if cv_mode == 0:
            return obs
        elif cv_mode == 1:
            hs = self._get_control_variate_img(self.img_masks.masks).to(device)
            paired = torch.vstack([hs, obs.mean(1)])    # aligning with image massk
            def post_process(x):
                return x.unsqueeze(1)
        elif cv_mode == 2:
            hs = self._get_control_variate_txt(self.txt_masks.masks).to(device)
            paired = torch.vstack([hs, obs.mean(0)])    # aligning with text massk
            def post_process(x):
                return x.unsqueeze(0)
        elif cv_mode == 3:
            hs = self._get_control_variate_ensemble(self.img_masks.masks, self.txt_masks.masks).to(device)
            paired = torch.vstack([hs.flatten(), obs.flatten()])    # aligning 2-D matrices
            def post_process(x):
                return x.reshape(obs.shape)
        else:
            log(f'UNKNOWN control variate mode, should be one of {{0,1,2,3}}, get {cv_mode}') 

        cov_mat = torch.cov(paired)
        var_h, cov = cov_mat[0]
        beta = cov / var_h if var_h > 0 else 0
        return obs - post_process(hs * beta)
   
    # ---> h = ∥ϵ∥_1/p - 0.5 
    def _get_control_variate_txt(self, masks):
        # Feature presence in 1-D aligning with image domain
        num_features = masks.shape[-1]
        hs = masks.sum(dim=1) / num_features - 0.5 
        return hs
    
    def _get_control_variate_img(self, masks):
        # Feature presence in 1-D aligning with text domain
        num_features = masks.shape[-2:].numel()
        hs = masks.sum(dim=(1,2,3)) / num_features - 0.5 
        return hs
    
    def _get_control_variate_ensemble(self, img_masks, txt_masks):
        # The ensemble should produce the feature presence ratio with a 2-D matrix
        num_features = img_masks.shape[-2:].numel() + txt_masks.shape[-1]
        hs = (img_masks.sum(dim=(1,2,3)).unsqueeze(1) + txt_masks.sum(dim=1).unsqueeze(0)) / num_features - 0.5
        return hs

    # ---> the optimal beta is empirically estimated based on the collected observation
    def _estimate_beta(self, ys, hs):
        obs = torch.cat([hs, ys])
        cov_mat = torch.cov(obs)
        cov, var_h = cov_mat[0, 1], cov_mat[0, 0]
        return cov/var_h
    
    """ ============== Feature attribution under GEFI framework ============== """
    def attribution_img(self, img, obs, visual_enhanced=True, enable_cv=False):
        estimated = self._gradient_estimate_img(obs, self.img_masks, enable_cv)
        attr = estimated.mean(0).cpu()
        
        baseline = self.img_baseline.get_baseline(img).cpu()
        diff = (img.cpu() - baseline)[0].mean(0).abs()
        enhanced = attr*diff
        return (attr, enhanced) if visual_enhanced else attr
    
    def attribution_txt(self, txt, obs, enable_cv=False):
        valid_len = len(get_contents(txt[0]))
        final_mask = torch.zeros_like(txt[0])
        final_mask[1:valid_len+1] = 1

        estimated = self._gradient_estimate_txt(obs, self.txt_masks, enable_cv)
        return estimated * final_mask
 
    def _gradient_estimate_img(self, obs, masks, enable_cv):
        obs = obs.mean(1).cuda()
        if enable_cv:
            hs = self._get_control_variate_img(masks.masks).cuda()
            beta = self._estimate_beta(obs.unsqueeze(0), hs.unsqueeze(0))    # 1-D vector of the same size
            obs = obs - hs * beta
        estimated = (obs @masks.weights.cuda().flatten(1)) / masks.weights.shape[0]
        return estimated.reshape(*masks.weights.shape[1:]).cpu()

    def _gradient_estimate_txt(self, obs, masks, enable_cv):
        obs = obs.mean(0).cuda()
        if enable_cv:
            hs = self._get_control_variate_txt(masks.masks.cuda())  # 1-D vector corresponds to query budget
            beta = self._estimate_beta(obs.unsqueeze(0), hs.unsqueeze(0))    # 1-D vector of the same size
            obs = obs - hs * beta
        return ((obs @ masks.weights.cuda()) / masks.weights.shape[0]).cpu()
 

# Filtering out content tokens from full token sequence
def get_contents(x, sot_id=49406, eot_id=49407):
    eot_positions = (x == eot_id).nonzero(as_tuple=False)
    if eot_positions.numel() == 0:
        raise ValueError("Input sequence does not contain eot_id.")
    eot_pos = eot_positions[0, 0].item()

    if x[0].item() != sot_id:
        raise ValueError("Input sequence does not start with sot_id.")
    return x[1:eot_pos] 
