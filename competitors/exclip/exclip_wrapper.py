# Wrapper unifies method outputs for easier processing during evaluation

import torch
import torch.nn.functional as F
import clip
from competitors.exclip.explanation import OpenAIExplainer
from utils import log


class ExCLIPWrapper:
    def __init__(self, clipmodel, device='cuda', N=10, text_layer=None, image_layer=None):
        """
        Args:
            clipmodel: The CLIP model instance (e.g., ViT-B/32) to explain.
            device (str): Compute device ('cuda' or 'cpu').
            N (int): Number of interpolation steps for the Integrated Jacobian approximation.
                     Higher N yields more accurate gradients but requires more memory/computation.
            text_layer (int): The transformer layer index to hook for text (default 11 for ViT-B).
            image_layer (int): The transformer layer index to hook for image (default 11 for ViT-B).
        """
        # patch_size cannot be customized as it is determined by the architecture of the CLIP model
        self.device = device
        self.N = int(N)
        self.image_size = clipmodel.visual.input_resolution

        # Initialize the exCLIP explainer for OpenAI's CLIP model
        self.explainer = OpenAIExplainer(clipmodel, device=self.device)

        # Determine the default layers to hook for text if not provided
        if text_layer is None:
            self.text_layer = len(clipmodel.transformer.resblocks) - 1
        else:
            self.text_layer = text_layer

        # Determine the default layers to hook for image if not provided, with support for both ViT and ResNet-based CLIP models
        if image_layer is None:
            if hasattr(clipmodel.visual, "transformer"):
                self.image_layer = len(clipmodel.visual.transformer.resblocks) - 1
            else:
                self.image_layer = 4 # To support ResNet-based CLIP models, which have 5 residual blocks (0-4)
        else:
            self.image_layer = image_layer
            
        log(f"Initialized ExCLIPWrapper. Text Layer: {self.text_layer}, Image Layer: {self.image_layer}")


    def compute_interaction_values(self, clipmodel, preprocess, img_pil, txt_str):
        """
        Runs the exCLIP explainer to compute 2nd-order attributions for all token-patch pairs.
        --------------------
        Args:
            clipmodel: The CLIP model.
            preprocess: The CLIP image preprocessor.
            img_pil (PIL.Image): The original input image in PIL format.
            txt_str (str): The raw input text string (caption).
        --------------------
        Returns:
            interaction_values (torch.Tensor): A tensor of shape (n_tokens, grid_size, grid_size)
                                               containing the attribution scores.
        """
        # exCLIP expects the fully preprocessed tensors on the target device
        img_input = preprocess(img_pil).unsqueeze(0).to(self.device)
        txt_input = clip.tokenize([txt_str]).to(self.device)
        
        # Dynamically record the image size (usually 224) for later interpolation
        self.image_size = img_input.shape[-1]
        if self.explainer.img_ref.shape[-1] != self.image_size:
            self.explainer.image_dim = self.image_size
            self.explainer.img_ref = self.explainer._make_img_ref()
        
        # Compute explanations for all token-patch interactions.
        # For ViT models, it returns a CPU tensor of shape (n_tokens, grid_size, grid_size).
        log(f"Computing interaction values with N={self.N} interpolation steps...", lvl=0)
        interactions_values = self.explainer.explain(
            txt_input,
            img_input,
            text_layer=self.text_layer,
            image_layer=self.image_layer,
            N=self.N,
            verbose=False
        ) # (n_tokens, n_patches, n_patches)
        
        return interactions_values


    def interaction_anchored_on_text(self, interaction_values, tkn_idx, interp_mode='nearest'):
        """
        Extract the 2nd-order interaction scores between a specific text token 
        and all image patches, reshaping them into a 2D attribution map.
        --------------------
        Args:
            interaction_values (torch.Tensor): The interaction values returned by compute_interaction_values.
            tkn_idx (int): The index of the target text token.
            interp_mode (str): 'nearest', 'bicubic', or 'none'. Determines how the patch-level 
                               scores are resized to the original pixel resolution.
        --------------------
        Returns:
            (torch.Tensor): A 2D attribution map aligned with the requested interpolation mode.
        """
        if tkn_idx >= interaction_values.shape[0]:
            # Fallback if the requested token is out of bounds (e.g., in the padding region)
            grid_size = interaction_values.shape[-1]
            attr_map = torch.zeros((grid_size, grid_size), dtype=torch.float32)
        else:
            # Extract the 2D attribution map for the specific token
            attr_map = interaction_values[tkn_idx].float() # n_tokens includes <SOT> and <EOT>

        attr_map = attr_map.unsqueeze(0).unsqueeze(0) # [1, 1, grid_size, grid_size]
        
        if interp_mode == 'nearest':
            # Pure patch-level visualization without smoothing
            attr_map = F.interpolate(attr_map, size=(self.image_size, self.image_size), mode='nearest')
        elif interp_mode == 'bicubic':
            # Smooth interpolation from patch-level to pixel-level
            attr_map = F.interpolate(attr_map, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False)
        elif interp_mode == 'none':
            # To obtain the raw patch-level attribution map without any interpolation
            # Not for visualization, but for direct analysis at the patch level
            pass
        else:
            raise ValueError(f"Unsupported interpolation mode: {interp_mode}")

        return attr_map.squeeze(0).squeeze(0)


    def interaction_anchored_on_image(self, interaction_values, patch_idx):
        """
        Extracts the 2nd-order interaction scores between a specific image patch 
        and all text tokens, returning a 1D attribution array for the text.
        --------------------
        Args:
            interaction_values (torch.Tensor): The pre-computed attributions tensor of shape 
                                               (n_tokens, grid_size, grid_size) from compute_interaction_values.
            patch_idx (int or tuple): The index of the target image patch. If an integer is provided, 
                                      it is treated as the flattened 1D index. If a tuple (row, col) 
                                      is provided, it represents the 2D grid position.
        --------------------
        Returns:
            torch.Tensor: A 1D tensor of shape (n_tokens,) containing the attribution 
                          scores for each text token.
        """
        grid_size = interaction_values.shape[-1]
        
        if isinstance(patch_idx, tuple):
            row, col = patch_idx
        else:
            row = patch_idx // grid_size
            col = patch_idx % grid_size
            
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            raise ValueError(f"patch_idx {patch_idx} (row={row}, col={col}) is out of bounds for grid size {grid_size}x{grid_size}.")
            
        attr_scores = interaction_values[:, row, col]
        
        return attr_scores


    def visual_enhancement(self, img_tensor, attr_map):
        """
        Args:
            img_tensor (torch.Tensor): The preprocessed image tensor [1, 3, H, W].
            attr_map (torch.Tensor): The 2D attribution map.
        """
        diff = img_tensor[0].mean(0).cpu().abs()
        enhanced = attr_map * diff
        return enhanced
