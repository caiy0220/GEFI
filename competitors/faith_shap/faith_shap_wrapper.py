# Wrapper unifies method outputs for easier processing during evaluation

import torch
import torch.nn.functional as F
import numpy as np
from competitors.game_utils import build_vision_language_game, text_token_to_player_index
from competitors.faith_shap.solver import Solver
from utils import log

class FaithShapWrapper:
    def __init__(self, patch_size=32, budget=1e4, use_lasso=True, use_ridge=False, alpha=1e-3):
        """
        Args:
            patch_size (int): The size of the image patches (e.g., 32 for ViT-B/32). 
                              This determines the granularity of the image players.
            budget (int):  Query budget, the total number of sampled perturbations (forward passes).
            use_lasso (bool): Whether to use Lasso regression for sparse feature selection.
            use_ridge (bool): Whether to use Ridge regression for stable coefficient estimation.
            alpha (float): The regularization strength when using Lasso or Ridge regression.
                           Not used if both use_lasso and use_ridge are False (Linear regression).
        """
        self.patch_size = int(patch_size)
        self.budget = int(budget)
        self.use_lasso = use_lasso
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.game = None


    def compute_interaction_values(self, clipmodel, preprocess, img_pil, txt_str, random_state=None):
        """
        Set up the vision-language game and compute the Faith-Shap interaction values.
        --------------------
        Args:
            clipmodel: The CLIP model.
            preprocess: The CLIP image preprocessor.
            img_pil (PIL.Image): The original input image in PIL format.
            txt_str (str): The raw input text string (caption).
            random_state (int): Random seed.
        --------------------
        Returns:
            explanation_dict (dict): A dictionary mapping interaction tuples (e.g., (0, 1))
                                     to their Faith-Shap scores.
        """
        if random_state is not None:
            np.random.seed(int(random_state))
        
        # Define the cooperative game for CLIP
        # Translate the image patches and text tokens into players in the game.
        log(f"Initializing vision-language game...", lvl=0)
        self.game = build_vision_language_game(
            model=clipmodel,
            processor=preprocess,
            input_image=img_pil,
            input_text=txt_str,
            patch_size=self.patch_size,
            batch_size=64
        )
        
        # Generate random binary perturbations (coalitions)
        # Uniform binary sampling simulates the Shapley/Banzhaf perturbation space
        X = np.random.binomial(1, 0.5, size=(self.budget, self.game.n_players_image + self.game.n_players_text))
        
        # Evaluate the CLIP model on the sampled perturbations
        log(f"Evaluating the game value function on {self.budget} perturbations...", lvl=0)
        Y = self.game.value_function(X)    # Y: an array of similarity scores of shape (budget,)
        
        # Initialize the Faith-Shap Solver to compute interaction scores up to 2nd order
        log(f"Initializing Faith-Shap solver...", lvl=0)
        solver = Solver(
            X=X, 
            Y=Y, 
            valid_X=None, 
            valid_Y=None, 
            order_of_interaction=2, 
            continuous=False, 
            is_completeness=False
        )
        
        # Compute interaction scores and return (Explanation_Dict, train_infidelity, valid_infidelity)
        # The explanation dictionary keys are integers for 1st-order and tuples for 2nd-order.
        log(f"Computing interaction values with budget={self.budget}...", lvl=0)
        explanation_dict, _, _ = solver.solve_faith_shap(
            lasso=self.use_lasso, 
            ridge=self.use_ridge,
            alpha=self.alpha
        )
        
        return explanation_dict


    def interaction_anchored_on_text(self, explanation_dict, tkn_idx, interp_mode='nearest'):
        """
        Extract the 2nd-order interaction scores between a specific text token 
        and all image patches, reshaping them into a 2D attribution map.
        --------------------
        Args:
            explanation_dict (dict): The dictionary of Faith-Shap scores from compute_interaction_values.
            tkn_idx (int): The index of the target text token.
            interp_mode (str): 'nearest', 'bicubic', or 'none'. Determines how the patch-level 
                               scores are resized to the original pixel resolution.
        --------------------
        Returns:
            (torch.Tensor): A 2D attribution map aligned with the requested interpolation mode.
        """
        txt_player_idx = text_token_to_player_index(self.game, tkn_idx)
        if txt_player_idx is None:
            return torch.zeros((self.game.image_size, self.game.image_size), dtype=torch.float32)
        
        # Initialize an empty 2D grid for the patch-level attribution scores
        grid_size = self.game.grid_size
        attr_map = torch.zeros((grid_size, grid_size), dtype=torch.float32)
        
        # Extract the 2nd-order cross-modal interaction
        for img_player_idx in range(self.game.n_players_image):
            interaction_tuple = tuple(sorted((img_player_idx, txt_player_idx)))
            
            # Fetch the score if it exists, otherwise default to 0.0
            if interaction_tuple in explanation_dict:
                val = explanation_dict[interaction_tuple]
            else:
                val = 0.0

            # Map the 1D image player index back to 2D (row, col) coordinates
            row = img_player_idx // grid_size
            col = img_player_idx % grid_size
            attr_map[row, col] = val
            
        attr_map = attr_map.unsqueeze(0).unsqueeze(0) # [1, 1, grid_size, grid_size]
        
        if interp_mode == 'nearest':
            # Pure patch-level visualization without smoothing
            attr_map = F.interpolate(attr_map, size=(self.game.image_size, self.game.image_size), mode='nearest')
        elif interp_mode == 'bicubic':
            # Smooth interpolation from patch-level to pixel-level
            attr_map = F.interpolate(attr_map, size=(self.game.image_size, self.game.image_size), mode='bicubic', align_corners=False)
        elif interp_mode == 'none':
            # # To obtain the raw patch-level attribution map without any interpolation
            # (For visualization, this assumes patch_size=1 was set during __init__, i.e., grid_size == image_size)
            pass
        else:
            raise ValueError(f"Unsupported interpolation mode: {interp_mode}")
            
        return attr_map.squeeze(0).squeeze(0)


    def interaction_anchored_on_image(self, interaction_values, patch_idx):
        """
        Extracts the 2nd-order interaction scores between a specific image patch 
        and all valid text tokens, returning a 1D attribution array for the text.
        --------------------
        Args:
            interaction_values (dict): The dictionary of Faith-Shap scores from compute_interaction_values.
            patch_idx (int or tuple): The index of the target image patch. If an integer is provided, 
                                      it is treated as the flattened 1D index. If a tuple (row, col) 
                                      is provided, it represents the 2D grid position.
        --------------------
        Returns:
            torch.Tensor: A 1D tensor of shape (n_players_text,) containing the attribution 
                          scores for each valid text token (excluding <BOS> and <EOS>).
        """
        # Handle 2D grid coordinates if a tuple (row, col) is passed
        if isinstance(patch_idx, tuple):
            row, col = patch_idx
            patch_idx = row * self.game.grid_size + col
            
        # Ensure the patch index is within valid bounds
        if patch_idx < 0 or patch_idx >= self.game.n_players_image:
            raise ValueError(f"patch_idx {patch_idx} is out of bounds for {self.game.n_players_image} image players.")
            
        # Initialize a 1D tensor to hold the attribution scores for all valid text tokens
        attr_scores = torch.zeros(self.game.n_players_text, dtype=torch.float32)
        
        # Loop over all text players to extract the 2nd-order cross-modal interaction
        for txt_idx in range(self.game.n_players_text):
            # Map the local text index to the global game player index
            txt_player_idx = self.game.n_players_image + txt_idx
            
            # Faith-Shap solver stores 2nd-order interactions as sorted tuples
            interaction_tuple = tuple(sorted((patch_idx, txt_player_idx)))
            
            # Fetch the interaction score from the dictionary solved by Faith-Shap
            # Use .get() for safe fallback to 0.0 if the interaction was not sampled or was pruned
            val = interaction_values.get(interaction_tuple, 0.0)
                
            # Assign the interaction value to the corresponding text token position
            attr_scores[txt_idx] = val
            
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
