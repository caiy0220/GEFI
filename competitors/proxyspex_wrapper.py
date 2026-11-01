import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from .game_utils import build_vision_language_game, text_token_to_player_index
import spectralexplain as spex
from spectralexplain.utils import lgboost_to_fourier
from utils import log


def custom_proxy_spex(samples, num_train_samples=None, **kwargs):
    '''
    A monkey-patched version of the original spex.utils.proxy_spex that adjusts
    the energy threshold for sparsification (e.g., preserving 99.9% of the total
    Fourier spectrum energy instead of the default 95%). This allows for retaining
    a denser and more comprehensive set of higher-order Fourier coefficients.
    '''
    THRESHOLD = 0.999    # The energy threshold for the sparsification of Fourier coefficients, the original spex implementation uses 0.95

    if num_train_samples is None:
        num_train_samples = len(samples[0])
    
    max_order = kwargs.get('max_order', 5)
    max_depth_choices = [max(1, max_order - 2), max_order]
    param_grid = {
        'max_depth': sorted(list(set(max_depth_choices))),
        'n_estimators': [500, 1000, 5000],
        'learning_rate': [0.01, 0.1],
    }

    cols = [f"f{i}" for i in range(samples[0].shape[1])]

    train_X = pd.DataFrame(
        samples[0][:num_train_samples],
        columns=cols
    )
    train_y = samples[1][:num_train_samples]
    
    base_model = lgb.LGBMRegressor(
        verbose=-1,
        n_jobs=8,
        random_state=0
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=0,
        n_jobs=1
    )

    grid_search.fit(train_X, train_y)

    best_model = grid_search.best_estimator_
    four_dict = lgboost_to_fourier(best_model)
    
    list_keys = list(four_dict.keys())
    nfc_idx = None
    if tuple([0] * samples[0].shape[1]) in list_keys:
        nfc_idx = list_keys.index(tuple([0] * samples[0].shape[1]))
    four_coefs = np.array(list(four_dict.values()))
    if nfc_idx is not None:
        four_coefs[nfc_idx] = 0
    four_coefs_sq = four_coefs ** 2
    tot_energy = np.sum(four_coefs_sq)
    sorted_four_coefs = np.sort(four_coefs_sq)[::-1]
    thresh_idx = np.argmin(np.cumsum(sorted_four_coefs / tot_energy) < THRESHOLD)
    thresh = np.sqrt(sorted_four_coefs[thresh_idx])
    four_dict_trunc = {k: v for k,v in four_dict.items() if abs(v) > thresh}
    support = np.array(list(four_dict_trunc.keys()))
    
    X = np.real(np.exp(train_X @ (1j * np.pi * support.T)))
    reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

    regression_coefs = {}
    for coef in range(support.shape[0]):
        regression_coefs[tuple(support[coef, :].astype(int))] = reg.coef_[coef]
    return regression_coefs


class ProxySPEXWrapper:
    def __init__(self, patch_size=32, budget=1e4):
        """
        Args:
            patch_size (int): The size of the image patches (e.g., 32 for ViT-B/32). 
            budget (int): The target budget or number of model evaluations. 
                          Note: spectralexplain heavily relies on its sparse Fourier 
                          transform algorithm, so the actual number of evaluations 
                          might be dynamically determined or scaled by this budget.
            random_state (int): Random seed for reproducibility.
        """
        self.patch_size = int(patch_size)
        self.budget = int(budget)
        self.game = None
        spex.utils.proxy_spex = custom_proxy_spex


    def compute_interaction_values(self, clipmodel, preprocess, img_pil, txt_str, random_state=None):
        """
        Set up the vision-language game and compute the ProxySPEX interaction values (Fourier coefficients)
        --------------------
        Args:
            clipmodel: The CLIP model.
            preprocess: The CLIP image preprocessor.
            img_pil (PIL.Image): The original input image in PIL format.
            txt_str (str): The raw input text string (caption).
        --------------------
        Returns:
            explanation_dict (dict): A dictionary mapping interaction tuples (e.g., (0, 1))
                                     to their Fourier coefficient scores.
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
            
        # Initialize the ProxySPEX explainer
        # Compute the raw Fourier coefficients
        log(f"Initializing ProxySPEX explainer and computing Fourier coefficients with budget={self.budget}...", lvl=0)
        explainer = spex.Explainer(
            value_function=self.game.value_function, 
            features=self.game.n_players_image + self.game.n_players_text,
            sample_budget=self.budget,
            max_order=2,
            algorithm="proxyspex"
        )
        
        # Retrieve the Fourier coefficients as interaction values (rather than converting to Shapley/Banzhaf)
        log(f"Retrieving Fourier coefficients as interaction values...", lvl=0)
        interaction_dict = explainer.interactions(index="fourier").get_interactions()
        
        return interaction_dict


    def interaction_anchored_on_image(self, interaction_dict, patch_idx):
        """
        Extract the 2nd-order interaction scores between a specific image patch 
        and all valid text tokens, returning a 1D attribution array for the text.
        --------------------
        Args:
            interaction_dict (dict): The dictionary of Fourier coefficients from compute_interaction_values.
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
        
        # Extract the 2nd-order cross-modal interaction for the specific image patch
        for txt_idx in range(self.game.n_players_text):
            # Map the local text index to the global game player index
            txt_player_idx = self.game.n_players_image + txt_idx
            
            # Formulate the 2nd-order interaction tuple (always sorted in subsets)
            interaction_tuple = tuple(sorted((patch_idx, txt_player_idx)))
            
            # Fetch the interaction score directly from the dictionary (default to 0.0)
            # This safely handles cases where Fourier coefficients were pruned by the sparsity threshold
            val = interaction_dict.get(interaction_tuple, 0.0)
                
            # Assign the interaction value to the corresponding text token position
            attr_scores[txt_idx] = val
            
        return attr_scores


    def interaction_anchored_on_text(self, interaction_dict, tkn_idx, interp_mode='nearest'):
        """
        Extract the 2D attribution map for a specific text token and applies interpolation.
        --------------------
        Args:
            interaction_dict: The dictionary of Fourier coefficients from compute_interaction_values.
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
            if interaction_tuple in interaction_dict:
                val = interaction_dict[interaction_tuple]
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


    def visual_enhancement(self, img_tensor, attr_map):
        """
        Args:
            img_tensor (torch.Tensor): The preprocessed image tensor [1, 3, H, W].
            attr_map (torch.Tensor): The 2D attribution map.
        """
        diff = img_tensor[0].mean(0).cpu().abs()
        enhanced = attr_map * diff
        return enhanced
