from copy import deepcopy

import numpy as np
import torch

from shapiq import Game


class VisionLanguageGame(Game):
    """
    A general interface for Huggingface CLIP, SigLIP
    """
    def __init__(self, model, processor, input_image, input_text, batch_size=1, verbose=False):
        self.model = model
        self.model_type = "clip" 
        model_name = getattr(model, "name_or_path", "")
        if "siglip2" in model_name:
            self.model_type = "siglip2"
        elif "siglip" in model_name:
            self.model_type = "siglip"
        self.processor = processor
        self.input_image = input_image
        self.input_text = input_text
        self.batch_size = batch_size
        self.device = next(self.model.parameters()).device

        self.inputs = self._processor_function(input_image, input_text)

        self.image_size = model.vision_model.embeddings.image_size
        self.patch_size = model.vision_model.embeddings.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]
        self.n_channels = model.vision_model.embeddings.config.num_channels
        self.grid_size = self.image_size // self.patch_size
        self.n_players_image = int(self.image_size / self.patch_size) ** 2 

        self.text_player_positions = self._get_text_player_positions()
        self.n_players_text = len(self.text_player_positions)
        self.text_context_length = self.inputs.input_ids.size(1)

        # get the normalization value
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)

        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
        )


    def _processor_function(self, input_image, input_text):
        """
        Input: list of images of length N, list of texts of length M.
        Output: a dictionary of processed inputs with {'input_ids', 'attention_mask', 'pixel_values'}
        """
        if self.model_type == "siglip" or self.model_type == "siglip2": 
            inputs = self.processor(
                images=input_image, 
                text=input_text, 
                return_tensors="pt", 
                padding="max_length",
                max_length=64
            )
        elif self.model_type == "clip": 
            inputs = self.processor(
                images=input_image, 
                text=input_text, 
                return_tensors="pt", 
                padding=True
            )
        inputs = self._ensure_attention_mask(inputs)
        return inputs


    def value_function(self, coalitions, batch_size=None):
        """ Baseline value function
        Input: Coalitions of the game as a boolean np.array of shape (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions, )."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions = coalitions.shape[0]
        coalitions_image = torch.from_numpy(coalitions[:, :self.n_players_image])
        coalitions_text = torch.from_numpy(coalitions[:, self.n_players_image:])

        text_attention_masks = self._generate_text_attention_mask(coalitions_text, n_coalitions)
        # [n_coallitions, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(coalitions_image)
        # {'input_ids', 'attention_mask', 'pixel_values'}
        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        #:# batch processing
        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            if batch_index < batch_iters:
                inputs = deepcopy(inputs_original)
                inputs['attention_mask'] = text_attention_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]
                inputs['pixel_values'] = inputs['pixel_values'] *\
                      image_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]
            elif batch_left > 0: # process last batch (once)
                inputs = self._processor_function([self.input_image]*batch_left, [self.input_text]*batch_left)
                inputs['attention_mask'] = text_attention_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]
                inputs['pixel_values'] = inputs['pixel_values'] *\
                      image_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]
            else:
                break 
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))
            # take only the diagonal predictions - a naive approach
            outputs = torch.diagonal(outputs.logits_per_image).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)

        return coalitions_outputs.numpy()


    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        """ Efficient value function
        Input: Coalitions of the game as two boolean np.arrays of shapes 
            (n_coalitions_image, n_players_image) and (n_coalitions_text, n_players_text).
        Output: Model outputs for the coalitions of shape (n_coalitions_image, n_coalitions_text)."
        """
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions_image = coalitions_image.shape[0]
        n_coalitions_text = coalitions_text.shape[0]

        text_attention_masks = self._generate_text_attention_mask(
            torch.from_numpy(coalitions_text), n_coalitions_text
        )

        # [n_coalitions_image, n_channels, image_size, image_size]
        image_binary_masks = self._generate_image_binary_mask(torch.from_numpy(coalitions_image))
        # {'input_ids', 'attention_mask', 'pixel_values'}
        inputs_original = self._processor_function([self.input_image]*batch_size, [self.input_text]*batch_size)

        #:# batch processing
        batch_iters_image = n_coalitions_image // batch_size
        batch_iters_text = n_coalitions_text // batch_size
        batch_left_image = n_coalitions_image % batch_size
        batch_left_text = n_coalitions_text % batch_size
        if batch_left_text > 0: # to be copied in (batch_iters_image - 1) iterations
            inputs_left_text = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_left_text)

        coalitions_outputs = []
        for batch_index_image in range(batch_iters_image + 1):
            coalitions_outputs_image = []
            if batch_index_image < batch_iters_image:
                inputs_image = deepcopy(inputs_original)
                inputs_image['pixel_values'] = inputs_image['pixel_values'] *\
                      image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]
            elif batch_left_image > 0: # process last image batch (once)
                inputs_image = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_size)
                inputs_image['pixel_values'] = inputs_image['pixel_values'] *\
                      image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]
            else:
                break 
            for batch_index_text in range(batch_iters_text + 1):
                if batch_index_text < batch_iters_text:
                    inputs = deepcopy(inputs_image)
                    inputs['attention_mask'] = text_attention_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)]
                elif batch_left_text > 0 and batch_index_image < batch_iters_image: # process last text batch in non-terminal image batch
                    inputs = deepcopy(inputs_left_text)
                    inputs['pixel_values'] = inputs['pixel_values'] *\
                        image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]
                    inputs['attention_mask'] = text_attention_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)]                    
                elif batch_left_text > 0 and batch_left_image > 0: # process last text and image batch (once)
                    inputs = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_left_text)
                    inputs['pixel_values'] = inputs['pixel_values'] *\
                        image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]
                    inputs['attention_mask'] = text_attention_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)]                 
                else:
                    break
                with torch.no_grad():
                    outputs = self.model(**inputs.to(self.device))
                outputs = outputs.logits_per_image.cpu()
                coalitions_outputs_image.append(outputs)
            coalitions_outputs.append(torch.concat(coalitions_outputs_image, axis=1))
        coalitions_outputs = torch.concat(coalitions_outputs, axis=0)

        return coalitions_outputs.numpy()
    

    #:# ---------- utility functions ---------- #:#

    def _get_text_player_positions(self):
        input_ids = self.inputs.input_ids[0].detach().cpu()
        attention_mask = self._get_attention_mask(self.inputs)[0].detach().cpu().bool()
        tokenizer = getattr(self.processor, "tokenizer", None)
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        if not special_ids and self.model_type == "clip":
            valid_positions = torch.nonzero(attention_mask, as_tuple=True)[0].tolist()
            return valid_positions[1:-1]
        return [
            idx for idx, token_id in enumerate(input_ids.tolist())
            if attention_mask[idx] and int(token_id) not in special_ids
        ]

    def _generate_text_attention_mask(self, coalitions_text, n_coalitions):
        text_attention_masks = self._get_attention_mask(self.inputs).repeat(n_coalitions, 1).clone()
        text_attention_masks[:, self.text_player_positions] = coalitions_text.to(text_attention_masks.dtype)
        return text_attention_masks.int()

    def _ensure_attention_mask(self, inputs):
        if "attention_mask" not in inputs:
            input_ids = inputs["input_ids"]
            tokenizer = getattr(self.processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = 0
            inputs["attention_mask"] = (input_ids != pad_token_id).long()
        return inputs

    def _get_attention_mask(self, inputs):
        if "attention_mask" in inputs:
            return inputs["attention_mask"]
        return self._ensure_attention_mask(inputs)["attention_mask"]

    def _generate_image_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        # Expand each coalition value into a patch
        binary_masks = coalitions\
            .repeat_interleave(self.patch_size**2, dim=1)\
                .reshape(n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        # Rearrange to form the final batch of full-size images
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, self.image_size, self.image_size)
        # Add image channel dimension
        binary_masks = binary_masks\
            .repeat((self.n_channels, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks
