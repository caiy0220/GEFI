import warnings

from competitors.fixlip.game_openai import CLIPGame
from competitors.fixlip.game_huggingface import VisionLanguageGame


def is_hf_dual_encoder(model):
    return hasattr(model, "hf_model") or hasattr(model, "vision_model")


def _as_scalar_patch_size(patch_size):
    if isinstance(patch_size, (tuple, list)):
        return patch_size[0]
    return patch_size


def build_vision_language_game(model, processor, input_image, input_text, patch_size, batch_size=64):
    if hasattr(model, "hf_model"):
        model = model.hf_model
    if hasattr(processor, "processor"):
        processor = processor.processor

    if is_hf_dual_encoder(model):
        model_patch_size = _as_scalar_patch_size(model.vision_model.embeddings.patch_size)
        if int(model_patch_size) != int(patch_size):
            warnings.warn(
                "For HuggingFace/SigLIP models, the game uses the model's "
                f"architecture patch_size={model_patch_size} instead of the "
                f"user-provided patch_size={patch_size}.",
                stacklevel=2,
            )
        return VisionLanguageGame(
            model=model,
            processor=processor,
            input_image=input_image,
            input_text=input_text,
            batch_size=batch_size,
        )

    return CLIPGame(
        model=model,
        processor=processor,
        input_image=input_image,
        input_text=input_text,
        patch_size=patch_size,
        batch_size=batch_size,
    )


def text_token_to_player_index(game, token_idx):
    if hasattr(game, "text_player_positions"):
        try:
            local_idx = game.text_player_positions.index(int(token_idx))
        except ValueError:
            return None
        return game.n_players_image + local_idx

    model_type = getattr(game, "model_type", "openai_clip")
    if model_type == "clip" or model_type == "openai_clip":
        local_idx = int(token_idx) - 1
    else:
        local_idx = int(token_idx)

    if local_idx < 0 or local_idx >= game.n_players_text:
        return None
    return game.n_players_image + local_idx
