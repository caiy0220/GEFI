import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


OPENAI_CLIP_MODELS = {
    "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
}

SIGLIP2_MODEL_ALIASES = {
    "SigLIP2-B/16": "google/siglip2-base-patch16-224",
    "SigLIP2-L/16": "google/siglip2-large-patch16-256",
}

SIGLIP2_RESERVED_NAMES = {
    "siglip2-b/16",
    "siglip2-l/16",
    "siglip2-base-patch16-224",
    "siglip2-large-patch16-256",
    "google/siglip2-base-patch16-224",
    "google/siglip2-large-patch16-256",
}


def canonical_model_name(name: str) -> str:
    if name in SIGLIP2_MODEL_ALIASES:
        return SIGLIP2_MODEL_ALIASES[name]
    if name.lower() in SIGLIP2_RESERVED_NAMES:
        raise ValueError('Use `SigLIP2-B/16` or `SigLIP2-L/16` as the --model value for SigLIP-2 models.')
    return name


def is_siglip_model(name: str) -> bool:
    return "siglip" in name.lower()


def is_openai_clip_model(name: str) -> bool:
    return name in OPENAI_CLIP_MODELS

def load_clip(name='ViT-B/32', device='cuda'):
    """
    Backward-compatible loader for CLIP-like dual encoders.

    Args:
        name: ViT-B/32, ViT-B/16, RN50, RN50x4, SigLIP2-B/16, SigLIP2-L/16
        device: cuda, cpu
    -----
    Returns:
        model, preprocessing pipeline
    """
    bundle = load_model_bundle(name, device=device)
    return bundle.model, bundle.preprocess

def tokenize_text(tokenizer, texts):
    return tokenizer.tokenize(texts)


class OpenAIClipTokenizerAdapter:
    def __init__(self):
        import clip
        from clip.simple_tokenizer import SimpleTokenizer

        self._clip = clip
        self._tokenizer = SimpleTokenizer()
        self.context_length = 77
        self.pad_token_id = 0
        self.bos_token_id = 49406
        self.eos_token_id = 49407

    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return self._clip.tokenize(list(texts))

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = [int(i) for i in token_ids]
        return self._tokenizer.decode(ids)

    def id_to_token(self, token_id: int) -> str:
        return self._tokenizer.decoder[int(token_id)]

    def content_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.detach().cpu()
        eot_positions = (ids == self.eos_token_id).nonzero(as_tuple=False)
        if eot_positions.numel() == 0:
            raise ValueError("Input sequence does not contain the CLIP EOT token.")
        eot_pos = int(eot_positions[0, 0].item())
        return torch.arange(1, eot_pos, dtype=torch.long, device=input_ids.device)

    def build_queries(self, input_ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)

        content_pos = self.content_positions(input_ids)
        contents = input_ids[content_pos]
        valid_masks = masks[:, content_pos]

        batch_size = len(masks)
        padded_len = input_ids.size(0)
        queries = torch.full(
            (batch_size, padded_len),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        queries[:, 0] = self.bos_token_id

        if contents.numel() > 0:
            contents_expanded = contents.unsqueeze(0).expand(batch_size, -1)
            row_idxs = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(valid_masks)
            col_idxs = valid_masks.cumsum(dim=1).to(device)
            queries[row_idxs[valid_masks], col_idxs[valid_masks]] = contents_expanded[valid_masks]
            eot_new_pos = col_idxs[:, -1] + 1
        else:
            eot_new_pos = torch.ones(batch_size, dtype=torch.long, device=device)

        queries[torch.arange(batch_size, device=device), eot_new_pos] = self.eos_token_id
        return queries

    def conditioned_pair(self, input_ids: torch.Tensor, token_indices) -> torch.Tensor:
        input_ids = input_ids[0] if input_ids.dim() == 2 else input_ids
        token_indices = _as_index_set(token_indices)
        masks = torch.ones((1, input_ids.numel()), dtype=torch.bool, device=input_ids.device)
        for idx in token_indices:
            if 0 <= idx < input_ids.numel():
                masks[:, idx] = False
        manipulated = self.build_queries(input_ids, masks)[0]
        return torch.vstack([input_ids, manipulated])


class HFTokenizerAdapter:
    def __init__(self, tokenizer, context_length: Optional[int] = None):
        self._tokenizer = tokenizer
        self.context_length = _resolve_context_length(tokenizer, context_length)
        self.pad_token_id = _first_not_none(tokenizer.pad_token_id, tokenizer.eos_token_id, 0)
        self.bos_token_id = _normalize_token_id(tokenizer.bos_token_id)
        self.eos_token_id = _normalize_token_id(tokenizer.eos_token_id)
        self.special_token_ids = {
            int(i)
            for i in tokenizer.all_special_ids
            if i is not None
        }

    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self._tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = [int(i) for i in token_ids]
        return self._tokenizer.decode(ids, skip_special_tokens=True).strip()

    def id_to_token(self, token_id: int) -> str:
        return self._tokenizer.convert_ids_to_tokens(int(token_id))

    def content_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.detach().cpu().tolist()
        positions = [
            i for i, token_id in enumerate(ids)
            if token_id != self.pad_token_id and token_id not in self.special_token_ids
        ]
        return torch.tensor(positions, dtype=torch.long, device=input_ids.device)

    def build_queries(self, input_ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)

        content_pos = self.content_positions(input_ids)
        contents = input_ids[content_pos]
        valid_masks = masks[:, content_pos]

        batch_size = len(masks)
        padded_len = input_ids.size(0)
        queries = torch.full(
            (batch_size, padded_len),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )

        insert_at = 0
        if self.bos_token_id is not None:
            queries[:, 0] = int(self.bos_token_id)
            insert_at = 1

        if contents.numel() > 0:
            contents_expanded = contents.unsqueeze(0).expand(batch_size, -1)
            row_idxs = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(valid_masks)
            col_idxs = valid_masks.cumsum(dim=1).to(device) - 1 + insert_at
            queries[row_idxs[valid_masks], col_idxs[valid_masks]] = contents_expanded[valid_masks]
            next_pos = valid_masks.sum(dim=1).to(device) + insert_at
        else:
            next_pos = torch.full((batch_size,), insert_at, dtype=torch.long, device=device)

        if self.eos_token_id is not None:
            next_pos = torch.clamp(next_pos, max=padded_len - 1)
            queries[torch.arange(batch_size, device=device), next_pos] = int(self.eos_token_id)

        return queries

    def conditioned_pair(self, input_ids: torch.Tensor, token_indices) -> torch.Tensor:
        input_ids = input_ids[0] if input_ids.dim() == 2 else input_ids
        token_indices = _as_index_set(token_indices)
        masks = torch.ones((1, input_ids.numel()), dtype=torch.bool, device=input_ids.device)
        for idx in token_indices:
            if 0 <= idx < input_ids.numel():
                masks[:, idx] = False
        manipulated = self.build_queries(input_ids, masks)[0]
        return torch.vstack([input_ids, manipulated])

    def token_span(self, caption: str, target_phrase: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        match = _find_phrase(caption, target_phrase)
        if not match:
            return None, None

        encoded = self._tokenizer(
            caption,
            add_special_tokens=True,
            truncation=True,
            max_length=self.context_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        input_ids = encoded["input_ids"]
        if offsets is None:
            return self._prefix_based_span(caption, target_phrase, input_ids)

        start_char, end_char = match.span()
        positions = []
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if max(start, start_char) < min(end, end_char):
                positions.append(idx)

        if not positions:
            return None, None
        decoded = self.decode([input_ids[i] for i in positions])
        return torch.tensor(positions, dtype=torch.long), decoded

    def _prefix_based_span(self, caption: str, target_phrase: str, full_ids: List[int]):
        match = _find_phrase(caption, target_phrase)
        if not match:
            return None, None

        prefix_ids = self.encode(caption[:match.start()])
        target_ids = self.encode(caption[match.start():match.end()])
        start = _find_subsequence(full_ids, target_ids, min_start=len(prefix_ids))
        if start is None:
            return None, None
        positions = list(range(start, start + len(target_ids)))
        decoded = self.decode([full_ids[i] for i in positions])
        return torch.tensor(positions, dtype=torch.long), decoded


class SigLIP2Adapter(torch.nn.Module):
    def __init__(self, model, tokenizer_adapter: HFTokenizerAdapter):
        super().__init__()
        self.model = model
        self.text_handler = tokenizer_adapter
        self.context_length = tokenizer_adapter.context_length

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return _as_feature_tensor(features)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.text_handler.pad_token_id).to(self.device, dtype=torch.long)
        features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return _as_feature_tensor(features)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def hf_model(self):
        return self.model


class HFImagePreprocess:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    preprocess: object
    tokenizer: object

    def tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        return self.tokenizer.tokenize(texts)

    @property
    def raw_model(self):
        return getattr(self.model, "hf_model", self.model)

    @property
    def raw_processor(self):
        return getattr(self.preprocess, "processor", self.preprocess)


def load_model_bundle(name: str, device: str = "cuda") -> ModelBundle:
    canonical = canonical_model_name(name)
    if is_siglip_model(canonical):
        return _load_siglip2(canonical, device)
    return _load_openai_clip(canonical, device)


def load_tokenizer(name: str):
    canonical = canonical_model_name(name)
    if is_siglip_model(canonical):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("SigLIP-2 tokenization requires the `transformers` package.") from exc

        tokenizer = AutoTokenizer.from_pretrained(canonical)
        return HFTokenizerAdapter(tokenizer)
    return OpenAIClipTokenizerAdapter()


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


def get_context_length(model_or_tokenizer) -> int:
    if hasattr(model_or_tokenizer, "context_length"):
        return int(model_or_tokenizer.context_length)
    if hasattr(model_or_tokenizer, "text_handler"):
        return int(model_or_tokenizer.text_handler.context_length)
    return 77


def get_image_size(model, default: int = 224) -> int:
    if hasattr(model, "visual") and hasattr(model.visual, "input_resolution"):
        return int(model.visual.input_resolution)

    raw_model = getattr(model, "hf_model", model)
    embeddings = getattr(getattr(raw_model, "vision_model", None), "embeddings", None)
    image_size = getattr(embeddings, "image_size", None)
    if image_size is None:
        vision_config = getattr(getattr(raw_model, "config", None), "vision_config", None)
        image_size = getattr(vision_config, "image_size", None)
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    return int(image_size or default)

def _load_openai_clip(name: str, device: str) -> ModelBundle:
    import clip

    model, preprocess = clip.load(name, device=device)
    model.eval()
    tokenizer = OpenAIClipTokenizerAdapter()
    model.text_handler = tokenizer
    model.context_length = tokenizer.context_length
    return ModelBundle(model=model, preprocess=preprocess, tokenizer=tokenizer)


def _load_siglip2(name: str, device: str) -> ModelBundle:
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise ImportError("SigLIP-2 support requires the `transformers` package.") from exc

    model = AutoModel.from_pretrained(name).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer_adapter = HFTokenizerAdapter(
        tokenizer,
        context_length=getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None),
    )
    adapter = SigLIP2Adapter(model, tokenizer_adapter).eval()
    return ModelBundle(
        model=adapter,
        preprocess=HFImagePreprocess(processor),
        tokenizer=tokenizer_adapter,
    )


def _resolve_context_length(tokenizer, context_length: Optional[int]) -> int:
    if context_length is not None:
        return int(context_length)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is not None and model_max_length < 10000:
        return int(model_max_length)
    return 64


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return _normalize_token_id(value)
    return None


def _normalize_token_id(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return int(value[0])
    return int(value)


def _as_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if isinstance(features, (list, tuple)) and len(features) > 0:
        return features[0]
    raise TypeError(f"Cannot extract feature tensor from {type(features)}")


def _as_index_set(token_indices):
    if isinstance(token_indices, torch.Tensor):
        return {int(i) for i in token_indices.flatten().tolist()}
    if isinstance(token_indices, (list, tuple, set)):
        return {int(i) for i in token_indices}
    return {int(token_indices)}


def _find_phrase(caption: str, target_phrase: str):
    words = target_phrase.lower().strip().split()
    escaped_words = [re.escape(w) for w in words]
    if not escaped_words:
        return None
    pattern = r"(?<!\w)" + r"\s+".join(escaped_words) + r"(?!\w)"
    return re.search(pattern, caption.lower())


def _find_subsequence(seq: Sequence[int], target: Sequence[int], min_start: int = 0):
    if not target:
        return None
    for start in range(max(0, min_start), len(seq) - len(target) + 1):
        if list(seq[start:start + len(target)]) == list(target):
            return start
    return None
