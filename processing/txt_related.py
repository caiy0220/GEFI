import clip, torch, re
import numpy as np
from typing import List, Optional, Tuple
from model_adapter import _find_phrase

TXT_SIZE = 77

class TokenPosInSeq:
    def __init__(self, tok_id, tok_decoded, tok_pos):
        self.id = tok_id
        self.txt = tok_decoded
        self.pos = tok_pos

    def __str__(self):
        return f'token "{self.txt}", ID "{self.id}", index in text "{self.pos}"'

def remove_suffix(tkn, sep_tkn='</w>'):
    if sep_tkn in tkn:
        tkn = tkn[:tkn.index(sep_tkn)]
    return tkn

# Standard pipeline to decode token_id given by CLIP tokenizer
def id2token(tokenizer, tkn_id):
    if hasattr(tokenizer, "id_to_token"):
        return tokenizer.id_to_token(tkn_id)
    return tokenizer.decoder[tkn_id]

# Converting words to tokens with identification of their relative position in given text
def word2token(txt, words, tokenizer, sep_tkn='</w>'):
    '''
    Args:
        txt:    [torch.Tensor] single sentence (in form of tokens) that contains the target words
        words:  [List(Str)] A list of target words to be traced in the given sentence
        tokenizer:  Tokenizer that processed 'txt'
    -----
    Returns:
        tkns:   [List(TokenPosInSeq)] List of tokens that compose all words in 'words'
    '''
    txt = txt[0] if txt.dim() == 2 else txt
    tkns = []
    for w in words:
        for entry in get_pos(txt, w, tokenizer):
            tkn = remove_suffix(entry.txt)
            tkns.append((tkn, entry.pos[0]))
    return tkns

def get_pos(tokens, target_token, tokenizer):
    ids = clip.tokenize([target_token])
    cores = [t for t in ids[0].tolist() if t not in (49406, 49407, 0)]
    matched = []
    for id in cores: 
        decoded = tokenizer.id_to_token(id) if hasattr(tokenizer, "id_to_token") else tokenizer.decoder[id]
        pos = (tokens == id).nonzero().flatten().tolist() # tokens include the <SOT> token at index 0
        matched.append(TokenPosInSeq(id, decoded, pos))
    return matched


def get_token_span(caption: str, target_phrase: str, tokenizer) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if hasattr(tokenizer, "token_span"):
        return tokenizer.token_span(caption, target_phrase)
    return _clip_token_span(caption, target_phrase, tokenizer)


def get_clip_token_span(caption, target_phrase, tokenizer=None):
    """
    Args:
        caption:       [str] The full text caption containing the target phrase.
        target_phrase: [str] The specific phrase in the caption for which we want to find the token indices.
        tokenizer:     The tokenizer used to encode the caption and target phrase (e.g., clip.simple_tokenizer.SimpleTokenizer, without the special tokens like <SOT> and <EOT>).
    -----
    Returns:
        token_indices: [np.ndarray] An array of integers representing the token indices in the tokenized caption (including the <SOT> token at index 0, e.g., the caption tokenized by clip.tokenize).
        decoded_text:  [str] The text decoded from the identified token indices for verification.
    """
    if hasattr(tokenizer, "token_span"):
        token_indices, decoded_text = tokenizer.token_span(caption, target_phrase)
        if token_indices is None:
            return None, None
        return np.array(token_indices.tolist()), decoded_text

    if tokenizer is None:
        import clip
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    else:
        import clip

    words = target_phrase.lower().strip().split()
    escaped_words = [re.escape(w) for w in words]
    r'''
    pattern = r'\b' + r'\s+'.join(escaped_words) + r'\b' fails when the target phrase starts with a non-word character,
    e.g., in 3514685438.txt where the target phrase is " ' Food Basket" rather than "Food Basket" (possibly due to an annotation error in the original dataset).
    '''
    pattern = r'(?<!\w)' + r'\s+'.join(escaped_words) + r'(?!\w)'
    # Find the exact character-level position of the first occurrence of the target_phrase
    match = re.search(pattern, caption.lower())
    if not match:
        return None, None
    start_char_idx = match.start()
    end_char_idx = match.end()
    # Extract the prefix string before the target phrase and the target phrase string itself
    prefix_str = caption[:start_char_idx]
    target_str = caption[start_char_idx:end_char_idx]
    # Use the tokenizer to calculate the number of tokens
    prefix_tokens = tokenizer.encode(prefix_str)
    target_tokens = tokenizer.encode(target_str)
    # Start index = 1 (accounting for the <SOT> token at the beginning) + number of prefix tokens
    start_token_idx = 1 + len(prefix_tokens)
    # End index = Start index + number of tokens the target phrase is split into
    end_token_idx = start_token_idx + len(target_tokens)
    token_indices = np.arange(start_token_idx, end_token_idx)
    # Decode the tokens corresponding to the target phrase for verification
    full_tokens = clip.tokenize([caption])[0]
    matched_token_ids = full_tokens[token_indices].tolist()
    decoded_text = tokenizer.decode(matched_token_ids)
    return token_indices, decoded_text


def get_key_token_indices(
    caption: str,
    ig_attr,
    tokenizer,
    tokenized_text: Optional[torch.Tensor] = None,
    topk: int = 5,
) -> List[int]:
    indices = []
    seen = set()

    for entry in ig_attr:
        _, token_text, _ = entry
        token_text = str(token_text).replace("</w>", "").strip()
        if not token_text:
            continue
        span, _ = get_token_span(caption, token_text, tokenizer)
        if span is None:
            continue
        for pos in span.tolist():
            if pos not in seen:
                indices.append(int(pos))
                seen.add(int(pos))
                break
        if len(indices) >= topk:
            return indices

    if tokenized_text is not None and hasattr(tokenizer, "content_positions"):
        text = tokenized_text[0] if tokenized_text.dim() == 2 else tokenized_text
        for pos in tokenizer.content_positions(text).tolist():
            if pos not in seen:
                indices.append(int(pos))
                seen.add(int(pos))
            if len(indices) >= topk:
                break

    return indices[:topk]


def _clip_token_span(caption: str, target_phrase: str, tokenizer):
    match = _find_phrase(caption, target_phrase)
    if not match:
        return None, None
    prefix_str = caption[:match.start()]
    target_str = caption[match.start():match.end()]
    prefix_tokens = tokenizer.encode(prefix_str)
    target_tokens = tokenizer.encode(target_str)
    start_token_idx = 1 + len(prefix_tokens)
    end_token_idx = start_token_idx + len(target_tokens)
    token_indices = torch.arange(start_token_idx, end_token_idx, dtype=torch.long)
    full_tokens = tokenizer.tokenize([caption])[0]
    matched_token_ids = full_tokens[token_indices].tolist()
    decoded_text = tokenizer.decode(matched_token_ids)
    return token_indices, decoded_text