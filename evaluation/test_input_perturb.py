from pathlib import Path
import random
import argparse, os, pickle
import torch
import utils
from utils import log
import model_adapter
from processing import img_related
from tqdm import tqdm
from torch.nn import functional as F

def conditioned_txt_pair(txt, token_indices, tokenizer=None):
    if tokenizer is not None and hasattr(tokenizer, "conditioned_pair"):
        return tokenizer.conditioned_pair(txt, token_indices)

    txt = txt[0] if txt.dim() == 2 else txt
    keep = torch.ones_like(txt, dtype=torch.bool)
    keep[token_indices] = 0   # locating the token to be removed
    manipulated = txt[keep]
    padding = torch.zeros(len(keep) - sum(keep), dtype=txt.dtype)
    return torch.vstack([txt, torch.concat([manipulated, padding])])

# Feature removal following feature ranking
def mask_out_pix(img, idxs, v_default):
    '''
    Args:
        img:        [torch.Tensor] Input image of shape (B, C, H, W).
                    (Assuming the input to be a single image batch, i.e., B=1.)
        idxs:       [List] 1-D vector of feature indices indicating the ranking
        v_default:  [torch.Tensor] in shape of (C, H, W) => a full image indicating feature-specific baseline
                    [int, float] in single value => constant baseline, e.g. 0
    -----
    Returns:
        img:    [torch.Tensor] image after manipulation, of the same shape as the input
    '''
    rids, cids = idx2pos(torch.tensor(idxs, dtype=torch.long), img.shape[-1])
    v = v_default if isinstance(v_default, (int, float)) else v_default[:, rids, cids]
    img[:, :, rids, cids] = v
    return img

# 1-D indices to 2-D correspond to image shape
def idx2pos(idxs, width=img_related.IMG_SIZE[-1]):
    row_idxs = idxs // width
    col_idxs = idxs % width
    return row_idxs, col_idxs


def intxn_guided_edit_img(model, img_org, intxn, txt_pair, v_default=0, 
                          remove_ratio=0.5, remove_stride=0.01, 
                          pbar=None, deletion='del', random_seed=42):
    """
    Parameters:
        model:         A dual-encoder structure model
        img_org:       [torch.Tensor] Image in shape [1, C, H, W].
        intxn:         [torch.Tensor] Attribution matrix in shape [H, W].
                       [None]  => Baseline of random removal.
        txt_pair:      [torch.Tensor] The text pairs in form of token ids
                       to verify interaction impacts under constrastive conditions.
        v_default:     [torch.Tensor] in shape of (C, H, W) => a full image indicating feature-specific baseline.
                       [int, float] in single value => constant baseline, e.g. 0.
        remove_ratio:  The ratio of features to be removed.
        remove_stride: The stride of recording model reaction.
        pbar:          Showing the manipulation process with nested progress bar.
    """
    img = img_org.clone()
    num_features = img.shape[-2:].numel() 
    total_remove = int(num_features * remove_ratio)
    query_stride = int(num_features * remove_stride)

    # Get feature ranking to guide removal
    if intxn is None:   # Random baseline if nothing provided
        # Random seed allows reproducability of the baseline results
        # The random ranking varies across different test cases, 
        # but it is consistent for a specific input as the seed is determined by its index
        random.seed(random_seed)    
        pix_ranking = utils.random_ranking(num_features)
        pix_ranking = pix_ranking if deletion == 'del' else pix_ranking[::-1]
    else:
        # Only ranking is related, which guides the removal process
        _, pix_ranking = utils.get_feature_ranking(intxn, deletion == 'del')

    queries = [img.clone()]
    for left in range(0, total_remove, query_stride):
        right = min(left + query_stride, total_remove)
        img = mask_out_pix(img, pix_ranking[left:right], v_default)
        queries.append(img.clone())
        if pbar:
            pbar.set_postfix_str('Current edition: [{}/{}]'.format(left, total_remove))
    
    queries = torch.concatenate(queries)
    
    with torch.no_grad():
        img_embeddings = model.encode_image(queries) # [num_queries, D]
        img_embeddings = F.normalize(img_embeddings, dim=-1)
        txt_embeddings = model.encode_text(txt_pair) # [2, D]
        txt_embeddings = F.normalize(txt_embeddings, dim=-1)
        trends = img_embeddings @ txt_embeddings.T # [num_queries, 2]
        trends = trends - trends[0] # Record the changes only

    return trends.T.cpu(), queries.cpu()

def main(args):
    args.explainer = args.explainer.lower()
    args.dataset = args.dataset.lower()

    log(f'Loading dual-encoder model: [{args.model}]')
    model_bundle = model_adapter.load_model_bundle(args.model, device=args.device)
    model, preprocess, tokenizer = model_bundle.model, model_bundle.preprocess, model_bundle.tokenizer

    if args.dataset == "mscoco":
        dataset_path = args.coco_path
        ig_path = args.coco_ig_path
        allow_text_span = False
    elif args.dataset == "flickr30k":
        dataset_path = args.flickr_path
        ig_path = args.flickr_ig_path
        allow_text_span = True
    else:
        raise ValueError("Invalid dataset name. Choose from 'mscoco', 'flickr30k'.")

    log(f'Loading dataset: [{args.dataset}]')
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    with open(ig_path, "rb") as f:
        ig_results = pickle.load(f)

    assert args.eval_mode in ['del', 'ins'], "Invalid evaluation mode. Choose from 'del' or 'ins'."

    root_expl_path = args.root_expl_path
    root_result_path = args.root_result_path
    expl_dir_name = args.explainer

    result_dir_name = f'{expl_dir_name}/trend_{args.eval_mode}'

    expl_dir = str(Path(utils.get_file_path(root_expl_path, args.model, args.dataset, expl_dir_name, "*", "*")).parent)
    if args.explainer != 'random':
        log(f'Explanations are loaded from [{expl_dir}]')

    trd_dir = str(Path(utils.get_file_path(root_result_path, args.model, args.dataset, result_dir_name, "*", "*")).parent)
    os.makedirs(trd_dir, exist_ok=True)
    log(f'Changing trends will be saved to [{trd_dir}]')

    log(f'==>   Mode [{args.eval_mode.upper()}]. Start evaluating {len(dataset)} samples from the {args.dataset} dataset...')
    num_queries = 0
    early_stop = -1

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # Extract sample contents
        img = sample["cropped_image"]
        caption = sample["caption"]

        assert args.square_img == False, "The cropping process has been done in advance. Should not perform padding here."

        img = preprocess(img).unsqueeze(0)
        txt = model_bundle.tokenize([caption])
        
        if args.explainer == "random":
            ig_attr = ig_results[idx]["ig_attr"]
            if model_adapter.is_siglip_model(args.model):
                token_indices = model_adapter.get_key_token_indices(
                    caption, ig_attr, tokenizer, tokenized_text=txt, topk=args.num_key_idx
                )
            else:
                token_indices = [token_idx for token_idx, _, _ in ig_attr[:min(args.num_key_idx, len(ig_attr))]]

        if args.explainer == "random":
            interaction_values = None
        else:
            expl_path = utils.get_file_path(root_expl_path, args.model, args.dataset, expl_dir_name, idx, sample["filepath"])
            interaction_values = pickle.load(open(expl_path, 'rb'))
            token_indices = list(interaction_values.keys())

        txt_pairs = [conditioned_txt_pair(txt, token_idx, tokenizer=tokenizer) for token_idx in token_indices]

        trends = []
        for token_idx, txt_pair in zip(token_indices, txt_pairs):
            intxn = None if interaction_values is None else torch.tensor(interaction_values[token_idx]).to(device=args.device)
            trend_pair, _ = intxn_guided_edit_img(
                model, img.to(device=args.device), intxn,
                txt_pair.to(device=args.device),
                remove_ratio=1.0, remove_stride=0.01,
                deletion=args.eval_mode, random_seed=idx    
            )
            trends.append(trend_pair)

        trends = {token_idx: trend for token_idx, trend in zip(token_indices, trends)}
        
        trd_path = utils.get_file_path(root_result_path, args.model, args.dataset, result_dir_name, idx, sample["filepath"])
        with open(trd_path, 'wb') as f:
            pickle.dump(trends, f)

        num_queries += 1
        if early_stop > 0 and idx >= early_stop: break


if __name__ == '__main__':

    config_pth = 'configs/explain_eval_config.yaml'
    config = utils.load_config(config_pth)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    ns = argparse.Namespace()
    ns.__dict__.update(config)
    for k in ns.__dict__:
        parser.add_argument('--'+k, type=type(getattr(ns, k)))
    _args = parser.parse_known_args(namespace=ns)[0]

    log(_args)

    main(_args)
