from PIL import Image
import argparse, os, pickle
import torch
import utils
from utils import log
import model_adapter
from competitors import load_explainer
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def main(args):
    args.explainer = args.explainer.lower()
    args.dataset = args.dataset.lower()
    assert 0 < args.gefi_img_ratio < 1, "GEFI image ratio should be strictly between 0 and 1."

    log(f'Loading dual-encoder model: [{args.model}]')
    model_bundle = model_adapter.load_model_bundle(args.model, device=args.device)
    clipmodel, preprocess, tokenizer = model_bundle.model, model_bundle.preprocess, model_bundle.tokenizer

    log(f'Loading dataset: [{args.dataset}]')
    if args.dataset == "mscoco":
        dataset_path = args.coco_path
        ig_path = args.coco_ig_path
    elif args.dataset == "flickr30k":
        dataset_path = args.flickr_path
        ig_path = args.flickr_ig_path
    else:
        raise ValueError("Invalid dataset name. Choose from 'mscoco' or 'flickr30k'.")

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)    # Preprocessed data

    with open(ig_path, "rb") as f:
        ig_results = pickle.load(f) # IG-selected anchor tokens

    log(f'Loading explainer: [{args.explainer}]')
    root_path = args.root_expl_path
    dir_name = args.explainer

    utils.set_seed(args.seed)
    expl = load_explainer.load_explainer(args, clipmodel)
    expl_timer = utils.Timer()

    expl_dir = str(Path(utils.get_file_path(root_path, args.model, args.dataset, dir_name, "*", "*")).parent)
    os.makedirs(expl_dir, exist_ok=True)
    log(f'Derived explanations will be saved to [{expl_dir}]')
    log(f'which will be reused for different evaluation tests\n')

    log(f'==>   Start explaining {len(dataset)} samples from the {args.dataset} dataset...')
    time_expl, time_extract = [], []
    early_stop = -1

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # Extract sample contents
        img = sample["cropped_image"]
        caption = sample["caption"]
        txt = model_bundle.tokenize([caption])

        ig_attr = ig_results[idx]["ig_attr"]
        if model_adapter.is_siglip_model(args.model):
            key_indices = model_adapter.get_key_token_indices(
                caption, ig_attr, tokenizer, tokenized_text=txt, topk=args.num_key_idx
            )
        else:
            key_indices = [token_idx for token_idx, _, _ in ig_attr[:min(args.num_key_idx, len(ig_attr))]]
        if len(key_indices) == 0:
            continue

        assert args.square_img == False, "The cropping process has been done in advance. Should not perform padding here."

        if 'gefi' in args.explainer:
            # Collect explanations - GEFI differs slightly in observation processing pipeline
            img = preprocess(img).unsqueeze(0)

            expl_timer.tictoc('explain')
            obs = expl.collect_observations(clipmodel, img, txt, cv_mode=1)
            time_expl.append(expl_timer.tictoc('explain'))

            expl_timer.tictoc('extract')
            token_heatmaps = expl.interaction_anchored_on_text(obs, key_indices).to(device='cpu', dtype=torch.float16)
        else:
            expl_timer.tictoc('explain')
            scores = expl.compute_interaction_values(clipmodel, preprocess, img, caption)
            time_expl.append(expl_timer.tictoc('explain'))

            expl_timer.tictoc('extract')
            token_heatmaps = [expl.interaction_anchored_on_text(scores, token_idx).to(device='cpu', dtype=torch.float16) for token_idx in key_indices]

        token_heatmaps = {token_idx: token_heatmap.numpy() for token_idx, token_heatmap in zip(key_indices, token_heatmaps)}
        time_extract.append(expl_timer.tictoc('extract'))

        # Save explanations for later evaluation
        expl_path = utils.get_file_path(root_path, args.model, args.dataset, dir_name, idx, sample["filepath"])
        with open(expl_path, 'wb') as f:
            pickle.dump(token_heatmaps, f)

        if early_stop > 0 and idx >= early_stop: break

    avg_time_expl = torch.mean(torch.tensor(time_expl)).item()
    avg_time_extract = torch.mean(torch.tensor(time_extract)).item()
    total_time_each_entry = torch.tensor(time_expl) + torch.tensor(time_extract)
    var_time = torch.std(total_time_each_entry).item()
    with open('./time_cost.txt', 'a') as f:
        msg = f'{args.model}, {args.dataset:>6}, {args.explainer.lower():>9}     -     Time cost: {avg_time_expl:>6.4f} + {avg_time_extract:>6.4f} = {avg_time_expl + avg_time_extract:>6.4f} ± {var_time:>6.4f}\n'
        f.write(msg)


if __name__ == '__main__':

    config_pth = 'configs/explain_eval_config.yaml'
    config = utils.load_config(config_pth)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    ns = argparse.Namespace()
    ns.__dict__.update(config)
    # Support overwriting config file with command line input
    for k in ns.__dict__:
        parser.add_argument('--'+k, type=type(getattr(ns, k)))
    _args = parser.parse_known_args(namespace=ns)[0]

    log(_args)

    main(_args)