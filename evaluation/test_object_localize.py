import pickle
import torch
import argparse
from tqdm import tqdm
import utils
from utils import log
from matplotlib import pyplot as plt
import numpy as np
import model_adapter
from processing import txt_related
from evaluation import measures

def main(args):
    args.explainer = args.explainer.lower()
    args.dataset = args.dataset.lower()

    tokenizer = model_adapter.load_tokenizer(args.model)

    if args.dataset == "mscoco":
        dataset_path = args.coco_path
        allow_text_span = False
    elif args.dataset == "flickr30k":
        dataset_path = args.flickr_path
        allow_text_span = True
    else:
        raise ValueError("Invalid dataset name. Choose from 'mscoco', 'flickr30k'.")

    if args.masking_mode == "bbox":
        mode = "bbox"
    elif args.masking_mode == "seg":
        if args.dataset == "flickr" or args.dataset == "flickr_s":
            mode = "bbox"
            log("Segmentation masks are not available for Flickr30k. Defaulting to bounding boxes for masking.")
        else:
            mode = "seg"
    else:
        raise ValueError("Invalid masking mode. Choose from 'bbox' or 'seg'.")

    log(f'Loading dataset: [{args.dataset}]')
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    hits = []
    pge = []
    iou = []
    num_queries = 0

    for idx, sample in enumerate(tqdm(dataset)):

        cropped_image = sample["cropped_image"]
        
        if mode == "bbox":
            target_mask_tensor = sample["mask_bbox"].to(args.device)
        elif mode == "seg":
            target_mask_tensor = sample["mask_seg"].to(args.device)

        caption = sample["caption"]
        target = sample["synonym"] if "synonym" in sample else sample["phrase"]
            
        # Get the list of target token indices
        target_indices, _ = txt_related.get_clip_token_span(caption, target, tokenizer)
        if target_indices is None or len(target_indices) == 0:
            raise ValueError(f"Text '{target}' not found in caption: '{caption}'")

        if not allow_text_span and len(target_indices) > 1:
            hits.append(np.nan)
            pge.append(np.nan)
            iou.append(np.nan)
            continue
        
        # Load attributions
        expl_name = args.explainer if args.explainer != 'random' else 'proxyspex'
        expl_path = utils.get_file_path(args.root_expl_path, args.model, args.dataset, expl_name, idx, sample['filepath'])
        interaction_values = pickle.load(open(expl_path, 'rb'))
            
        # Extract and merge heatmaps for all tokens in the key index span
        token_heatmaps = [torch.tensor(interaction_values[token_idx]) for token_idx in target_indices if token_idx in interaction_values]

        if args.explainer == 'random':
            utils.set_seed(idx)
            buff = [torch.rand_like(mat) for mat in token_heatmaps]
            token_heatmaps = buff


        if len(token_heatmaps) == 0:
            hits.append(np.nan)
            pge.append(np.nan)
            iou.append(np.nan)
            continue

        heatmap_avg = torch.stack(token_heatmaps).mean(dim=0).to(args.device)
        resolution = int(heatmap_avg.shape[-1])
        if target_mask_tensor.shape[-2:] != heatmap_avg.shape[-2:]:
            target_mask_tensor = torch.nn.functional.interpolate(
                target_mask_tensor[None, None].float(),
                size=heatmap_avg.shape[-2:],
                mode="nearest",
            )[0, 0].to(args.device)
        
        # Calculate PGA (Point Game Accuracy)
        hits.append(measures.pga_hit(target_mask_tensor, heatmap_avg, args.explainer, resolution, args.patch_size))
            
        # Calculate PGE (Point Game Energy)
        pge.append(measures.pge_compute(target_mask_tensor, heatmap_avg, quantile=args.PGE_quantile, positive_only=args.positive_only))

        # Calculate IoU (Intersection over Union)
        iou.append(measures.iou(target_mask_tensor, heatmap_avg, resolution, args.patch_size))
        
        num_queries += 1

    print("\n")
    log(f"Evaluated Explainer: {args.explainer}")
    log(f"Dataset: {'MSCOCO' if args.dataset == 'mscoco' else 'Flickr30k'}")
    log(f"Evaluation Mode: {'Bounding Box' if mode == 'bbox' else 'Segmentation Mask'}")
    log(f"Total Evaluated Queries: {num_queries}")
    log(f"Point Game Accuracy (PGA): {np.nanmean(hits):.4f}")
    log(f"Intersection over Union (IoU): {np.nanmean(iou):.4f} +- {np.nanstd(iou):.4f}")
    log(f"Point Game Energy (PGE):   {np.nanmean(pge):.4f} +- {np.nanstd(pge):.4f}")


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