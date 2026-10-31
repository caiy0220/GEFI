import fiftyone as fo
from tqdm import tqdm
from utils import log
import torch, re, pickle

import json
from collections import defaultdict
from processing import img_related
from PIL import Image

# Replace prefix with the storage directory of the dataset
prefix = '.'
name = 'mscoco'

def main():
    try:
        ds = fo.load_dataset(name)
        ds.delete()
    except ValueError:
        'Name not occupied, SKIP'

    log('=====> Create dataset entry <=====')
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=f"{prefix}/datasets/mscoco/val2017",
        labels_path=f"{prefix}/datasets/mscoco/annotations/instances_val2017.json",
        persistent=True,
        name=name,
    )

    ''' Add captions to dataset '''
    captions_json = f"{prefix}/datasets/mscoco/annotations/captions_val2017.json"
    with open(captions_json, "r") as f:
        coco = json.load(f)

    # Map COCO image_id -> filename
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # Collect all captions per filename
    filename_to_captions = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        filename = id_to_filename[image_id]
        filename_to_captions[filename].append(caption)

    # Add captions to FiftyOne samples
    log('=====> Add captions to FiftyOne samples <=====')
    for sample in tqdm(dataset):
        filename = sample.filename
        caps = filename_to_captions.get(filename, [])

        # Store all captions
        sample["captions"] = caps

        # Store one long joined field for easy viewing
        sample["captions_text"] = "\n\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(caps)]
        )

        # Optional: store first caption only
        sample["caption"] = caps[0] if caps else None

        sample.save()

    log('=====> Dataset preparation done <=====')

    """ 
    ================================================================
    |               Filtering Instances by Mask Size               |
    ================================================================
    """
    log('=====> Now filtering instances by minimum mask size <=====')
    with open('./configs/coco_synonyms.json') as f:
        COCO_SYNONYMS = json.load(f)

    RESOLUTION = 224
    MIN_BBOX_SIZE = 32
    dataset = fo.load_dataset(name)

    def get_synonyms(label):
        # Categories not explicitly listed above will default to their own name (lowercased)
        # (use .get(key, [key]) for fallback handling)
        return COCO_SYNONYMS.get(label, [label.lower()])

    def get_bbox_seg_masks(label, dets, segs, captions, img_height, img_width, 
                        scales, shifts, resolution):

        synonyms_list = get_synonyms(label)
        det_mask = torch.zeros((img_height, img_width), dtype=torch.int8)
        seg_mask = torch.zeros((img_height, img_width), dtype=torch.bool)

        # Iterate through all captions of the image
        for caption in captions:
            cap_lower = caption.lower()

            # Check if any synonym of the label appears in the caption
            for syn in synonyms_list:

                # Use regex to ensure we match whole words only
                if not re.search(r'\b' + re.escape(syn) + r'\b', cap_lower):
                    continue
                
                any_valid_bbox = False

                for det, seg in zip(dets, segs):
                    left, top, width, height = det.bounding_box 

                    abs_bbox = [left * img_width, top * img_height, width * img_width, height * img_height]
                    abs_bbox = [round(v) for v in abs_bbox]

                    if img_related.truncated_bbox(abs_bbox, scales, shifts, resolution, MIN_BBOX_SIZE):
                        continue

                    any_valid_bbox = True

                    det_mask = img_related.mask_bbox(det_mask, abs_bbox)

                    seg_mh, seg_mw = seg.mask.shape
                    seg_left, seg_top = max(0, min(abs_bbox[0], img_width-seg_mw)), max(0, min(abs_bbox[1], img_height-seg_mh))
                    seg_right, seg_bottom = seg_left + seg_mw, seg_top + seg_mh
                    buff = torch.zeros((img_height, img_width), dtype=torch.bool)
                    buff[seg_top:seg_bottom, seg_left:seg_right] = torch.tensor(seg.mask, dtype=torch.bool) 
                    seg_mask = seg_mask + buff
                
                # Convert to Boolean for storage efficiency
                det_mask = (img_related.tensor_crop(det_mask, resolution) > 63).to(dtype=torch.bool)
                seg_mask = img_related.tensor_crop(seg_mask, resolution).to(dtype=torch.bool)
                
                if any_valid_bbox:
                    return det_mask, seg_mask, syn, caption
                else:
                    return None, None, None, None

        return None, None, None, None

    filtered_data = []
    # `allow_repeat=True` => Merge annotations of different entities corresponding
    #                        to the same object
    allow_repeat = False 
    # Consider only the first caption for coco
    # Consider all captions when `num_captions=5`
    num_captions = 1    

    for index, sample in enumerate(tqdm(dataset)):
        if not sample.has_field("detections") or sample.detections is None:
            continue
        if not sample.has_field("segmentations") or sample.segmentations is None:
            continue
        if not sample.has_field("captions") or not sample["captions"]:
            continue
        
        # Count the occurrences of each object label in the image
        label_counts = {}
        for det in sample.detections.detections:
            label_counts[det.label] = label_counts.get(det.label, 0) + 1
            
        if allow_repeat:
            labels = [label for label, _ in label_counts.items()]
        else:
            # Find labels that appear only once in the image
            labels = [label for label, count in label_counts.items() if count == 1]

        # Preparing image with cropping
        raw_image = Image.open(sample.filepath).convert("RGB")
        img_width, img_height = raw_image.size  # The original image size, relevant to pre-processing of bbox and seg

        cropped_image, scales, shifts = img_related.image_crop(raw_image, RESOLUTION)
        
        # Check if any of the synonyms for these labels appear in the captions
        for label in labels:
            # Obtain the detections and segmentations corresponding to this label
            # Contains only one element when 'allow_repeat=False'
            dets = (d for d in sample.detections.detections if d.label == label)
            segs = (s for s in sample.segmentations.detections if s.label == label)

            if dets is None or segs is None:
                continue

            # Get bounding-box by taking union of all bboxes
            det_mask, seg_mask, syn, caption = get_bbox_seg_masks(
                label, dets, segs, sample['captions'][:num_captions],
                img_height, img_width, scales, shifts, RESOLUTION
            )
            
            if det_mask is None:
                continue    # No synonym match found

            filtered_data.append({
                "index": index,                 # Index of the image sample in the original dataset
                "filepath": sample.filepath,    # Add filepath to allow tracing back to images
                "cropped_image": cropped_image, # The center cropped image
                "label": label,                 # The original object label
                "synonym": syn,                 # The specific synonym that matched in the caption
                "caption": caption,             # The corresponding real caption
                "mask_bbox": det_mask,          # Binary mask for bounding box
                "mask_seg": seg_mask,           # Binary mask for segmentation
            })

    log(f"=====> Found {len(filtered_data)} valid instances. <=====")
    path = "./datasets/mscoco/mscoco_filtered.pkl"    

    with open(path, "wb") as f:
        pickle.dump(filtered_data, f)
        
    log(f"=====> Saved to {path} <=====")

if __name__ == '__main__':
    log("=====> Data preparation for COCO <=====")
    log("Make sure data exist in folders: ")
    log("   - ./datasets/mscoco/val2017")
    log("   - ./datasets/mscoco/annotations")
    main()