""" 
================================================================
|               Determining Anchor Tokens with IG              |
================================================================
"""
from clip.simple_tokenizer import SimpleTokenizer
from captum.attr import LayerIntegratedGradients
import model_adapter
from processing import txt_related
import functools, torch, sys, pickle
import clip
from tqdm import tqdm
from utils import log
from nltk.corpus import stopwords


def forward_text_ig(model, text, image):
    text_features = model.encode_text(text)
    image_features = model.encode_image(image)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return (image_features * text_features).sum(dim=-1)

def get_stop_words():
    try:
        return stopwords.words('english')
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return stopwords.words("english")

def main():
    # dsname = sys.argv[1]
    # assert dsname in ['mscoco', 'flickr30k'], "Invalid dataset name. Choose from 'mscoco'."
    dsname = 'mscoco'

    src_path = f'./datasets/{dsname}/{dsname}_filtered.pkl'
    dst_path = f'./datasets/{dsname}/{dsname}_filtered_ig.pkl'

    with open(src_path, "rb") as f:
        dataset = pickle.load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 0
    # Filtering out stop words and punctuations
    PUNCTUATIONS = {".", ",", "!", "?", ";", ":", "-", "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}
    IGNORE = set(get_stop_words()) | PUNCTUATIONS

    clipmodel, preprocess = model_adapter.load_clip(device=DEVICE)
    tokenizer = SimpleTokenizer()


    expl_ig = LayerIntegratedGradients(functools.partial(forward_text_ig, clipmodel), clipmodel.token_embedding)

    ig_results = []

    log('=====> Scanning through dataset to determine anchoring tokens <=====')
    for sample_idx, sample in enumerate(tqdm(dataset)):

        img = sample["cropped_image"]
        img = preprocess(img).unsqueeze(0).to(DEVICE)
        caption = sample["caption"]
        txt = clip.tokenize([caption]).to(DEVICE)
        text_baseline = torch.zeros_like(txt).to(DEVICE)

        txt_attr, _ = expl_ig.attribute(
            inputs=txt,
            baselines=text_baseline,
            additional_forward_args=(img,),
            return_convergence_delta=True,
            n_steps=10,
        )

        txt_attr = txt_attr.detach().sum(-1)[0].cpu().numpy()
        full_span, decoded_text = txt_related.get_clip_token_span(caption, caption, tokenizer)
        # decoded_text = np.array(decoded_text.lower().split())
        # print(sample["caption"], tokenizer.decode(txt[0].cpu().numpy()), sum(txt[0].cpu().numpy() != 0), sum(np.array(txt_attr) != 0), decoded_text, sep="\n", end="\n\n")

        rank_token = []
        for token_idx in full_span:
            word = tokenizer.decode([txt[0][token_idx].cpu().item()]).lower().strip()
            if word in IGNORE:
                continue
            # token_span, _ = tools.get_clip_token_span(caption, word, tokenizer)
            # rank_token.append([token_span, word, txt_attr[token_span].sum()])
            rank_token.append([token_idx, word, txt_attr[token_idx]])

        rank_token = sorted(
            rank_token,
            key=lambda x: x[-1],
            reverse=True
        )

        ig_results.append({"filepath": sample["filepath"], "ig_attr": rank_token})

    idx = 0
    log('=====> DONE, showing examples of anchoring tokens <=====')
    log(ig_results[idx]["filepath"], end="\n\n")
    log(dataset[idx]["caption"], end="\n\n")
    for token_idx, word, attr in ig_results[idx]["ig_attr"]:
        log(f"{token_idx} / {word}: {attr}")

    with open(dst_path, "wb") as f:
        pickle.dump(ig_results, f)

    print(f"Saved to {dst_path}")

if __name__ == '__main__':
    main()