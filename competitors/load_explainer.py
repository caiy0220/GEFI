from gefi import gefi
import model_adapter
from processing import txt_related

'''
==================================================================
|                         Load Explainer                         |
==================================================================
'''
def load_explainer(args, m):
    expl_name = args.explainer
    txt_size = getattr(m, "context_length", txt_related.TXT_SIZE)
    image_size = model_adapter.get_image_size(m)
    img_size = (image_size, image_size)
    if expl_name == 'gefi':
        expl = gefi.GEFI4Clip(
            args.budget, img_size, txt_size, 
            img_baseline=args.img_baseline, 
            txt_baseline=args.txt_baseline,
            truncation=args.gefi_truncation,
            img_ratio=args.gefi_img_ratio,
            patch_size=args.gefi_patch_size,
        )
    elif expl_name == 'exclip':
        from competitors.exclip.exclip_wrapper import ExCLIPWrapper

        if getattr(args, "model", "").lower().find("siglip") >= 0:
            raise ValueError("exCLIP is tied to OpenAI CLIP internals and is not supported for SigLIP-2.")
        # Prefer the default setting for the competitors
        # As ther tests were conducted also on CLIP models
        expl = ExCLIPWrapper(m, device=args.device, N=args.exclip_N)
    elif expl_name == 'fixlip':
        from competitors.fixlip.fixlip_wrapper import FIxLIPWrapper
        expl = FIxLIPWrapper(patch_size=args.patch_size, budget=args.budget)
    elif expl_name == 'faithshap':
        from competitors.faith_shap.faith_shap_wrapper import FaithShapWrapper
        expl = FaithShapWrapper(patch_size=args.patch_size, budget=args.budget)
    elif expl_name == 'proxyspex':
        from competitors.proxyspex_wrapper import ProxySPEXWrapper
        expl = ProxySPEXWrapper(patch_size=args.patch_size, budget=args.budget)
    else:
        raise ValueError(f'Unsupported explainer type {args.explainer}')

    return expl