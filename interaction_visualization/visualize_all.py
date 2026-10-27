import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from interaction_visualization import score_scaling
from interaction_visualization import visualize_image, visualize_text

def visualize_interactions(example, interactions, img, txt, tokenizer, temperature=3,
                           fontsize=10, text_band_height=110, title=''):
    """Lay out plain captions first, then color their existing token boxes."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    positions = tokenizer.content_positions(txt[0]).detach().cpu()
    content_ids = txt[0, positions.to(txt.device)].detach().cpu()
    tokens, has_spaces = visualize_text.decode_clip_tokens(tokenizer, content_ids)
    n = len(interactions)
    if not n:
        raise ValueError("At least one interaction is required.")

    width, height = example["image"].size
    panel_width = 3.5
    panel_height = panel_width * (height + text_band_height) / width
    fig, axes = plt.subplots(1, 2 * n, figsize=(panel_width * 2 * n, panel_height+0.4), squeeze=False)
    axes = axes[0]

    if title is not None:
        fig.suptitle(title, fontsize=13, y=0.98)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.02, 
        top=0.98 if title is None else 0.90, 
        wspace=0.05,
    )

    edgecolor = visualize_image.color_hex2rgb("#32CD32")
    pos_color, neg_color = (205/255, 92/255, 92/255), (65/255, 105/255, 225/255)

    first_heatmap = interactions[0]["enhanced_text_to_image"].detach().cpu().numpy()
    limit = max(float(np.abs(first_heatmap).max()), 1e-8)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    # Keep the paper ordering: image anchors followed by text-anchored heatmaps.
    for i, result in enumerate(interactions):
        annotated = visualize_image.highlight_masked_region(
            img.detach().cpu(), result["mask"].detach().cpu().bool(),
            alpha=0.4, boundary_width=7, edgecolor=edgecolor,
        )
        axes[i].imshow(annotated)
        axes[n + i].imshow(result["enhanced_text_to_image"].detach().cpu().numpy(),
                           cmap="coolwarm", norm=norm)

    # Draw all text once, without interaction colors. Keep the geometry/artists
    # returned by the package so recoloring does not redraw or reposition text.
    token_boxes = []
    for ax in axes:
        _, bbox_infos = visualize_text.draw_text_only(
            ax, tokens, has_spaces, img_width=width, img_height=height,
            band_height=text_band_height, fontsize=fontsize,
            gap_word=6, gap_subword=2, return_bbox_infos=True,
        )
        token_boxes.append(bbox_infos)

    for i, result in enumerate(interactions):
        scores = result["image_to_text"].detach().cpu()
        if len(scores) != len(tokens):
            raise ValueError("Interaction scores do not align with caption tokens.")
        alphas = score_scaling.scores_to_alpha(scores, temperature=temperature)
        # draw_text_only returns (bbox_info, text_artist) pairs.
        for (info, _), score, alpha in zip(token_boxes[i], scores.tolist(), alphas):
            color = pos_color if score >= 0 else neg_color
            visualize_text.set_token_bbox_color(info, facecolor=(*color, alpha), edgecolor="none")

        anchor_positions = set(result["token_indices"].tolist())
        for (info, _), position in zip(token_boxes[n + i], positions.tolist()):
            if position in anchor_positions:
                visualize_text.set_token_bbox_color(info, facecolor="none", edgecolor=edgecolor)

    # Retain box metadata for later recoloring or animation without drawing text again.
    fig.token_bbox_infos = token_boxes
    fig.canvas.draw_idle()
    return fig, axes

