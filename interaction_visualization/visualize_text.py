from matplotlib import pyplot as plt
from processing import txt_related
from matplotlib.patches import FancyBboxPatch
import numpy as np

# convert token to raw texts
def decode_clip_tokens(tokenizer, txt):
    tkns = []
    has_spaces = [] # different space signals distinguishing token and subword token
                    # enable the application of different spaces 
    breakdown_tkn = '</w>'
    for id in txt:
        tkn = txt_related.id2token(tokenizer, id.item())
        if breakdown_tkn in tkn:
            tkn = tkn.replace(breakdown_tkn, '')
            has_spaces.append(True) 
        else:
            has_spaces.append(False)
        tkns.append(tkn)
    return tkns, has_spaces

# draw text span
def draw_span(ax, tkn, x, y, fontsize, facecolor='none', edgecolor='none', alpha=1):
    return ax.text(
        x, y, tkn,
        fontsize=fontsize,
        va="top", ha="left",
        alpha=alpha,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=facecolor, edgecolor=edgecolor)
    )

# Recording bbox information for anchor box animation
def artist_bbox_info(ax, artist, renderer, token_index, token):
    bbox_patch = artist.get_bbox_patch()
    display_bbox = bbox_patch.get_window_extent(renderer=renderer)
    data_corners = ax.transData.inverted().transform(display_bbox.get_points())
    x0, x1 = sorted(data_corners[:, 0])
    y0, y1 = sorted(data_corners[:, 1])

    return {
        "token_index": token_index,
        "token": token,
        "text_xy": artist.get_position(),
        "bbox": (x0, y0, x1, y1),
        "bbox_xywh": (x0, y0, x1 - x0, y1 - y0),
        "display_bbox": display_bbox.bounds,
        "artist": artist,
        "bbox_patch": bbox_patch,
    }

# Configuring bbox color to highlight text anchors and interactions
def set_token_bbox_color(bbox_info, facecolor='none', edgecolor='none'):
    patch = bbox_info["bbox_patch"]
    patch.set_facecolor(facecolor)
    patch.set_edgecolor(edgecolor)
    return patch

def prepare_canvas(ax, img_width, img_height, band_height):
    ax.set_xlim(0, img_width), ax.set_ylim(img_height + band_height, 0)
    ax.axis("off")

    band = plt.Rectangle(
        (0, img_height), img_width, band_height,
        facecolor="white", edgecolor="none", alpha=0.95
    )
    ax.add_patch(band)

    fig = ax.figure
    fig.canvas.draw()
    return fig

# Draw texts under the image frame without any color code
# yet, placeholders for bbox are created and returned
def draw_text_only(ax, tokens, has_spaces, img_width, img_height, 
                   band_height=100, fontsize=16, gap_word=10, gap_subword=4, 
                   line_gap=10, x_pad=6, y_pad=6, return_bbox_infos=True):
    fig = prepare_canvas(ax, img_width, img_height, band_height)
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)

    def px_to_data_x(px):
        return px * img_width / ax_bb.width

    def px_to_data_y(px):
        return px * (img_height + band_height) / ax_bb.height

    # Estimate one line height from an invisible probe text
    probe = ax.text(0, 0, "Ag", fontsize=fontsize, va="top", ha="left", alpha=0)
    # fig.canvas.draw()
    line_height = px_to_data_y(probe.get_window_extent(renderer=renderer).height) + line_gap
    probe.remove()

    x = x_pad
    y = img_height + y_pad

    bbox_infos = []
    for i, (tok, has_space) in enumerate(zip(tokens, has_spaces)):
        # Measure token width with the same transparent bbox style as the final text
        probe = draw_span(ax, tok, 0, 0, fontsize, 'none', 'none')
        # fig.canvas.draw()
        token_width = px_to_data_x(probe.get_window_extent(renderer=renderer).width)
        probe.remove()

        gap = gap_word if has_space else gap_subword

        if x + token_width > img_width - x_pad:
            x = x_pad
            y += line_height

        artist = draw_span(ax, tok, x, y, fontsize, 'none', 'none')
        # fig.canvas.draw()
        artist.update_bbox_position_size(renderer)
        bbox_infos.append((artist_bbox_info(ax, artist, renderer, i, tok), artist))
        x += token_width + gap

    if return_bbox_infos:
        # bbox statistics ('xywh') and the corresponding entity ('bbox_patch')
        # are saved in bbox_infos, which is a list of dictionaries
        return ax, bbox_infos

    return ax
