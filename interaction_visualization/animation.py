"""Compose interaction animation frames without redrawing Matplotlib text."""

import numpy as np
from time import perf_counter
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import base64
import json
from io import BytesIO
from uuid import uuid4
from IPython.display import HTML, display



def _region_palette(frames, indices, box, colors, reserved, thumbnail_size=None):
    """Build a regional palette, repeating its last entry in unused slots."""
    samples = []
    for index in indices:
        sample = frames[index].crop(box)
        if thumbnail_size is not None:
            sample.thumbnail(thumbnail_size, Image.Resampling.BILINEAR)
        samples.append(sample)
    width, height = samples[0].size
    sheet = Image.new("RGB", (width, height * len(samples)))
    for row, sample in enumerate(samples):
        sheet.paste(sample, (0, row * height))
    adaptive = colors - len(reserved) // 3
    palette = sheet.quantize(colors=adaptive, method=Image.Quantize.MEDIANCUT)
    entries = palette.getpalette()[:adaptive * 3] + list(reserved)
    entries += entries[-3:] * (256 - colors)
    palette.putpalette(entries)
    return palette


def save_gif_shared_palette(frames, path, durations, *, loop=None, key_colors=(),
                            caption_top=None, image_colors=192, dither_image=True):
    """Quantize once from sampled frames, then map every frame to that palette.

    A shared palette avoids repeated adaptive color searches during saving and
    keeps unchanged colors stable between frames. key_colors are RGB floats.
    With caption_top, reserve separate image/caption palettes (192/64 by default),
    dither only the image, and map the full-resolution caption without dithering.
    """
    started = perf_counter()
    indices = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
    if caption_top is not None:
        width, height = frames[0].size
        image_box = (0, 0, width, caption_top)
        caption_box = (0, caption_top, width, height)
        reserved = [0, 0, 0, 255, 255, 255] + [
            round(channel * 255) for color in key_colors for channel in color
        ]
        image_palette = _region_palette(
            frames, indices, image_box, image_colors, reserved, (256, 256)
        )
        caption_colors = 256 - image_colors
        caption_palette = _region_palette(
            frames, indices, caption_box, caption_colors, reserved
        )
        combined_palette = (
            image_palette.getpalette()[:image_colors * 3]
            + caption_palette.getpalette()[:caption_colors * 3]
        )
        palette_built = perf_counter()
        indexed_frames = []
        dither = Image.Dither.FLOYDSTEINBERG if dither_image else Image.Dither.NONE
        for frame in frames:
            image = frame.crop(image_box).quantize(palette=image_palette, dither=dither)
            caption = frame.crop(caption_box).quantize(
                palette=caption_palette, dither=Image.Dither.NONE
            )
            # Padded palette entries repeat the last real color; canonicalize
            # their indices before putting both regions into the global palette.
            indices_image = np.minimum(np.asarray(image), image_colors - 1)
            indices_caption = np.minimum(np.asarray(caption), caption_colors - 1)
            indices_caption = indices_caption + image_colors
            indexed = Image.fromarray(np.vstack([indices_image, indices_caption]))
            indexed.putpalette(combined_palette)
            indexed_frames.append(indexed)
        mapped = perf_counter()
        options = {} if loop is None else {"loop": loop}
        indexed_frames[0].save(
            path, format="GIF", save_all=True, append_images=indexed_frames[1:],
            duration=durations, optimize=False, **options,
        )
        saved = perf_counter()
        return {
            "Palette construction": palette_built - started,
            "Palette mapping": mapped - palette_built,
            "GIF encoding": saved - mapped,
        }
    sheet = Image.new("RGB", (160 * len(indices), 160), "white")
    for column, index in enumerate(indices):
        sample = frames[index].copy()
        sample.thumbnail((160, 160), Image.Resampling.BILINEAR)
        sheet.paste(sample, (column * 160, 0))
    reserved = [(0, 0, 0), (255, 255, 255)] + [
        tuple(round(channel * 255) for channel in color) for color in key_colors
    ]
    adaptive_colors = 256 - len(reserved)
    palette = sheet.quantize(colors=adaptive_colors, method=Image.Quantize.MEDIANCUT)
    palette.putpalette(
        palette.getpalette()[:adaptive_colors * 3]
        + [channel for color in reserved for channel in color]
    )
    palette_built = perf_counter()
    indexed_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    mapped = perf_counter()
    options = {} if loop is None else {"loop": loop}
    indexed_frames[0].save(
        path, format="GIF", save_all=True, append_images=indexed_frames[1:],
        duration=durations, optimize=False, **options,
    )
    saved = perf_counter()
    return {
        "Palette construction": palette_built - started,
        "Palette mapping": mapped - palette_built,
        "GIF encoding": saved - mapped,
    }


class _AxesRenderer:
    """Cache the canvas and the mapping from data to cropped pixel coordinates."""

    def __init__(self, ax, image_shape):
        canvas = ax.figure.canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        x0, y0, x1, y1 = np.rint(ax.get_window_extent().extents).astype(int)
        self.background = Image.fromarray(
            rgba[rgba.shape[0]-y1:rgba.shape[0]-y0, x0:x1].copy()
        ).convert("RGB")
        # Map data coordinates into the cropped image's top-left pixel system.
        self.origin = np.array([x0, y1])
        self.transform = ax.transData.frozen()
        height, width = image_shape
        self.image_box = tuple(np.rint(self._pixel_box((0, 0, width, height))).astype(int))
        left, top, right, bottom = self.image_box
        self.image_size = (right - left, bottom - top)
        self.line_width = max(1, round(ax.figure.dpi / 72))
        self.corner_radius = 2 * abs(self.transform.get_matrix()[0, 0])

    def _pixel_box(self, box):
        x, y, width, height = box
        points = self.transform.transform([(x, y), (x + width, y + height)])
        points = (points - self.origin) * [1, -1]
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        return left, top, right, bottom


class TextAnchorRenderer(_AxesRenderer):
    """Rasterize the static axes once, then paste heatmaps and draw the outline."""

    def __init__(self, ax, image_shape, norm, cmap, edgecolor,
                 linewidth=2.0, padding=0.18):
        super().__init__(ax, image_shape)
        self.norm = norm
        self.cmap = cmap
        self.edgecolor = tuple(round(channel * 255) for channel in edgecolor)
        # Line width is in points, like Matplotlib; padding is in data units,
        # matching draw_bbox's round,pad=0.18,rounding_size=2 box style.
        self.line_width = max(1, round(linewidth * ax.figure.dpi / 72))
        self.outline_padding = padding

    def render(self, values, box, opacity):
        """Compose an RGB frame with a scalar heatmap and a moving token outline."""
        rgb = self.cmap(self.norm(values), bytes=True)[..., :3]
        heatmap = Image.fromarray(rgb).resize(self.image_size, Image.Resampling.BILINEAR)
        frame = self.background.copy()
        frame.paste(heatmap, self.image_box[:2])
        draw = ImageDraw.Draw(frame, "RGBA")
        x, y, width, height = box
        pad = self.outline_padding
        outline_box = np.asarray(self._pixel_box(
            (x - pad, y - pad, width + 2 * pad, height + 2 * pad)
        ))
        # Pillow strokes inward; expand by half the width to center the stroke
        # on the padded boundary, as FancyBboxPatch does.
        half_width = self.line_width / 2
        outline_box += np.array([-1, -1, 1, 1]) * half_width
        draw.rounded_rectangle(
            tuple(outline_box), radius=self.corner_radius + half_width,
            outline=(*self.edgecolor, round(255 * opacity)), width=self.line_width,
        )
        return frame


class ImageAnchorRenderer(_AxesRenderer):
    """Cache the RGB image and caption; compose image anchors and token fills."""

    def __init__(self, ax, image, boxinfo, edgecolor, positive_color, negative_color,
                 boundary_width=7, background_alpha=0.4):
        super().__init__(ax, image.shape[:2])
        self.image = np.asarray(image, dtype=np.uint8)
        self.edgecolor = np.rint(np.asarray(edgecolor) * 255).astype(np.uint8)
        self.positive_color = tuple(round(c * 255) for c in positive_color)
        self.negative_color = tuple(round(c * 255) for c in negative_color)
        self.boundary_filter = ImageFilter.MaxFilter(boundary_width // 2 * 2 + 1)
        self.faded_image = np.rint(
            self.image * background_alpha + 255 * (1 - background_alpha)
        ).astype(np.uint8)
        self.caption_top = self.image_box[3]
        self.caption_box = (0, self.caption_top, *self.background.size)
        self.caption = self.background.crop(self.caption_box)
        self.token_boxes = [
            tuple(np.asarray(self._pixel_box(info["bbox_xywh"])) - [0, self.caption_top, 0, self.caption_top])
            for info in boxinfo
        ]

    def render(self, mask, scores, alphas):
        """Compose a moving anchor and signed token highlights beneath cached text."""
        mask = np.asarray(mask, dtype=bool)
        dilated = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).filter(self.boundary_filter)
        ) > 0
        image = self.faded_image.copy()
        image[mask] = self.image[mask]
        image[dilated & ~mask] = self.edgecolor
        image = Image.fromarray(image).resize(self.image_size, Image.Resampling.BILINEAR)

        caption = Image.new("RGB", self.caption.size, "white")
        draw = ImageDraw.Draw(caption, "RGBA")
        for box, score, alpha in zip(self.token_boxes, scores, alphas):
            color = self.positive_color if score > 0 else self.negative_color
            draw.rounded_rectangle(
                box, radius=self.corner_radius, fill=(*color, round(255 * alpha)),
            )
        # The cached black text on white stays above the colored box fills.
        caption = ImageChops.multiply(caption, self.caption)
        frame = self.background.copy()
        frame.paste(image, self.image_box[:2])
        frame.paste(caption, self.caption_box[:2])
        return frame

# Create animation of anchor box, shifting from one token to another
# Return a list of 
def bbox_transition(box_start, box_end, frames=11):
    """
    Returns
    --------------------
    frames: [ndarray] an array of bbox statistics in shape (n, 4), 
                      each row corresponds to bbox location ``xywh`` in each frame,
    t:      [ndarray] an array of relative position between two boxes,
                      can be reused to configure status (e.g., visibility) during transition
    """
    t = np.linspace(0, 1, frames)
    frames = []
    for v0, v1 in zip(box_start, box_end):
        frames.append((1-t)*v0 + t*v1)
    frames = np.vstack(frames).T
    return frames, t

def display_frames(frames, durations, loop=False):
    encoded = []
    for frame in frames:
        buffer = BytesIO()
        frame.save(buffer, format="PNG")
        encoded.append(
            "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        )

    element_id = f"animation_{uuid4().hex}"
    display(HTML(f"""
        <img id="{element_id}" />
        <script>
        (() => {{
            const images = {json.dumps(encoded)};
            const durations = {json.dumps(list(durations))};
            const element = document.getElementById("{element_id}");
            let i = 0;
            function advance() {{
                element.src = images[i];
                const delay = durations[i];
                if (i < images.length - 1 || {json.dumps(loop)}) {{
                    i = (i + 1) % images.length;
                    setTimeout(advance, delay);
                }}
            }}
            advance();
        }})();
        </script>
    """))
