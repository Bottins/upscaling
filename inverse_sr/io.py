from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .priors import gradient_magnitude


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def pick_image(image_arg: str | None, dataset_root: Path) -> Path:
    if image_arg:
        candidate = Path(image_arg)
        if candidate.is_file():
            return candidate.resolve()
        if dataset_root.exists():
            matches = [p for p in dataset_root.rglob("*") if p.name.lower() == image_arg.lower()]
            if matches:
                return matches[0].resolve()
        raise FileNotFoundError(f"Immagine non trovata: {image_arg}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")
    images = list_images(dataset_root)
    if not images:
        raise RuntimeError(f"Nessuna immagine trovata in {dataset_root}")
    for image in images:
        if image.name.lower() == "butterfly.png":
            return image.resolve()
    return images[0].resolve()


def load_image(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    Image.fromarray(_to_uint8_image(tensor)).save(path)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _select_zoom_boxes(image: torch.Tensor, crop_size: int, max_boxes: int = 2) -> list[tuple[int, int, int, int]]:
    _, height, width = image.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    score_map = gradient_magnitude(image).sum(dim=0, keepdim=True).unsqueeze(0)
    pooled = torch.nn.functional.avg_pool2d(score_map, kernel_size=(crop_h, crop_w), stride=1)
    heat = pooled[0, 0].clone()

    boxes: list[tuple[int, int, int, int]] = []
    suppression = crop_size // 2
    for _ in range(max_boxes):
        flat_index = int(torch.argmax(heat).item())
        top = flat_index // heat.shape[1]
        left = flat_index % heat.shape[1]
        boxes.append((left, top, crop_w, crop_h))

        y0 = max(0, top - suppression)
        y1 = min(heat.shape[0], top + suppression)
        x0 = max(0, left - suppression)
        x1 = min(heat.shape[1], left + suppression)
        heat[y0:y1, x0:x1] = -1e9
    return boxes


def _draw_history_plot(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    entry: dict,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10,
                           fill=(255, 255, 255), outline=(205, 205, 205), width=2)
    draw.text((x0 + 12, y0 + 10), "Training loss", font=title_font, fill=(25, 25, 25))

    history = entry.get("history") or []
    if not history:
        draw.text((x0 + 12, y0 + 48), "no optimization", font=body_font, fill=(100, 100, 100))
        return

    points = [
        (float(row["epoch"]), float(row["objective"]))
        for row in history
        if "epoch" in row and "objective" in row
    ]
    if len(points) < 2:
        draw.text((x0 + 12, y0 + 48), "history too short", font=body_font, fill=(100, 100, 100))
        return

    start_value = points[0][1]
    end_value = points[-1][1]
    best_value = min(value for _, value in points)
    summary = f"start {start_value:.4f}   end {end_value:.4f}   best {best_value:.4f}"
    draw.text((x0 + 12, y0 + 40), summary, font=body_font, fill=(70, 70, 70))

    plot_x0 = x0 + 42
    plot_y0 = y0 + 72
    plot_x1 = x1 - 16
    plot_y1 = y1 - 26
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=(150, 150, 150), width=2)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=(150, 150, 150), width=2)

    xs = [epoch for epoch, _ in points]
    ys = [value for _, value in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1e-3
        y_max += 1e-3

    pixel_points: list[tuple[float, float]] = []
    for epoch, value in points:
        px = plot_x0 + (epoch - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)
        py = plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)
        pixel_points.append((px, py))
    draw.line(pixel_points, fill=accent, width=3)

    start_x, start_y = pixel_points[0]
    end_x, end_y = pixel_points[-1]
    r = 4
    draw.ellipse((start_x - r, start_y - r, start_x + r, start_y + r), fill=(120, 120, 120))
    draw.ellipse((end_x - r, end_y - r, end_x + r, end_y + r), fill=accent)

    best_index = int(np.argmin(np.asarray(ys)))
    best_x, best_y = pixel_points[best_index]
    draw.ellipse((best_x - r, best_y - r, best_x + r, best_y + r), outline=(34, 139, 34), width=2)

    draw.text((plot_x0 - 6, plot_y0 - 18), f"{y_max:.3f}", font=body_font, fill=(90, 90, 90))
    draw.text((plot_x0 - 6, plot_y1 + 4), f"{y_min:.3f}", font=body_font, fill=(90, 90, 90))
    draw.text((plot_x0, plot_y1 + 4), f"ep {int(x_min)}", font=body_font, fill=(90, 90, 90))
    draw.text((plot_x1 - 48, plot_y1 + 4), f"ep {int(x_max)}", font=body_font, fill=(90, 90, 90))


def save_comparison_strip(entries: list[dict], path: Path) -> None:
    if not entries:
        raise ValueError("Nessuna entry da salvare nel comparison.")

    title_font = _load_font(20)
    body_font = _load_font(16)
    image_h, image_w = entries[0]["image"].shape[-2:]
    crop_size = max(32, min(image_h, image_w) // 4)
    zoom_boxes = _select_zoom_boxes(entries[0]["image"], crop_size=crop_size, max_boxes=2)
    zoom_size = crop_size * 2
    text_w = 210
    pad = 12
    gap = 12
    card_w = image_w + text_w + pad * 3
    crop_gap = 10
    crops_h = zoom_size + 28
    card_h = image_h + crops_h + pad * 3 + crop_gap
    plot_h = 180
    canvas_w = card_w * len(entries) + gap * (len(entries) - 1)
    canvas_h = card_h + gap + plot_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    colors = [(219, 68, 55), (66, 133, 244), (15, 157, 88)]

    for idx, entry in enumerate(entries):
        x0 = idx * (card_w + gap)
        y0 = 0
        x1 = x0 + card_w - 1
        y1 = y0 + card_h - 1
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10,
                               fill=(255, 255, 255), outline=(205, 205, 205), width=2)

        image = Image.fromarray(_to_uint8_image(entry["image"]))
        image_draw = ImageDraw.Draw(image)
        for box_index, (left, top, crop_w, crop_h) in enumerate(zoom_boxes):
            color = colors[box_index % len(colors)]
            image_draw.rectangle(
                (left, top, left + crop_w - 1, top + crop_h - 1),
                outline=color,
                width=3,
            )
        canvas.paste(image, (x0 + pad, y0 + pad))

        text_x = x0 + image_w + pad * 2
        text_y = y0 + pad
        draw.text((text_x, text_y), entry["title"], font=title_font, fill=(25, 25, 25))

        line_y = text_y + 34
        for line in entry.get("lines", []):
            draw.text((text_x, line_y), line, font=body_font, fill=(55, 55, 55))
            line_y += 24

        crop_y = y0 + image_h + pad * 2 + crop_gap
        crop_x = x0 + pad
        for box_index, (left, top, crop_w, crop_h) in enumerate(zoom_boxes):
            color = colors[box_index % len(colors)]
            crop = image.crop((left, top, left + crop_w, top + crop_h))
            crop = crop.resize((zoom_size, zoom_size), resample=BICUBIC)
            canvas.paste(crop, (crop_x, crop_y))
            draw.rectangle(
                (crop_x, crop_y, crop_x + zoom_size - 1, crop_y + zoom_size - 1),
                outline=color,
                width=3,
            )
            draw.text((crop_x, crop_y + zoom_size + 4), f"crop {box_index + 1}",
                      font=body_font, fill=color)
            crop_x += zoom_size + crop_gap

        plot_y0 = card_h + gap
        plot_y1 = plot_y0 + plot_h - 1
        accent = colors[idx % len(colors)]
        _draw_history_plot(
            canvas=canvas,
            draw=draw,
            rect=(x0, plot_y0, x1, plot_y1),
            entry=entry,
            title_font=title_font,
            body_font=body_font,
            accent=accent,
        )

    canvas.save(path)
