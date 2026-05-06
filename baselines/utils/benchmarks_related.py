import base64
import os
import re
from io import BytesIO

from utils.config_utils import get_config_value, require_config_value


def encode_pil_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_image_file_to_base64(image_path):
    if 'https' in image_path:
        import requests

        response = requests.get(image_path)
        return base64.b64encode(response.content).decode('utf-8')
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def resolve_embedding_path(cfg, field, benchmark_name, stem):
    roots = require_config_value(cfg, f'baselines.{field}')
    if isinstance(roots, str):
        root = roots
    else:
        root = get_config_value(roots, benchmark_name)
    if not root:
        raise ValueError(f'Baseline requires cfg.baselines.{field}.{benchmark_name}.')
    return os.path.join(str(root), f'{stem}.safetensors')


def allowed_page_indices(benchmark_name, sample, benchmark_cfg, page_count):
    if benchmark_name == 'mmlongbench':
        max_pages = int(require_config_value(benchmark_cfg, 'max_pages'))
        return list(range(min(page_count, max_pages)))
    if benchmark_name == 'longdocurl':
        images = sample.get('images')
        if isinstance(images, str):
            images = [images]
        if not images:
            raise ValueError('LongDocURL retrieval requires sample["images"] to derive the page mask.')

        page_indices = []
        for image_path in images:
            filename = os.path.basename(str(image_path))
            match = re.search(r'_(\d+)\.[^.]+$', filename)
            if not match:
                raise ValueError(f'Cannot parse LongDocURL page index from image path: {image_path}')
            page_index = int(match.group(1))
            if 0 <= page_index < page_count:
                page_indices.append(page_index)
        page_indices = sorted(set(page_indices))
        if not page_indices:
            raise ValueError('LongDocURL image-derived page mask is empty after clipping to embedding page count.')
        return page_indices
    raise ValueError(f'Unsupported benchmark for page mask: {benchmark_name}')
