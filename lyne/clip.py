from . import _core
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
import numpy as np


_clipseg_transform = None
def _get_clipseg_transform(model_name='CIDAS/clipseg-rd64-refined'):
    global _clipseg_transform
    if _clipseg_transform is None:
        _clipseg_transform = CLIPSegProcessor.from_pretrained(model_name)
    return _clipseg_transform


_clipseg_model = None
def _get_clipseg_model(model_name='CIDAS/clipseg-rd64-refined'):
    global _clipseg_model
    if _clipseg_model is None:
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    return _clipseg_model


def get_attention(transform, model, image, captions):
    inputs = transform(text=captions, images=[image] * len(captions), padding=True, return_tensors='pt')
    logits = model.forward(**inputs).logits
    if len(captions) <= 1:
        logits = logits.unsqueeze(0)

    return logits.detach().numpy()


@_core.Op.using(_core.I.image) >> _core.I.attention
def generate_attention(image, captions):
    if isinstance(captions, str):
        captions = {captions: 1.0}
    elif not isinstance(captions, dict):
        captions = {c: 1.0 for c in captions}

    caption_texts = list(captions.keys())
    masks = get_attention(_get_clipseg_transform(), _get_clipseg_model(), image, caption_texts)

    if len(captions) > 1:
        pos = np.zeros(masks.shape[1:])
        neg = np.zeros(masks.shape[1:])
        for layer, weight in zip(masks, captions.values()):
            pos = np.add(pos, weight * np.maximum(layer, 0.0))
            neg = np.add(neg, weight * np.minimum(layer, 0.0))
        mask = np.where(pos > 0.0, pos, neg)
    else:
        mask = masks[0]

    return cv2.resize(mask, (image.shape[1], image.shape[0]))
