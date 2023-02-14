from . import _core
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from pathlib import Path as _Path


def _scale_rect(rect, target, max_size):
    if len(rect) == 4:
        x0, y0, x1, y1 = rect
    else:
        x0, y0 = rect
        x1, y1 = rect
    w = x1 - x0
    h = y1 - y0
    cx = x0 + w / 2
    cy = y0 + h / 2

    tw, th = target
    mw, mh = max_size

    sx = max(1., w / tw)
    sy = max(1., h / th)
    s = max(sx, sy)

    sx = min(s, mw / tw)
    sy = min(s, mh / th)
    s = min(sx, sy)
    nw = s * tw
    nh = s * th

    nx0 = cx - nw / 2
    ny0 = cy - nh / 2
    nx1 = cx + nw / 2
    ny1 = cy + nh / 2

    if nx0 < 0:
        nx1 += 0 - nx0
        nx0 = 0
    if nx1 > mw:
        nx0 -= nx1 - mw
        nx1 = mw
    if ny0 < 0:
        ny1 += 0 - ny0
        ny0 = 0
    if ny1 > mh:
        ny0 -= ny1 - mh
        ny1 = mh

    return (int(nx0), int(ny0), int(nx1), int(ny1))


@_core.Op.using(_core.I.path) >> _core.I.image
def open_image(path):
    return cv2.imread(str(path))


@_core.Op.using(_core.I.image, _core.I.path)
def save_image(image, path, overwrite=False):
    path = _Path(path)
    if overwrite or not path.is_file():
        os.makedirs(path.parent, exist_ok=True)
        cv2.imwrite(str(path), image)


@_core.Op.using(_core.O.image, _core.O.path, _core.O.skip)
def save_skipped_image(image, path, reason):
    if reason:
        path = path.parent / 'skip' / reason / path.name
        os.makedirs(path.parent, exist_ok=True)
        cv2.imwrite(str(path), image)


@_core.Op.using(_core.I.image) >> _core.I.focus
def calc_focus(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grayscale, cv2.CV_64F).var()


@_core.Op.using(_core.I.image) >> _core.I.lightness
def calc_lightness(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return cv2.mean(hls)[1] #Get middle channel


@_core.Op.using(_core.I.image) >> _core.I.collage
def calc_collage(im, ignore_padding=0):
    if ignore_padding:
        pad = ignore_padding
        im = im[pad:-pad, pad:-pad]
    im = cv2.Canny(im, 127, 255, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    im = cv2.dilate(im, kernel)

    vlines = abs(np.diff(im.mean(axis=0)))
    hlines = abs(np.diff(im.mean(axis=1)))
    return max(vlines.max(), hlines.max())


@_core.Op.using(_core.I.image)
def show_image(image):
    from matplotlib import pyplot as plt
    from PIL import Image

    if not isinstance(image, Image.Image):
        if len(image.shape) > 2:
            channels = image.shape[2]
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            elif channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.axis('off')
    plt.axis('tight')
    plt.axis('image')
    plt.show()


@_core.Op.using(_core.I.image) >> _core.I.image
def add_alpha_channel(image, alpha):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    alpha = np.uint8(alpha.clip(0, 255))
    alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))
    image[:, :, 3] = alpha
    return image


@_core.Op >> _core.I.image
def create_grid(*images):
    import math
    from matplotlib import pyplot as plt
    from PIL import Image
    import io

    div = math.ceil(math.sqrt(len(images)))
    for index, im in enumerate(images):
        if len(im.shape) > 2:
            channels = im.shape[2]
            if channels == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            elif channels == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        plt.subplot(div, div, index + 1, frameon=False, aspect='equal')
        plt.imshow(im)
        plt.axis('off')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return Image.open(buffer)
