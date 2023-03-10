from . import _core
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from pathlib import Path as _Path


def _scale_rect(rect, target, max_size, scale_to_max=False):
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

    if scale_to_max:
        sx = max(1., w / mw)
        sy = max(1., h / mh)
        s = min(sx, sy)
    else:
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


@_core.Op.using(_core.I.image, _core.I.path, _core.I.skip, __skip=False)
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
def resize_image(image, size):
    return cv2.resize(image, size)


@_core.Op.using(_core.I.image) >> _core.I.image
def crop_image(image, area, focal_point=None):
    if len(area) == 4:
        x0, y0, x1, y1 = area
    else:
        if focal_point is None:
            focal_point = (50%Rel, 50%Rel)
        h, w, *_ = image.shape
        focal_point = (
            _core.Value.to_abs(focal_point[0], w),
            _core.Value.to_abs(focal_point[1], h),
        )
        x0 = focal_point[0] - area[0] // 2
        x1 = x0 + area[0]
        y0 = focal_point[1] - area[1] // 2
        y1 = y0 + area[1]

    return image[y0:y1, x0:x1]


@_core.Op >> _core.I.bbox
def get_mask_bbox(mask, threshold=None, padding=None):
    if threshold is None:
        threshold = (mask.min() + mask.max()) / 2
    else:
        treshold = _core.Value.to_abs(threshold, mask.min(), mask.max())

    _, mask = cv2.threshold(mask, threshold, 255, 0)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    cur_h, cur_w, *_ = mask.shape
    if padding:
        x = max(0, x - _core.Value.to_abs(padding, cur_w))
        y = max(0, y - _core.Value.to_abs(padding, cur_h))
        w = min(cur_w, w + 2 * _core.Value.to_abs(padding, cur_w))
        h = min(cur_h, h + 2 * _core.Value.to_abs(padding, cur_h))

    return x, y, x + w, y + h


@_core.Op.using(_core.I.image, _core.I.bbox) >> _core.I.bbox
def scale_bbox(image, rect, target, scale_to_max=False):
    max_size = (image.shape[1], image.shape[0])
    return _scale_rect(rect, target, max_size, scale_to_max)


@_core.Op.using(_core.I.image, _core.I.bbox) >> _core.I.image
def draw_rect(image, rect, color=(0, 0, 0), thickness=2):
    image = image.copy()
    left, top, right, bottom = (int(item) for item in rect)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    return image


@_core.Op.using(_core.I.image.shape) >> _core.I.gradient
def create_gradient(shape, exp=2, max_val=1.):
    shape = shape[:2]
    if exp == 0:
        gradient = np.full(shape, max_val)
    else:
        cx = shape[1] // 2
        cy = shape[0] // 2
        f = lambda y, x:  -abs(y - cy) ** exp - abs(x - cx) ** exp
        gradient = np.fromfunction(f, shape)
        gradient -= gradient.min()
        gradient *= max_val / gradient.max()
    return gradient


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
    from matplotlib import pyplot as plt, colors
    from PIL import Image
    import io

    div = math.ceil(math.sqrt(len(images)))
    for index, im in enumerate(images):
        if im is None:
            continue

        if len(im.shape) > 2:
            channels = im.shape[2]
            if channels == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            elif channels == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        plt.subplot(div, div, index + 1, frameon=False, aspect='equal')
        if len(im.shape) == 2:
            plt.imshow(im, cmap='coolwarm', norm=colors.TwoSlopeNorm(vcenter=0.))
        else:
            plt.imshow(im)
        plt.axis('off')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    np_array = np.array(Image.open(buffer))
    return  cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
