from . import _core
import numpy as np

@_core.Op.using(_core.S) >> _core.S
def progress(stream):
    import tqdm
    for item in tqdm.tqdm(list(stream)):
        yield item


@_core.Op.using(_core.S) >> _core.S
def shuffle(stream):
    import random
    result = list(stream)
    random.shuffle(result)
    return result


@_core.Op >> _core.I.skip
def cond_size(arr, min_size=None, max_size=None):
    if min_size and any(dim < lim for dim, lim in zip(arr.shape, min_size) if lim != -1):
        return 'too_small'
    elif max_size and any(dim > lim for dim, lim in zip(arr.shape, max_size) if lim != -1):
        return 'too_large'
    else:
        return None


@_core.Op.using(_core.S) >> _core.S
def head(stream, n=5):
    for index, item in enumerate(stream):
        if index >= n:
            break
        yield item


@_core.Op.using(_core.S) >> _core.S
def fields(stream, keep=None, drop=None):
    assert bool(keep) != bool(drop), "Must provide either 'keep' or 'remove'"
    if isinstance(keep, str):
        keep = [keep]
    if isinstance(drop, str):
        drop = [drop]

    for item in stream:
        for key in list(item):
            if keep and key not in keep and key != 'skip':
                delattr(item, key)
            elif drop and key in drop:
                delattr(item, key)
        yield item


@_core.Op.using(_core.S) >> _core.S
def purge_skipped(stream):
    for item in stream:
        if not item.skip:
            yield item


@_core.Op
def scale_array(arr, source_range=None, target_range=None, clip=False):
    arr_min = arr.min()
    arr_max = arr.max()
    arr_range = arr_max - arr_min

    src_min, src_max = _core.Value.to_abs(source_range, arr_min, arr_max)
    tgt_min, tgt_max = _core.Value.to_abs(target_range, arr_min, arr_max)
    if src_min is None: src_min = arr_min
    if src_max is None: src_max = arr_max
    if tgt_min is None: tgt_min = arr_min
    if tgt_max is None: tgt_max = arr_max

    assert src_max >= src_min, f'{src_max} is less than {src_min}'
    assert tgt_max >= tgt_min, f'{tgt_max} is less than {tgt_min}'
    arr = tgt_min + (tgt_max - tgt_min) * (arr - src_min) / (src_max - src_min)
    if clip:
        arr = arr.clip(tgt_min, tgt_max)
    return arr


@_core.Op
def clip_array(arr, min_value, max_value):
    arr_min = arr.min()
    arr_max = arr.max()

    min_value = _core.Value.to_abs(min_value, arr_min, arr_max)
    max_value = _core.Value.to_abs(max_value, arr_min, arr_max)

    assert max_value >= min_value, f'{max_value} is less than {min_value}'
    return arr.clip(min_value, max_value)
