from . import _core
from pathlib import Path as _Path
import os
import shutil


@_core.Op.using(_core.I.path) >> _core.S.path
def list_dir(path, extensions=None, recursive=False):
    if extensions is None:
        extensions = ['.*']

    path = _Path(path)
    if path.is_dir():
        glob = path.rglob if recursive else path.glob
        for ext in extensions:
            yield from glob('*' + ext)
    else:
        yield path


@_core.Op.using(_core.I.path) >> _core.I.path
def change_filename(cur_path, new_filename):
    return _Path(cur_path).parent / new_filename


@_core.Op.using(_core.I.path) >> _core.I.path
def change_dir(cur_path, new_dir):
    return _Path(new_dir) / _Path(cur_path).name


@_core.Op.using(_core.I.path) >> _core.I.path
def change_ext(cur_path, new_ext):
    return _Path(cur_path).with_suffix(new_ext)


@_core.Op.using(_core.O.path, _core.O.skip)
def move_skipped(source, reason):
    skip_dir_name = 'skip'
    if reason:
        skip_dir = _Path(source).parent / skip_dir_name
        target = skip_dir / reason / source.name
        os.makedirs(skip_dir, exist_ok=True)
        shutil.move(source, target)
