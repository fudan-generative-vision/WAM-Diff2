from __future__ import annotations
from typing import Union, IO, List
from pathlib import Path
import json
import posixpath
from urllib.parse import urlsplit, urlunsplit
import os
import pyarrow.parquet as pq
import pyarrow as pa
import io
import lzma
import pickle
import gzip

try:
    import moxing as mox
    _HAS_MOX = True
except Exception:
    mox = None  # type: ignore
    _HAS_MOX = False

PathLike = Union[str, Path]

# ---------- helpers ----------

def _is_remote(p: PathLike) -> bool:
    s = str(p)
    return s.startswith(("obs://", "s3://", "gs://"))

def join(base: PathLike, *parts: PathLike) -> str:
    b = str(base)
    if _is_remote(b):
        scheme, netloc, path, query, frag = urlsplit(b)
        segs = [path] + [str(p) for p in parts]
        new_path = posixpath.join(*[s.lstrip("/") for s in segs if s])
        if not new_path.startswith("/"):
            new_path = "/" + new_path
        return urlunsplit((scheme, netloc, new_path, query, frag))
    return str(Path(b).joinpath(*map(str, parts)))

# ---------- FS ops ----------

def stem(p: PathLike) -> str:
    """
    Return the stem (filename without suffix) for local and remote paths.
    Equivalent to pathlib.Path(...).stem but works with remote storage.
    """
    s = str(p)
    if _is_remote(s):
        # remote: use basename + splitext
        base = basename(s)          # e.g. "file.txt"
        root, _ = posixpath.splitext(base)
        return root
    else:
        return Path(s).stem


def exists(p: PathLike) -> bool:
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return bool(mox.file.exists(str(p)))
    return Path(p).exists()

def isdir(p: PathLike) -> bool:
    """
    Check if path is a directory (local or remote).
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return bool(mox.file.is_directory(str(p)))
    return Path(p).is_dir()

def listdir(p: PathLike) -> List[str]:
    """
    List directory contents for local or obs:// paths.
    Remote returns full paths, local returns names (like os.listdir).
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return mox.file.list_directory(str(p))
    return [x.name for x in Path(p).iterdir()]

def basename(p: PathLike) -> str:
    """
    Return the final component of a path, works for local and obs:// URLs.
    """
    s = str(p)
    if _is_remote(s):
        # Strip trailing slash if present, then take the last segment
        return s.rstrip("/").split("/")[-1]
    return Path(s).name

def split(p: PathLike) -> tuple[str, str]:
    """
    Split path into (head, tail).
    For remote (obs://...), head keeps scheme+netloc+parent, tail is the last component.
    For local, same as os.path.split.
    """
    s = str(p)
    if _is_remote(s):
        # obs://bucket/key/... → ("obs://bucket/key/..", "lastpart")
        scheme, netloc, path, query, frag = urlsplit(s)
        parts = path.rstrip("/").split("/")
        tail = parts[-1] if parts else ""
        head_path = "/".join(parts[:-1])
        if head_path and not head_path.startswith("/"):
            head_path = "/" + head_path
        head = urlunsplit((scheme, netloc, head_path, query, frag))
        return head, tail
    else:
        return str(Path(s).parent), Path(s).name


def splitext(p: PathLike) -> tuple[str, str]:
    """
    Split path into (root, ext).
    Works like os.path.splitext, for both local and remote.
    """
    s = str(p)
    if _is_remote(s):
        base = basename(s)          # last part only
        root, ext = posixpath.splitext(base)
        # Reconstruct full root path (without extension)
        head, _ = split(s)
        return join(head, root), ext
    else:
        root, ext = Path(s).with_suffix("").as_posix(), Path(s).suffix
        return root, ext


def open_file(p: PathLike, mode: str = "r", **kwargs) -> IO:
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return mox.file.File(str(p), mode, **kwargs)  # type: ignore
    return open(p, mode, **kwargs)

def json_load(p: PathLike, **json_kwargs):
    encoding = json_kwargs.pop("encoding", "utf-8")
    with open_file(p, "r", encoding=encoding) as f:
        return json.load(f, **json_kwargs)

def image_open(p: PathLike):
    from PIL import Image
    if str(p).endswith(".parquet"):
        with open_file(p, 'rb') as fp:
            file_data = fp.read()
        table = pq.read_table(pa.py_buffer(file_data)).to_pandas()
        images = {}
        for cam_id in table.columns:
            data = table[cam_id].iloc[0]  # 取第一行的 bytes 数据
            try:
                img = Image.open(io.BytesIO(data))
                img.load()  # 强制加载，释放文件句柄
                images[cam_id] = img
                if cam_id == "cam_1":
                    break
            except Exception as e:
                raise RuntimeError(f"Failed to decode image from camera '{cam_id}' in {p}: {e}")
        return images["cam_1"]

    f = open_file(p, "rb")
    try:
        img = Image.open(f)
        # Force load into memory so we can close the file immediately
        img.load()
        f.close()
        return img
    except Exception:
        f.close()
        raise


def access(p: PathLike, mode: int = os.R_OK) -> bool:
    """
    Check if a path can be accessed with the given mode flags.
    Supported flags: os.R_OK, os.W_OK, os.X_OK (can be combined).

    - Local: delegates to os.access
    - Remote: simulates using mox.file.* capabilities
    """
    s = str(p)
    if _is_remote(s):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")

        # Check existence first
        if not mox.file.exists(s):
            return False

        # Remote storages don't really have POSIX perms.
        # We approximate:
        # - R_OK → can list dir or open file
        # - X_OK → dir is "enterable" (listable)
        # - W_OK → always assume True if object exists (since API allows writes)
        ok = True
        if mode & os.R_OK:
            if mox.file.is_directory(s):
                try:
                    mox.file.list_directory(s)
                except Exception:
                    ok = False
            else:
                try:
                    with mox.file.File(s, "rb") as f:
                        f.read(1)
                except Exception:
                    ok = False
        if mode & os.X_OK:
            if not mox.file.is_directory(s):
                ok = False
        if mode & os.W_OK:
            # Conservative: allow if it exists; remote APIs typically allow write
            ok = ok and True
        return ok

    # local fallback
    return os.access(s, mode)


def isfile(p: PathLike) -> bool:
    """
    Check if path is a regular file (local or remote).
    """
    s = str(p)
    if _is_remote(s):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        # must exist and not be a directory
        return mox.file.exists(s) and not mox.file.is_directory(s)
    return Path(s).is_file()

def makedirs(p: PathLike, exist_ok: bool = True) -> None:
    """
    Create a directory (and parents if needed).
    - Local: same as os.makedirs(..., exist_ok=...)
    - Remote: uses mox.file.make_dirs
    """
    s = str(p)
    if _is_remote(s):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        if mox.file.exists(s):
            if not exist_ok:
                raise FileExistsError(f"Remote path already exists: {s}")
            return
        mox.file.make_dirs(s)
    else:
        Path(s).mkdir(parents=True, exist_ok=exist_ok)

def getsize(p: PathLike) -> int:
    """
    Return the size of a file in bytes (local or remote).
    """
    s = str(p)
    if _is_remote(s):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return int(mox.file.get_size(s))
    return Path(s).stat().st_size


def get_str_path(p: PathLike):
    s = str(p)
    if "obs:/" in s and "obs://" not in s:
        s = s.replace("obs:/", "obs://")
    return s

def load_pkl(p: PathLike, use_lzma=False, use_gz=False):

    s = get_str_path(p)

    if use_lzma:
        if _is_remote(s):
            raw_data = lzma.decompress(mox.file.read(s, binary=True))
            res = pickle.load(io.BytesIO(raw_data))
        else:
            with lzma.open(s, "rb") as f:
                res = pickle.load(f)
    elif use_gz:
        if _is_remote(s):
            s = io.BytesIO(mox.file.read(s, binary=True))
        with gzip.open(s, "rb") as f:
            res = pickle.load(f)
    else:
        if _is_remote(s):
            raw_data =  mox.file.read(s, binary=True)
            res = pickle.load(io.BytesIO(raw_data))
        else:
            res = pickle.load(open(s, "rb"))

    return res