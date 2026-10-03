"""Write the outputs of a tool to disk: figures -> PNG / PDF / SVG, tables -> CSV (no UI)."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from .registry import Output

DEFAULT_DIR = str(Path(__file__).resolve().parent.parent / "output")
FORMATS = ("png", "pdf", "svg")


def _token(value) -> str:
    """One parameter value as it appears in a file name."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        value = f"{value:g}"
    return re.sub(r"[^A-Za-z0-9.+]+", "_", str(value)).strip("_") or "x"


def plot_tag(values: dict, defaults: dict, *, skip=("ffid",), flow: list[dict] | None = None, view: int | None = None,
             default_of=None, max_len: int = 120) -> str:
    """The parameters a figure was made with, for its file name: only those that differ from the defaults, as
    `key-value` joined by `_` (`clip_pct-95_agc_ms-200_sort_by-offset`), or `defaults` when nothing was changed.

    values / defaults   {param key: value used} / {param key: the value the widget starts with}
    flow / view         Pipeline: the steps (`{"step": key, "params": {...}}`) and the stage shown
    default_of          optional callable(Param) -> default of a step parameter (header-derived ones)
    Long tags are cut at a whole `key-value` and closed with a short hash, so different settings never share a name.
    """
    def norm(v):                                       # 5000 and 5000.0 are the same pick
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return float(f"{float(v):.10g}") if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    tokens = []
    for key, val in values.items():
        if key in skip or val is None or (isinstance(val, str) and len(val) > 24):
            continue
        base = defaults.get(key)
        if isinstance(val, (list, tuple, dict)):        # a table (velocity function, corner points): a short code
            if norm(val) != norm(base):
                tokens.append(f"{_slug(key)}-" + hashlib.md5(repr(norm(val)).encode()).hexdigest()[:6])
            continue
        same = (abs(float(val) - float(base)) < 1e-9 if isinstance(val, (int, float)) and isinstance(base, (int, float))
                else val == base)
        if not same:
            tokens.append(f"{_slug(key)}-{_token(val)}")
    if flow is not None:
        from .pipeline import get_step
        names = []
        for st in flow:
            names.append(st["step"])
            for p in get_step(st["step"]).params:
                v = (st.get("params") or {}).get(p.key)
                d = default_of(p) if default_of else p.default
                if isinstance(v, (list, tuple, dict)):
                    if norm(v) != norm(d):
                        tokens.append(f"{st['step']}.{_slug(p.key)}-" + hashlib.md5(repr(norm(v)).encode()).hexdigest()[:6])
                    continue
                if v is not None and not (isinstance(v, (int, float)) and isinstance(d, (int, float)) and not isinstance(v, bool)
                                          and abs(float(v) - float(d)) < 1e-9) and v != d:
                    tokens.append(f"{st['step']}.{_slug(p.key)}-{_token(v)}")
        tokens.insert(0, "steps-" + "+".join(names) if names else "steps-none")
        if view is not None:
            tokens.append(f"view-{view}")
    tag = "_".join(tokens) or "defaults"
    if len(tag) > max_len:
        keep = []
        for t in tokens:
            if len("_".join(keep + [t])) > max_len - 9:
                break
            keep.append(t)
        tag = "_".join(keep + [hashlib.md5(tag.encode()).hexdigest()[:8]])
    return tag


def _base(out_dir: Path, stem: str, title: str, tag: str) -> Path:
    """`<folder>/<stem>_<title>_<tag>` - the tag is replaced by its hash when the name would get too long."""
    name = f"{stem}_{_slug(title)}" + (f"_{tag}" if tag else "")
    if len(name) > 235:
        name = f"{stem}_{_slug(title)}"[:200] + "_" + hashlib.md5(tag.encode()).hexdigest()[:8]
    return out_dir / name


def save_data(outputs: list[Output], folder: str, stem: str, *, fmt: str = "ibm", tag: str = "") -> list[str]:
    """Write every "data" output (a processed shot) as a SEG-Y file; returns the paths written."""
    from .segy_write import write_shot_segy

    out_dir = Path(folder).expanduser()
    return [
        write_shot_segy(o.content, f"{_base(out_dir, stem, o.title, tag)}.sgy", fmt)
        for o in outputs if o.kind == "data"
    ]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "output"


def save_outputs(outputs: list[Output], folder: str, stem: str, *, fmt: str = "png", dpi: int = 150,
                 tables: bool = True, tag: str = "") -> list[str]:
    """Save every figure as `<folder>/<stem>_<title>_<tag>.<fmt>`; returns the paths written.

    fmt     png | pdf | svg  (pdf / svg are vector graphics; needs the figure a tool kept)
    dpi     resolution of PNG files
    tables  also write tables as CSV and text results as Markdown
    tag     the parameters the figures were made with, see plot_tag(): different settings -> different files
    """
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}; use one of {FORMATS}")
    out_dir = Path(folder).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for o in outputs:
        base = _base(out_dir, stem, o.title, tag)
        if o.kind == "image":
            path = Path(f"{base}.{fmt}")
            if o.figure is not None:
                o.figure.savefig(path, format=fmt, dpi=dpi, facecolor="white")
            elif fmt == "png":
                path.write_bytes(o.content)
            else:
                continue                                   # a bare PNG cannot become a vector file
        elif not tables:
            continue
        elif o.kind == "table":
            path = Path(f"{base}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=o.columns, extrasaction="ignore")
                w.writeheader()
                w.writerows(o.content)
        elif o.kind == "markdown":
            path = Path(f"{base}.md")
            path.write_text(f"# {o.title}\n\n{o.content}\n" if o.title else str(o.content))
        else:
            continue
        written.append(str(path))
    return written
