"""File-dependent facts a GUI needs to fill its widgets (no UI code).

`file_context(path)` opens the SEG-Y file (headers only) and returns what is in it:
the list of real FFIDs, the trace count of every shot, the record length.
`default_for(param, ctx, ffid)` turns a `Param.auto` tag into the actual value.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import grid as grid_mod, segy_io
from .registry import Param


@dataclass
class FileContext:
    sgy: segy_io.SegyFile
    index: segy_io.ShotIndex

    @property
    def ffids(self) -> list[int]:
        return [int(v) for v in self.index.ffids]

    @property
    def record_ms(self) -> float:
        return self.sgy.ns * self.sgy.dt_ms

    def shot_ntr(self, ffid: int) -> int:
        return self.index.ntr(ffid)


def file_context(path: str) -> FileContext:
    return FileContext(segy_io.open_segy(path), segy_io.shot_index(path))


def default_for(p: Param, ctx: FileContext | None, ffid: int = 0):
    """Value a widget starts with: the header-derived value when `p.auto` is set."""
    if ctx is None or not p.auto:
        return p.default
    if p.auto == "record_ms":                 # last time sample in the header: ns * dt
        return ctx.record_ms
    if p.auto == "last_trace":                # number of traces in the selected shot
        return ctx.shot_ntr(ffid) if ffid else ctx.shot_ntr(ctx.ffids[0])
    if p.auto == "first_ffid":
        return ctx.ffids[0]
    if p.auto == "last_ffid":
        return ctx.ffids[-1]
    if p.auto == "mute_velocity":             # mute line velocity, in the length unit of the file per second
        from . import geometry
        return 12000.0 if geometry.read_geometry(ctx.sgy.path, 60).unit == "ft" else 3500.0
    if p.auto == "velocity_table":            # a starting NMO velocity function in the unit of the file (ft/s or m/s)
        from . import geometry, stacking
        return stacking.default_velocity_rows(geometry.read_geometry(ctx.sgy.path, 60).unit)
    if p.auto == "grid_corners":              # the corner table: saved for this file, else the header grid's corners
        return grid_mod.corner_table(ctx.sgy.path)
    return p.default
