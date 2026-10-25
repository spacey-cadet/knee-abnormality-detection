"""Canonical orientation + laterality for the cached series arrays (no re-caching needed).

cache_data.py stores each series as a[D, row, col] in the scanner's own convention:
  * rows grow along the DICOM column-direction cosines c, columns along the row-direction cosines r
  * slices are sorted ascending along r x c, whose SIGN depends on the scanner
  * nothing knows whether the knee is a left or a right one
This module derives, from ONE DICOM header per series, the flips/transpose that bring every series into one
frame (patient LPS: x right->left, y anterior->posterior, z inferior->superior):
  Sagittal : slices ascending +x, columns +y (anterior->posterior), rows -z (superior at the top)
  Coronal  : slices ascending +y, columns +x (right->left),         rows -z
  Axial    : slices descending z (superior->inferior), columns +x,  rows +y
and then, for RIGHT knees, mirrors the medio-lateral axis (sagittal: reverse slice order; coronal/axial: flip
columns), so "medial" is always the same side. Works on the already padded+resized square arrays: transposes
and flips of a centred square are exact.

Per-slot code (int): -1 = unusable (missing series / IOP plane disagrees with the CSV plane / unreadable header),
else bit0 transpose, bit1 flip_cols, bit2 flip_rows, bit3 reverse_slices.
"""
import os
from pathlib import Path

import numpy as np

CANON = {0: (+1, (1, +1), (2, -1)),     # plane -> (slice sign, (col axis, sign), (row axis, sign)); axes 0=x 1=y 2=z
         1: (+1, (0, +1), (2, -1)),
         2: (-1, (0, +1), (1, +1))}
LAT_MIN_OFFSET_MM = 20.0                # |x| of the image centre below this -> side unknown (as in the public stack)


def tf_code(iop, plane_csv):
    """iop: 6 floats. plane_csv: 0/1/2 from the CSV. -> int code (see module docstring)."""
    try:
        iop = np.asarray(iop, float)
        if iop.shape != (6,) or not np.isfinite(iop).all():
            return -1
        r, c = iop[:3], iop[3:]
        nrm = np.cross(r, c)
        plane = int(np.abs(nrm).argmax())
        if plane != int(plane_csv) or abs(nrm[plane]) < 0.8:          # oblique or disagreeing: leave untouched
            return -1
        dsgn, (cax, csgn), (rax, rsgn) = CANON[plane]
        rev_d = dsgn * np.sign(nrm[plane]) < 0
        transpose = abs(r[cax]) < abs(r[rax])
        if transpose:
            r, c = c, r
        flip_cols = r[cax] * csgn < 0
        flip_rows = c[rax] * rsgn < 0
        return int(transpose) | int(flip_cols) << 1 | int(flip_rows) << 2 | int(rev_d) << 3
    except Exception:
        return -1


def apply_tf(a, code):
    """a: [D, H, W] (H == W) -> canonical [D, H, W] (a new contiguous array)."""
    if code < 0:
        return a
    if code & 1:
        a = a.transpose(0, 2, 1)
    if code & 2:
        a = a[:, :, ::-1]
    if code & 4:
        a = a[:, ::-1, :]
    if code & 8:
        a = a[::-1]
    return np.ascontiguousarray(a)


def apply_side(a, plane, side):
    """Mirror the medio-lateral axis of a CANONICAL series for right knees."""
    if side != "R":
        return a
    return np.ascontiguousarray(a[::-1] if int(plane) == 0 else a[:, :, ::-1])


def apply_study(arr, plane, present, codes, side):
    """arr [S, D, H, W] uint8 -> canonical copy. Slots with code -1 (or absent) are left as they are.
    The side flip is only applied to slots that were canonicalised (otherwise the axis meaning is unknown)."""
    out = arr.copy()
    for j in range(arr.shape[0]):
        if not present[j] or codes[j] < 0:
            continue
        out[j] = apply_side(apply_tf(arr[j], int(codes[j])), plane[j], side)
    return out


# ---------------------------------------------------------------- header reading (pydicom)
def _vec(v, n):
    try:
        x = [float(t) for t in v]
        return np.array(x) if len(x) >= n and np.isfinite(x[:n]).all() else None
    except Exception:
        return None


def series_info(sdir):
    """One header read (middle file by name). -> dict(iop, tag, cx) ; any field may be None."""
    import pydicom
    info = dict(iop=None, tag=None, cx=None)
    try:
        files = sorted(os.listdir(sdir))
        ds = pydicom.dcmread(Path(sdir) / files[len(files) // 2], stop_before_pixels=True, force=True)
    except Exception:
        return info
    iop = _vec(getattr(ds, "ImageOrientationPatient", None) or [], 6)
    info["iop"] = None if iop is None else iop.tolist()
    for t in ("Laterality", "ImageLaterality"):
        v = str(getattr(ds, t, "") or "").strip().upper()
        if v[:1] in ("L", "R"):
            info["tag"] = v[0]
            break
    ipp, ps = _vec(getattr(ds, "ImagePositionPatient", None) or [], 3), _vec(getattr(ds, "PixelSpacing", None) or [], 2)
    rows, cols = getattr(ds, "Rows", None), getattr(ds, "Columns", None)
    if iop is not None and ipp is not None and ps is not None and rows and cols:
        centre = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        info["cx"] = float(centre[0])
    return info


def study_side(infos):
    """infos: list of series_info dicts of one study -> (side 'L'/'R'/'', source 'tag'/'geometry'/'', tag_geo_disagree)."""
    tags = [i["tag"] for i in infos if i and i.get("tag")]
    xs = [i["cx"] for i in infos if i and i.get("cx") is not None]
    geo = ""
    if xs:
        m = float(np.median(xs))
        geo = "" if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    if tags:
        return tags[0], "tag", int(bool(geo) and geo != tags[0])
    return geo, ("geometry" if geo else ""), 0


def select_series(rows, K):
    """Replicates cache_data.process_study's choice: per plane the K best-ranked series -> list of (slot, row|None)."""
    from cache_data import rank_series
    from common import PLANES
    out = []
    for pi, p in enumerate(PLANES):
        cand = rank_series(rows[rows.Anatomical_Plane == p])
        for k in range(K):
            out.append((pi * K + k, pi, cand[k] if k < len(cand) else None))
    return out


def study_geometry(root, study, rows, K=1):
    """-> (codes list[S], side, side_src, disagree). Same series choice as the cache."""
    infos, codes, planes = [], [], []
    for slot, pi, r in select_series(rows, K):
        planes.append(pi)
        if r is None:
            infos.append(None); codes.append(-1); continue
        inf = series_info(Path(root) / study / r.SeriesInstanceUID)
        infos.append(inf)
        codes.append(tf_code(inf["iop"], pi) if inf["iop"] else -1)
    side, src, dis = study_side(infos)
    return codes, side, src, dis
