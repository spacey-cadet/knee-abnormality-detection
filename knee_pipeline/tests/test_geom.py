"""numpy-only checks of geom.py: every acquisition convention must land in the same canonical array."""
import itertools, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom import tf_code, apply_tf, apply_side, apply_study, study_side

rng = np.random.default_rng(0)
N = 12
V = rng.integers(0, 255, (N, N, N)).astype(np.uint8)          # V[ix, iy, iz], index grows along +x(left), +y(post), +z(sup)

def axis_vec(ax, sgn):
    v = np.zeros(3); v[ax] = sgn; return v

def acquire(plane, rax, rsgn, cax, csgn):
    """series array a[d,row,col] as cache_data.read_series builds it: columns grow along r, rows along c,
    slices ascending along n = r x c."""
    r, c = axis_vec(rax, rsgn), axis_vec(cax, csgn)
    n = np.cross(r, c); nax = int(np.abs(n).argmax()); nsgn = int(np.sign(n[nax]))
    D = np.empty((N, N, N), np.uint8)
    for d in range(N):
        for i in range(N):
            for j in range(N):
                idx = [0, 0, 0]
                idx[rax] = j if rsgn > 0 else N - 1 - j
                idx[cax] = i if csgn > 0 else N - 1 - i
                idx[nax] = d if nsgn > 0 else N - 1 - d
                D[d, i, j] = V[tuple(idx)]
    return D, list(r) + list(c), nax

def canonical(plane):
    D = np.empty((N, N, N), np.uint8)
    for d in range(N):
        for i in range(N):
            for j in range(N):
                if plane == 0:   D[d, i, j] = V[d, j, N - 1 - i]          # slices +x, cols +y, rows -z
                elif plane == 1: D[d, i, j] = V[j, d, N - 1 - i]          # slices +y, cols +x, rows -z
                else:            D[d, i, j] = V[j, i, N - 1 - d]          # slices -z, cols +x, rows +y
    return D

n_cases = 0
for plane in range(3):
    want = canonical(plane)
    for rax, cax in itertools.permutations(range(3), 2):
        nax = ({0, 1, 2} - {rax, cax}).pop()
        if nax != plane: continue
        for rsgn, csgn in itertools.product((1, -1), repeat=2):
            a, iop, _ = acquire(plane, rax, rsgn, cax, csgn)
            code = tf_code(iop, plane)
            assert code >= 0, (plane, rax, rsgn, cax, csgn)
            got = apply_tf(a, code)
            assert np.array_equal(got, want), f"plane {plane} r=({rax},{rsgn}) c=({cax},{csgn}) code={code}"
            n_cases += 1
print("canonical orientation identical for", n_cases, "acquisition conventions")

# plane disagreement / oblique / garbage -> untouched (-1)
assert tf_code([0, 1, 0, 0, 0, -1], 1) == -1
assert tf_code([0.7, 0.7, 0, 0, 0, -1], 0) == -1
assert tf_code([np.nan] * 6, 0) == -1 and tf_code(None, 0) == -1

# right-knee mirror: a right knee scanned = the same anatomy mirrored in x. After side-normalisation the left
# knee and its mirrored right twin must be identical.
Vm = V[::-1].copy()                                               # mirror x  (x -> -x)
for plane in range(3):
    left = canonical(plane)
    V_save = V.copy(); V[:] = Vm; right = canonical(plane); V[:] = V_save
    assert not np.array_equal(left, right) or plane == 99
    assert np.array_equal(apply_side(right, plane, "R"), left), plane
    assert np.array_equal(apply_side(left, plane, "L"), left)
print("right/left mirror normalisation OK for 3 planes")

# apply_study leaves absent / -1 slots untouched, side flip only on canonicalised slots
arr = rng.integers(0, 255, (3, 4, 6, 6)).astype(np.uint8)
out = apply_study(arr, np.array([0, 1, 2]), np.array([1, 1, 0]), [9, -1, 5], "R")
assert np.array_equal(out[1], arr[1]) and np.array_equal(out[2], arr[2])
assert np.array_equal(out[0], apply_side(apply_tf(arr[0], 9), 0, "R"))

# laterality: tag wins, geometry fallback, small offset -> unknown, disagreement counted
I = lambda tag=None, cx=None: dict(iop=None, tag=tag, cx=cx)
assert study_side([I("L", -80)]) == ("L", "tag", 1)
assert study_side([I(None, -80), I(None, -60)]) == ("R", "geometry", 0)
assert study_side([I(None, 80)]) == ("L", "geometry", 0)
assert study_side([I(None, 5)]) == ("", "", 0)
assert study_side([None, I()]) == ("", "", 0)
print("laterality rules OK\nALL GEOM TESTS PASS")
