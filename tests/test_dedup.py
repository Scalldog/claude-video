"""Frame-delta dedup: per-pixel difference, greedy de-duplication, integration."""
from __future__ import annotations

import subprocess
from pathlib import Path

import frames


# --- _frame_delta: mean absolute per-pixel difference ------------------------

def test_frame_delta_identical_is_zero():
    a = bytes([10] * 16)
    assert frames._frame_delta(a, a) == 0.0


def test_frame_delta_is_mean_absolute_difference():
    a = bytes([0, 0, 0, 0])
    b = bytes([4, 0, 0, 0])
    assert frames._frame_delta(a, b) == 1.0  # (4+0+0+0)/4


def test_frame_delta_mismatched_length_is_infinite():
    assert frames._frame_delta(bytes([1, 2]), bytes([1, 2, 3])) == float("inf")


# --- _dedupe_by_deltas: greedy drop vs last *kept* thumbnail ------------------

def _touch(dirpath: Path, n: int) -> list[dict]:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = dirpath / f"frame_{i:04d}.jpg"
        p.write_bytes(b"x")
        out.append({"index": i, "timestamp_seconds": float(i), "path": str(p), "reason": "scene-change"})
    return out


FLAT0 = bytes([0, 0, 0, 0])
FLAT255 = bytes([255, 255, 255, 255])


def test_dedupe_collapses_identical_run(tmp_path: Path):
    cands = _touch(tmp_path, 5)
    thumbs = [FLAT0, FLAT0, FLAT0, FLAT0, FLAT0]
    survivors, dropped = frames._dedupe_by_deltas(cands, thumbs, threshold=2.0)
    assert dropped == 4
    assert len(survivors) == 1
    assert survivors[0]["index"] == 0
    assert sorted(p.name for p in tmp_path.glob("frame_*.jpg")) == ["frame_0000.jpg"]


def test_dedupe_keeps_all_distinct(tmp_path: Path):
    cands = _touch(tmp_path, 4)
    thumbs = [FLAT0, FLAT255, FLAT0, FLAT255]
    survivors, dropped = frames._dedupe_by_deltas(cands, thumbs, threshold=2.0)
    assert dropped == 0
    assert [s["index"] for s in survivors] == [0, 1, 2, 3]
    assert len(list(tmp_path.glob("frame_*.jpg"))) == 4


def test_dedupe_compares_against_last_kept_not_previous(tmp_path: Path):
    """A,A,B,B,A with A/B far apart -> keep A0, B2, A4 (drops the repeats)."""
    cands = _touch(tmp_path, 5)
    survivors, dropped = frames._dedupe_by_deltas(
        cands, [FLAT0, FLAT0, FLAT255, FLAT255, FLAT0], threshold=2.0
    )
    assert [s["index"] for s in survivors] == [0, 1, 2]  # reindexed survivors
    assert dropped == 2


def test_dedupe_threshold_is_inclusive(tmp_path: Path):
    """Delta exactly == threshold is treated as a duplicate (<=)."""
    cands = _touch(tmp_path, 2)
    a = bytes([0, 0, 0, 0])
    b = bytes([8, 0, 0, 0])  # mean abs diff == 2.0
    survivors, dropped = frames._dedupe_by_deltas(cands, [a, b], threshold=2.0)
    assert dropped == 1
    assert len(survivors) == 1


def test_dedupe_empty_and_single_are_noops(tmp_path: Path):
    assert frames._dedupe_by_deltas([], [], threshold=2.0) == ([], 0)
    one = _touch(tmp_path, 1)
    survivors, dropped = frames._dedupe_by_deltas(one, [FLAT0], threshold=2.0)
    assert dropped == 0
    assert len(survivors) == 1


def test_dedupe_mismatched_thumb_count_is_noop(tmp_path: Path):
    """Fail open: if thumbs don't line up with candidates, change nothing."""
    cands = _touch(tmp_path, 3)
    survivors, dropped = frames._dedupe_by_deltas(cands, [FLAT0], threshold=2.0)
    assert dropped == 0
    assert len(survivors) == 3


# --- _thumb_frames + dedupe_perceptual: real ffmpeg over extracted JPEGs ------

def test_thumb_frames_match_candidate_count(cut_clip: Path, tmp_path: Path):
    out = frames.extract_scene_candidates(str(cut_clip), tmp_path / "f", max_frames=None)
    thumbs = frames._thumb_frames([Path(fr["path"]) for fr in out])
    assert len(thumbs) == len(out)
    assert all(len(t) == frames.DEDUP_THUMB * frames.DEDUP_THUMB for t in thumbs)


def test_dedupe_perceptual_collapses_static_clip(static_clip: Path, tmp_path: Path):
    out = frames.extract(str(static_clip), tmp_path / "f", fps=4.0, max_frames=10)
    n_before = len(out)
    survivors, dropped = frames.dedupe_perceptual(out)
    assert n_before > 1
    assert len(survivors) == 1
    assert dropped == n_before - 1
    assert len(list((tmp_path / "f").glob("frame_*.jpg"))) == 1


def test_dedupe_perceptual_keeps_distinct_cuts(cut_clip: Path, tmp_path: Path):
    """Distinct color shots differ in luma, so frame-delta keeps them all."""
    out = frames.extract_scene_candidates(str(cut_clip), tmp_path / "f", max_frames=None)
    n_before = len(out)
    survivors, dropped = frames.dedupe_perceptual(out)
    assert dropped == 0
    assert len(survivors) == n_before


# --- engine integration: dedup runs before the cap, reports deduped_count -----

def test_scene_engine_reports_zero_dedup_on_distinct(cut_clip: Path, tmp_path: Path):
    """13 cuts already clear half of a budget of 20, so no top-up round-trip
    runs and this isolates dedup reporting to the scene candidates alone."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=20,
    )
    assert meta["engine"] == "scene"
    assert meta["topup_count"] == 0
    assert meta["deduped_count"] == 0
    assert len(out) == len(list((tmp_path / "f").rglob("frame_*.jpg")))


def test_topup_frames_are_deduped_before_merge(tmp_path: Path):
    """9 cuts followed by 6s of one static color: the top-up uniform-samples
    the whole clip, so most of its candidates land on the static tail and must
    collapse via dedup exactly like the fallback branch's own extract() does."""
    from conftest import build_cut_clip, build_static_clip

    cuts = tmp_path / "cuts.mp4"
    tail = tmp_path / "tail.mp4"
    build_cut_clip(cuts, n=9, seg=0.2)
    build_static_clip(tail, duration=6.0)
    combined = tmp_path / "sparse_static.mp4"
    filelist = tmp_path / "concat.txt"
    filelist.write_text(f"file '{cuts}'\nfile '{tail}'\n")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(filelist),
            "-c", "copy", str(combined),
        ],
        check=True,
    )

    budget = 60
    out, meta = frames.extract_scene_or_uniform(
        str(combined), tmp_path / "f", fps=2.0, target_frames=budget,
        resolution=256, max_frames=budget, dedup=True,
    )

    assert meta["engine"] == "scene+uniform"
    assert meta["topup_count"] > 0
    assert meta["deduped_count"] > 0  # would be 0 if top-up skipped dedup
    assert meta["selected_count"] < budget  # not re-topped-up to fill the cap

    # _thumb_frames needs one contiguous numbered sequence per directory, and a
    # top-up merges frame_*.jpg (scene) with topup/frame_*.jpg — so check each
    # source directory's survivors separately rather than the merged list.
    paths = [Path(f["path"]) for f in out]
    parents = {p.parent for p in paths}
    assert len(parents) == 2  # scene dir + topup dir, or this test stops proving anything
    for parent in parents:
        group = [p for p in paths if p.parent == parent]
        thumbs = frames._thumb_frames(group)
        assert thumbs, f"expected thumbnails for {parent}"
        for a, b in zip(thumbs, thumbs[1:]):
            assert frames._frame_delta(a, b) > frames.DEDUP_THRESHOLD


def test_uniform_fallback_dedupes_static(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=4.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert meta["deduped_count"] > 0
    assert meta["selected_count"] == 1  # identical frames collapse to one
    assert len(out) == 1
    assert len(list((tmp_path / "f").glob("frame_*.jpg"))) == 1


def test_keyframe_uniform_fallback_dedupes_static(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["deduped_count"] > 0
    assert len(out) == 1


def test_dedup_false_disables_collapse(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=4.0, target_frames=12, max_frames=100,
        dedup=False,
    )
    assert meta["deduped_count"] == 0
    assert meta["selected_count"] > 1  # no collapse without dedup
    assert len(out) > 1
