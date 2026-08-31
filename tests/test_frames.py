"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

import subprocess
from pathlib import Path

import frames


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    """13 cuts against a budget of 100 mirrors the sparse-scene bug this suite
    exists to catch, so the scene path is expected to top up toward budget."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene+uniform"
    assert meta["topup_count"] > 0
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


def test_scene_selection_tops_up_toward_budget(tmp_path):
    from conftest import build_cut_clip

    clip = tmp_path / "sparse.mp4"
    build_cut_clip(clip, n=10, seg=1.0)

    selected, meta = frames.extract_scene_or_uniform(
        str(clip),
        tmp_path / "out",
        fps=2.0,
        target_frames=40,
        resolution=256,
        max_frames=40,
    )

    assert meta["engine"] == "scene+uniform"
    assert meta["topup_count"] > 0
    assert len(selected) > 10
    stamps = [f["timestamp_seconds"] for f in selected]
    assert stamps == sorted(stamps)
    assert [f["index"] for f in selected] == list(range(len(selected)))


def test_frames_uses_fps_mode_not_removed_vsync():
    source = Path(frames.__file__).read_text(encoding="utf-8")
    assert '"-vsync"' not in source
    assert source.count('"-fps_mode"') == 2


def _build_cuts_then_static_clip(tmp_path: Path, cuts_seg: float, tail_duration: float) -> Path:
    """9 cuts followed by a static tail, concatenated into one clip.

    Scene changes only exist in the leading cuts segment, so any frame in the
    tail must come from the top-up pass — this isolates top-up coverage from
    the scene engine's own spread."""
    from conftest import build_cut_clip, build_static_clip

    cuts = tmp_path / "cuts.mp4"
    tail = tmp_path / "tail.mp4"
    build_cut_clip(cuts, n=9, seg=cuts_seg)
    build_static_clip(tail, duration=tail_duration)
    combined = tmp_path / "cuts_then_static.mp4"
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
    return combined


def test_topup_covers_the_tail_of_the_range(tmp_path: Path):
    """The top-up pass must span the whole clip, not stop short partway
    through — a `-frames:v N` cap on a single ffmpeg pass makes it stop after
    N *output* frames rather than spreading N frames across the range, so a
    naive `max_frames=shortfall` request truncates before reaching the end.

    ``max_frames`` (the cap the caller passed) is only a generous topup
    extraction cap while it is >= ``fps * duration`` — an explicit `--fps`
    override recomputes ``target_frames`` uncapped by the frame cap (see
    watch.py), so ``max_frames`` alone can fall *below* the natural yield.
    This case is pinned deliberately below that natural yield
    (``target_frames=40`` at ``fps=2.0`` over a 20s clip, but
    ``max_frames=30``) so the test can't pass on the boundary case where
    ``max_frames == fps * duration`` happens to already cover the tail."""
    clip = _build_cuts_then_static_clip(tmp_path, cuts_seg=0.4, tail_duration=16.4)
    meta = frames.get_metadata(str(clip))
    duration = meta["duration_seconds"]

    out, frame_meta = frames.extract_scene_or_uniform(
        str(clip), tmp_path / "f", fps=2.0, target_frames=40,
        max_frames=30, dedup=False,
    )

    assert frame_meta["engine"] == "scene+uniform"
    assert frame_meta["topup_count"] > 0
    last_ts = max(fr["timestamp_seconds"] for fr in out)
    assert last_ts >= duration - 3.0, (
        f"last selected frame at {last_ts}s, clip is {duration}s — top-up stopped short of the tail"
    )


def test_topup_candidate_count_reconciles_with_selected_and_deduped(tmp_path: Path):
    """Guards against the report-line arithmetic going incoherent (e.g. more
    frames selected than candidates offered): on any top-up run,
    selected + deduped must equal the reported candidate count."""
    clip = _build_cuts_then_static_clip(tmp_path, cuts_seg=0.4, tail_duration=21.4)
    meta = frames.get_metadata(str(clip))
    fps, target = frames.auto_fps(meta["duration_seconds"], max_frames=25)

    out, frame_meta = frames.extract_scene_or_uniform(
        str(clip), tmp_path / "f", fps=fps, target_frames=target, max_frames=25,
    )

    assert frame_meta["topup_count"] > 0
    assert frame_meta["selected_count"] <= frame_meta["candidate_count"]
    assert frame_meta["selected_count"] + frame_meta["deduped_count"] == frame_meta["candidate_count"]
