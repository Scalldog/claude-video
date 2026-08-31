"""Transcript-cue timestamps: parsing, point extraction, and pinned merge."""
from __future__ import annotations

from pathlib import Path

import pytest

import frames
import transcribe


def test_parse_timestamps_mixed_formats():
    assert frames.parse_timestamps("30,1:05,90") == [30.0, 65.0, 90.0]


def test_parse_timestamps_strips_and_dedupes():
    assert frames.parse_timestamps(" 90 , 30, 30 ") == [30.0, 90.0]


def test_parse_timestamps_empty():
    assert frames.parse_timestamps("") == []
    assert frames.parse_timestamps("  ,  ") == []


def test_parse_timestamps_rejects_garbage():
    with pytest.raises(SystemExit):
        frames.parse_timestamps("4:bad")


def test_merge_frames_sorts_and_reindexes():
    primary = [
        {"index": 0, "timestamp_seconds": 1.0, "path": "a", "reason": "scene-change"},
        {"index": 1, "timestamp_seconds": 5.0, "path": "b", "reason": "scene-change"},
    ]
    pinned = [
        {"index": 0, "timestamp_seconds": 3.0, "path": "c", "reason": "transcript-cue"},
    ]
    merged = frames.merge_frames(primary, pinned)
    assert [f["path"] for f in merged] == ["a", "c", "b"]
    assert [f["index"] for f in merged] == [0, 1, 2]
    assert merged[1]["reason"] == "transcript-cue"


def test_merge_frames_keeps_all_pinned():
    pinned = [{"index": 0, "timestamp_seconds": 2.0, "path": "c", "reason": "transcript-cue"}]
    merged = frames.merge_frames([], pinned)
    assert [f["path"] for f in merged] == ["c"]


def test_extract_at_timestamps_one_frame_per_point(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_at_timestamps(str(cut_clip), tmp_path / "f", [0.5, 2.0, 4.0])
    assert meta["engine"] == "timestamps"
    assert meta["fallback"] is False
    assert len(out) == 3
    assert all(f["reason"] == "transcript-cue" for f in out)
    ts = [f["timestamp_seconds"] for f in out]
    assert ts == sorted(ts)
    assert len(out) == len(list((tmp_path / "f").glob("cue_*.jpg")))


def test_extract_at_timestamps_drops_out_of_window(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_at_timestamps(
        str(cut_clip), tmp_path / "f", [0.5, 2.0, 4.0],
        start_seconds=1.0, end_seconds=3.0,
    )
    assert [f["timestamp_seconds"] for f in out] == [2.0]
    assert meta["dropped_out_of_window"] == 2


def test_extract_at_timestamps_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_at_timestamps(
        str(cut_clip), tmp_path / "f", [0.5, 1.5, 2.5, 3.5, 4.5], max_frames=3,
    )
    assert len(out) == 3
    ts = [f["timestamp_seconds"] for f in out]
    assert ts[0] == 0.5 and ts[-1] == 4.5  # even-sample keeps first + last
    assert len(out) == len(list((tmp_path / "f").glob("cue_*.jpg")))


# --- transcribe.filter_range: on the local-whisper path whenever --start/--end is used ---

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "intro"},
    {"start": 5.0, "end": 10.0, "text": "middle"},
    {"start": 10.0, "end": 15.0, "text": "outro"},
]


def test_filter_range_no_bounds_returns_all():
    assert transcribe.filter_range(SEGMENTS, None, None) == SEGMENTS


def test_filter_range_keeps_only_overlapping_segments():
    out = transcribe.filter_range(SEGMENTS, 6.0, 12.0)
    assert [seg["text"] for seg in out] == ["middle", "outro"]


def test_filter_range_start_only_is_open_ended():
    """A segment touching the start boundary still overlaps, same as the
    explicit boundary-touch case below."""
    out = transcribe.filter_range(SEGMENTS, 10.0, None)
    assert [seg["text"] for seg in out] == ["middle", "outro"]


def test_filter_range_end_only_starts_from_zero():
    out = transcribe.filter_range(SEGMENTS, None, 5.0)
    assert [seg["text"] for seg in out] == ["intro", "middle"]


def test_filter_range_boundary_touch_counts_as_overlap():
    """A segment ending exactly at the range start still overlaps (>=/<=, not
    strict), since seg["end"] >= lo and seg["start"] <= hi both hold at the
    touching instant."""
    out = transcribe.filter_range(SEGMENTS, 5.0, 5.0)
    assert [seg["text"] for seg in out] == ["intro", "middle"]


def test_filter_range_excludes_segments_entirely_outside():
    out = transcribe.filter_range(SEGMENTS, 20.0, 30.0)
    assert out == []


def test_extract_at_timestamps_does_not_clobber_detail_frames(cut_clip: Path, tmp_path: Path):
    """Cue frames live alongside detail frames in the same dir without deleting
    them; detail frames may span out_dir and out_dir/topup when topped up."""
    d = tmp_path / "f"
    scene, _ = frames.extract_scene_or_uniform(
        str(cut_clip), d, fps=2.0, target_frames=50, max_frames=100,
    )
    cues, _ = frames.extract_at_timestamps(str(cut_clip), d, [1.0, 3.0])
    assert len(list(d.rglob("frame_*.jpg"))) == len(scene)
    assert len(list(d.glob("cue_*.jpg"))) == len(cues)
