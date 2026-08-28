import os
import sys
import shutil
from pathlib import Path
import pytest
import Sort_Rips

def test_sort_rips_conditional_processed_move(tmp_path, monkeypatch):
    # Set up source, dest, and processed paths
    source_root = tmp_path / "source"
    dest_root = tmp_path / "dest"
    processed_root = tmp_path / "processed"

    source_root.mkdir()
    dest_root.mkdir()
    processed_root.mkdir()

    # Create dummy folders in source root
    # 1. Folder that succeeds identification and is filed
    folder_success = source_root / "movie_success"
    folder_success.mkdir()
    (folder_success / "video.mkv").write_bytes(b"dummy")

    # 2. Folder that has "no video files" skip (intentional defer)
    # Put a non-video file in it so it is not completely empty, verifying it gets moved to processed.
    folder_no_video = source_root / "no_video_folder"
    folder_no_video.mkdir()
    (folder_no_video / "readme.txt").write_bytes(b"dummy")

    # 2b. Completely empty folder (will be deleted, not moved)
    folder_empty = source_root / "empty_folder"
    folder_empty.mkdir()

    # 3. Folder that is a TV disc deferral (intentional defer)
    folder_tv_defer = source_root / "tv_defer_folder"
    folder_tv_defer.mkdir()
    (folder_tv_defer / "video.mkv").write_bytes(b"dummy")

    # 4. Folder that fails with LLM error (failure outcome - should NOT be moved)
    folder_llm_fail = source_root / "llm_fail_folder"
    folder_llm_fail.mkdir()
    (folder_llm_fail / "video.mkv").write_bytes(b"dummy")

    # 5. Folder that fails with low confidence (failure outcome - should NOT be moved)
    folder_low_conf = source_root / "low_confidence_folder"
    folder_low_conf.mkdir()
    (folder_low_conf / "video.mkv").write_bytes(b"dummy")

    # Mock DEEPSEEK_API_KEY env var
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy_key")

    # Mock sys.argv for Sort_Rips
    test_args = [
        "Sort_Rips.py",
        "--source", str(source_root),
        "--dest", str(dest_root),
        "--processed", str(processed_root),
        "--no-whisper-fallback",
        "--no-verify-api"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Mock process_folder to simulate various outcomes
    def mock_process_folder(folder, *args, **kwargs):
        if folder.name == "movie_success":
            return "movie_success (2025).mkv", None
        elif folder.name in ("no_video_folder", "empty_folder"):
            return None, "no video files"
        elif folder.name == "tv_defer_folder":
            return None, Sort_Rips.TV_DISC_DEFER_REASON
        elif folder.name == "llm_fail_folder":
            return None, "LLM call failed"
        elif folder.name == "low_confidence_folder":
            return None, "low confidence (0.45) after sampling content"
        return None, "unknown reason"

    monkeypatch.setattr(Sort_Rips, "process_folder", mock_process_folder)

    Sort_Rips.main()

    # Assertions on filesystem state:
    # 1. Successful movie folder: moved to processed (moved=True)
    assert not (source_root / "movie_success").exists()
    assert (processed_root / "movie_success").exists()

    # 2. No video files folder (non-empty): moved to processed (intentional defer)
    assert not (source_root / "no_video_folder").exists()
    assert (processed_root / "no_video_folder").exists()

    # 2b. Completely empty folder: deleted (so neither in source nor processed)
    assert not (source_root / "empty_folder").exists()
    assert not (processed_root / "empty_folder").exists()

    # 3. TV disc defer folder: moved to processed (intentional defer)
    assert not (source_root / "tv_defer_folder").exists()
    assert (processed_root / "tv_defer_folder").exists()

    # 4. LLM failure folder: stayed in place (not moved)
    assert (source_root / "llm_fail_folder").exists()
    assert not (processed_root / "llm_fail_folder").exists()

    # 5. Low confidence folder: stayed in place (not moved)
    assert (source_root / "low_confidence_folder").exists()
    assert not (processed_root / "low_confidence_folder").exists()
