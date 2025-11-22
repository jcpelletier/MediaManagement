import sys
import os
import logging
import ffsubsync
import ffsubsync.cli as cli
from ffsubsync.constants import OFFSET_NOT_FOUND, OFFSET_TOO_LARGE

# --- Configuration ---
LOG_FILE = 'ffsubsync_audit.log'

# Max offset in seconds. If ffsubsync finds an offset greater than this,
# the subtitle is considered mismatched.
MAX_OFFSET_SECONDS = 60

# Threshold for treating subtitles as essentially synced.
SYNC_THRESHOLD_SECONDS = 5

# Video file extensions to search for
VIDEO_EXTS = ['.mp4', '.mkv', '.avi', '.mov']

# Subtitle file extensions
SUB_EXTS = ['.srt']

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)


def find_movie_pairs(root_dir):
    """
    Scans the directory for video/subtitle pairs.
    A pair is defined by having the same name but different extensions.
    """
    logging.info(f"Starting scan of directory: {root_dir}")
    movie_pairs = {}

    for dirpath, _, filenames in os.walk(root_dir):
        files_by_stem = {}

        for filename in filenames:
            name_stem, ext = os.path.splitext(filename)
            ext = ext.lower()

            if name_stem not in files_by_stem:
                files_by_stem[name_stem] = {'video': None, 'sub': None}

            if ext in VIDEO_EXTS:
                files_by_stem[name_stem]['video'] = os.path.join(dirpath, filename)
            elif ext in SUB_EXTS:
                if files_by_stem[name_stem]['sub'] is None:
                    files_by_stem[name_stem]['sub'] = os.path.join(dirpath, filename)

        for stem, files in files_by_stem.items():
            if files['video'] and files['sub']:
                movie_pairs[files['video']] = files['sub']

    logging.info(f"Found {len(movie_pairs)} video/subtitle pairs to process.")
    return movie_pairs


def process_sync(video_path, subtitle_path):
    """
    Attempts to auto-sync a single video/subtitle pair using ffsubsync.
    Overwrites the original subtitle file.
    """
    logging.info(f"Processing: {os.path.basename(video_path)}")
    logging.info(f"Subtitle: {os.path.basename(subtitle_path)}")

    try:
        sync_args = cli.SyncArgs(
            reference=video_path,
            srtin=[subtitle_path],
            srtout=subtitle_path,
            max_offset_seconds=MAX_OFFSET_SECONDS,
            suppress_output_if_offset_less_than=0.01
        )

        offset = ffsubsync.ffsubsync(sync_args)

        if offset == OFFSET_NOT_FOUND:
            logging.error("SYNC FAILURE: Could not find any alignment. Subtitle likely does not match audio.")

        elif offset == OFFSET_TOO_LARGE:
            logging.error(f"SYNC FAILURE: Required offset exceeded {MAX_OFFSET_SECONDS} seconds.")

        elif abs(offset) < SYNC_THRESHOLD_SECONDS:
            logging.info(f"Already Synced: Minor drift correction applied ({offset:.3f}s).")

        else:
            logging.warning(f"SYNC SUCCESS: Applied time correction of {offset:.3f}s.")

        return offset

    except Exception as e:
        logging.error(f"UNEXPECTED ERROR during sync of {os.path.basename(video_path)}: {e}")
        return None


def main():
    """
    Main entry point for command-line usage.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} \"<directory_path>\"")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.isdir(root_directory):
        logging.critical(f"Directory not found: {root_directory}")
        sys.exit(1)

    movie_pairs = find_movie_pairs(root_directory)

    for video_file, subtitle_file in movie_pairs.items():
        process_sync(video_file, subtitle_file)

    logging.info("Script Finished.")
    logging.info(f"Full log output saved to: {os.path.abspath(LOG_FILE)}")


if __name__ == "__main__":
    main()
