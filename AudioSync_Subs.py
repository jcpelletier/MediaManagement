import sys
import os
import logging
import subprocess

# --- Configuration ---
LOG_FILE = 'ffsubsync_audit.log'

# Max offset in seconds. If ffsubsync finds an offset greater than this,
# the subtitle is considered mismatched and ffsubsync will fail.
MAX_OFFSET_SECONDS = 60

# We can use this threshold just for our own logging.
# Note: we don't have the numeric offset from the CLI, so this is currently
# only used conceptually; the actual suppression is done by ffsubsync itself
# via --suppress-output-if-offset-less-than.
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
                # Take the first subtitle we see for a given stem
                if files_by_stem[name_stem]['sub'] is None:
                    files_by_stem[name_stem]['sub'] = os.path.join(dirpath, filename)

        for stem, files in files_by_stem.items():
            if files['video'] and files['sub']:
                movie_pairs[files['video']] = files['sub']

    logging.info(f"Found {len(movie_pairs)} video/subtitle pairs to process.")
    return movie_pairs


def process_sync(video_path, subtitle_path):
    """
    Attempts to auto-sync a single video/subtitle pair using ffsubsync CLI.
    Overwrites the original subtitle file.
    """
    logging.info(f"Processing: {os.path.basename(video_path)}")
    logging.info(f"Subtitle: {os.path.basename(subtitle_path)}")

    # Build ffsubsync CLI command:
    #   ffsubsync <video> -i <sub> --overwrite-input --max-offset-seconds N --suppress-output-if-offset-less-than 0.01
    cmd = [
        "ffsubsync",
        video_path,
        "-i", subtitle_path,
        "--overwrite-input",
        "--max-offset-seconds", str(MAX_OFFSET_SECONDS),
        "--suppress-output-if-offset-less-than", "0.01",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logging.error("SYNC FAILURE: ffsubsync exited with non-zero status.")
            if result.stdout:
                logging.error(f"ffsubsync stdout:\n{result.stdout}")
            if result.stderr:
                logging.error(f"ffsubsync stderr:\n{result.stderr}")
            return None

        # At this point, ffsubsync reported success and overwrote the input file.
        # We don't get the numeric offset directly from the CLI, but we can at
        # least log that the sync completed.
        if result.stdout.strip():
            logging.info("SYNC SUCCESS. ffsubsync output:")
            logging.info(result.stdout.strip())
        else:
            logging.info("SYNC SUCCESS. No output (likely a very small offset or suppressed output).")

        return True

    except FileNotFoundError:
        logging.critical(
            "ffsubsync command not found. Ensure ffsubsync is installed and on the PATH for this Jenkins job."
        )
        return None
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
