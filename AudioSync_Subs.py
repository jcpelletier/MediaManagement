import sys
import os
import glob
import logging
from ffsubsync import ffsubsync, cli
from ffsubsync.constants import OFFSET_NOT_FOUND, OFFSET_TOO_LARGE

# --- Configuration ---
# Set up logging to record activity, especially sync failures
LOG_FILE = 'ffsubsync_audit.log'
# Max offset in seconds. If ffsubsync finds an offset greater than this, 
# it treats it as a failed sync, suggesting the SRT likely doesn't match the audio.
# The ffsubsync default is 60 seconds.
MAX_OFFSET_SECONDS = 60 
# Threshold for considering subtitles 'already synced'.
# If the calculated offset is less than this, we consider it a minor drift correction.
SYNC_THRESHOLD_SECONDS = 5
# Video file extensions to search for
VIDEO_EXTS = ['.mp4', '.mkv', '.avi', '.mov']
# Subtitle file extensions to search for
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
    Example: 'Movie.mkv' and 'Movie.srt'
    """
    logging.info(f"🔍 Starting scan of directory: {root_dir}")
    movie_pairs = {}
    
    # Use os.walk for comprehensive traversal and better performance
    for dirpath, _, filenames in os.walk(root_dir):
        # Separate files by their name stem
        files_by_stem = {}
        for filename in filenames:
            name_stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if name_stem not in files_by_stem:
                files_by_stem[name_stem] = {'video': None, 'sub': None}
            
            if ext in VIDEO_EXTS:
                files_by_stem[name_stem]['video'] = os.path.join(dirpath, filename)
            elif ext in SUB_EXTS:
                # We'll prioritize the first .srt found for a given stem
                if files_by_stem[name_stem]['sub'] is None:
                    files_by_stem[name_stem]['sub'] = os.path.join(dirpath, filename)

        # Identify complete pairs
        for stem, files in files_by_stem.items():
            if files['video'] and files['sub']:
                # The video file path is the unique key
                movie_pairs[files['video']] = files['sub']
    
    logging.info(f"✅ Found {len(movie_pairs)} video/subtitle pairs to process.")
    return movie_pairs

def process_sync(video_path, subtitle_path):
    """
    Attempts to auto-sync a single video/subtitle pair using ffsubsync.
    Overwrites the original subtitle file.
    """
    logging.info(f"\n--- Processing: {os.path.basename(video_path)}")
    logging.info(f"   Subtitle: {os.path.basename(subtitle_path)}")

    # ffsubsync.cli.sync provides a structured way to call the sync function 
    # and returns the offset in seconds.
    try:
        # Arguments to pass to the underlying ffsubsync function
        sync_args = cli.SyncArgs(
            reference=video_path,
            srtin=[subtitle_path],
            srtout=subtitle_path,  # Overwrite the input file
            max_offset_seconds=MAX_OFFSET_SECONDS,
            # Suppress output if the offset is less than a minimal amount, 
            # effectively treating it as already synced.
            suppress_output_if_offset_less_than=0.01 
        )
        
        # The core synchronization call
        offset = ffsubsync(sync_args)

        # Check the result of the sync
        if offset == OFFSET_NOT_FOUND:
            # OFFSET_NOT_FOUND indicates sync failed (likely subtitle mismatch or max_offset exceeded)
            logging.error(f"🔴 SYNC FAILURE: Could not find an alignment. Subtitle likely does not match audio.")
        
        elif offset == OFFSET_TOO_LARGE:
            # OFFSET_TOO_LARGE is a specific error for offsets beyond the limit
            logging.error(f"🔴 SYNC FAILURE: Required offset was > {MAX_OFFSET_SECONDS}s. Subtitle likely does not match audio.")

        elif abs(offset) < SYNC_THRESHOLD_SECONDS:
            # Offset is very small, we assume it was already synced or only needed minor correction
            logging.info(f"🟢 Already Synced: Offset of {offset:.3f}s found (minor correction applied).")
            
        else:
            # A significant, successful offset was found and applied
            logging.warning(f"🟡 SYNC SUCCESS: Applied a major time correction of {offset:.3f}s. Subtitles were out of sync.")
        
        return offset

    except Exception as e:
        # Catch any unexpected errors during processing
        logging.error(f"❌ UNEXPECTED ERROR during sync of {os.path.basename(video_path)}: {e}")
        return None

def main():
    """
    Main function to handle command-line arguments and orchestrate the sync process.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} \"<directory_path>\"")
        sys.exit(1)

    root_directory = sys.argv[1]
    
    if not os.path.isdir(root_directory):
        logging.critical(f"Directory not found: {root_directory}")
        sys.exit(1)

    # Find all pairs
    movie_pairs = find_movie_pairs(root_directory)
    
    # Process each pair
    for video_file, subtitle_file in movie_pairs.items():
        process_sync(video_file, subtitle_file)
        
    logging.info("\n--- Script Finished ---")
    logging.info(f"Full log output saved to: {os.path.abspath(LOG_FILE)}")


if __name__ == "__main__":
    main()