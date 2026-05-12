"""
FER2013 Dataset Downloader

Downloads the FER2013 dataset from Kaggle and organizes it
into the expected folder structure for training.
"""

import os
import sys
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FER_DIR = os.path.join(BASE_PATH, "fer2013")


def check_existing():
    """Check if FER2013 is already downloaded and organized."""
    train_dir = os.path.join(FER_DIR, "train")
    test_dir = os.path.join(FER_DIR, "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        train_ok = all(os.path.isdir(os.path.join(train_dir, e)) for e in emotions)
        test_ok = all(os.path.isdir(os.path.join(test_dir, e)) for e in emotions)

        if train_ok and test_ok:
            train_count = sum(
                len(os.listdir(os.path.join(train_dir, e)))
                for e in emotions
                if os.path.isdir(os.path.join(train_dir, e))
            )
            test_count = sum(
                len(os.listdir(os.path.join(test_dir, e)))
                for e in emotions
                if os.path.isdir(os.path.join(test_dir, e))
            )
            print(f"[OK] FER2013 already exists!")
            print(f"     Train images: {train_count}")
            print(f"     Test images:  {test_count}")
            return True
    return False


def download_with_kagglehub():
    """Try to download using kagglehub (simplest method)."""
    try:
        import kagglehub
        print("[DOWNLOAD] Downloading FER2013 via kagglehub...")
        path = kagglehub.dataset_download("msambare/fer2013")
        print(f"   Downloaded to: {path}")

        # Copy to our project directory
        if os.path.normpath(path) != os.path.normpath(FER_DIR):
            print(f"[COPY] Copying to {FER_DIR}...")
            if os.path.exists(FER_DIR):
                shutil.rmtree(FER_DIR)
            shutil.copytree(path, FER_DIR)

        return True
    except ImportError:
        print("[WARN] kagglehub not installed.")
        return False
    except Exception as e:
        print(f"[WARN] kagglehub download failed: {e}")
        return False


def download_with_kaggle_api():
    """Try to download using the official Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        print("[DOWNLOAD] Downloading FER2013 via Kaggle API...")

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("msambare/fer2013", path=FER_DIR, unzip=True)

        print(f"   Downloaded to: {FER_DIR}")
        return True
    except ImportError:
        print("[WARN] kaggle API not installed.")
        return False
    except Exception as e:
        print(f"[WARN] Kaggle API download failed: {e}")
        return False


def print_manual_instructions():
    """Print instructions for manual download."""
    print("""
================================================================
              MANUAL DOWNLOAD INSTRUCTIONS
================================================================

  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013

  2. Click "Download" (you need a free Kaggle account)

  3. Extract the downloaded ZIP file

  4. Copy the contents so the structure looks like:

     face-rego-main/
     +-- fer2013/
         +-- train/
         |   +-- angry/     (3995 images)
         |   +-- disgust/   (436 images)
         |   +-- fear/      (4097 images)
         |   +-- happy/     (7215 images)
         |   +-- sad/       (4830 images)
         |   +-- surprise/  (3171 images)
         |   +-- neutral/   (4965 images)
         +-- test/
             +-- angry/     (958 images)
             +-- disgust/   (111 images)
             +-- fear/      (1024 images)
             +-- happy/     (1774 images)
             +-- sad/       (1247 images)
             +-- surprise/  (831 images)
             +-- neutral/   (1233 images)

  5. Re-run this script to verify the download.

================================================================
""")


def main():
    print("=" * 60)
    print("  FER2013 Dataset Setup")
    print("=" * 60)

    if check_existing():
        print("\n[OK] Dataset is ready! You can proceed to training.")
        return True

    print("\n[INFO] FER2013 not found. Attempting download...\n")

    # Try kagglehub first
    if download_with_kagglehub():
        if check_existing():
            print("\n[OK] Download complete! Dataset is ready for training.")
            return True

    # Try Kaggle API
    if download_with_kaggle_api():
        if check_existing():
            print("\n[OK] Download complete! Dataset is ready for training.")
            return True

    # Manual instructions
    print("\n[FAIL] Automatic download failed.")
    print_manual_instructions()

    print("TIP: You can install kagglehub to enable automatic download:")
    print("   pip install kagglehub")
    print("   Then re-run this script.")

    return False


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
