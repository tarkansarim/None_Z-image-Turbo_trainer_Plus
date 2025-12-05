from modelscope.hub.snapshot_download import snapshot_download
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python download_model.py <local_dir>")
    sys.exit(1)

model_dir = sys.argv[1]
print(f"Starting download to {model_dir}...")

try:
    snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir=model_dir)
    print("Download complete!")
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
