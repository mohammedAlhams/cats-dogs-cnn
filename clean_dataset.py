import os
from PIL import Image, UnidentifiedImageError

def remove_bad_images(root_dir):
    removed = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()  # Check if corrupted
            except (UnidentifiedImageError, OSError):
                print("DELETING BAD FILE:", filepath)
                os.remove(filepath)
                removed += 1
    print(f"\nDone! Removed {removed} corrupted images.")

if __name__ == "__main__":
    remove_bad_images("data")
