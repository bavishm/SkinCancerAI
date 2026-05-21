import os
import shutil

SRC_ROOT = "."
DEST_ROOT = "./final_results"

CHECKPOINT_PREFIX = "checkpoints_"
EVAL_DIR = "eval_data"

def should_copy(file_name):
    lower = file_name.lower()
    if lower.endswith(".pth"):
        return False
    if lower.endswith(".csv") and "class_distribution" in lower:
        return False
    return True

def copy_tree(src, dest):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dest, rel) if rel != "." else dest
        os.makedirs(dest_dir, exist_ok=True)

        for f in files:
            if not should_copy(f):
                continue
            src_path = os.path.join(root, f)
            dest_path = os.path.join(dest_dir, f)
            # Copy only if missing or different size/time
            if not os.path.exists(dest_path) or os.path.getsize(src_path) != os.path.getsize(dest_path):
                shutil.copy2(src_path, dest_path)

def main():
    os.makedirs(DEST_ROOT, exist_ok=True)

    # Copy all checkpoints_* folders
    for name in os.listdir(SRC_ROOT):
        if name.startswith(CHECKPOINT_PREFIX):
            src_path = os.path.join(SRC_ROOT, name)
            if os.path.isdir(src_path):
                dest_path = os.path.join(DEST_ROOT, name)
                copy_tree(src_path, dest_path)

    # Copy eval_data
    eval_path = os.path.join(SRC_ROOT, EVAL_DIR)
    if os.path.isdir(eval_path):
        dest_path = os.path.join(DEST_ROOT, EVAL_DIR)
        copy_tree(eval_path, dest_path)

if __name__ == "__main__":
    main()