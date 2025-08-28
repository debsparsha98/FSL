import os

def check_dataset(images_dir, labels_dir):
    images = set(os.path.splitext(f) for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    labels = set(os.path.splitext(f) for f in os.listdir(labels_dir) if f.lower().endswith(".txt"))
    print("Images without labels:", images - labels)
    print("Labels without images:", labels - images)

# Example usage:
# check_dataset("D:/projrct_root/data/images/train", "D:/projrct_root/data/labels/train")
