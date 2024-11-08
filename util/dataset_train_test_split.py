import os
import shutil
from sklearn.model_selection import train_test_split

# Define constants for dataset paths and split ratio
DATASET_DIR = '../dataset/chest_XRAY'
TRAIN_DIR = '../dataset/chest_XRAY/train/'
TEST_DIR = '../dataset/chest_XRAY/test/'
TEST_SIZE = 0.2
SEED = 42

def create_directory_structure(train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    """Create train and test directories with subfolders for each class."""
    for dir_path in [train_dir, test_dir]:
        for category in ['Normal', 'Tuberculosis']:
            os.makedirs(os.path.join(dir_path, category), exist_ok=True)


def split_dataset(dataset_dir=DATASET_DIR, test_size=TEST_SIZE, seed=SEED):
    """Split the dataset into training and testing sets."""
    splits = {}

    for category in ['Normal', 'Tuberculosis']:
        category_path = os.path.join(dataset_dir, category)
        images = os.listdir(category_path)

        # Split into train and test sets
        train_images, test_images = train_test_split(
            images, test_size=test_size, random_state=seed
        )
        splits[category] = {'train': train_images, 'test': test_images}

    return splits


def copy_images(splits, dataset_dir=DATASET_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    """Copy images into the respective train and test directories."""
    for category, split in splits.items():
        # Copy training images
        for img in split['train']:
            src_path = os.path.join(dataset_dir, category, img)
            dst_path = os.path.join(train_dir, category, img)
            shutil.copy(src_path, dst_path)

        # Copy testing images
        for img in split['test']:
            src_path = os.path.join(dataset_dir, category, img)
            dst_path = os.path.join(test_dir, category, img)
            shutil.copy(src_path, dst_path)


def main():
    """Main function to create directories, split dataset, and copy images."""
    print("Creating directory structure...")
    create_directory_structure()

    print("Splitting dataset...")
    splits = split_dataset()

    print("Copying images...")
    copy_images(splits)

    print("Dataset split into training and testing sets successfully!")


if __name__ == "__main__":
    main()
