import os
import shutil
import random

def organize_dataset():
    # Define the source directories for male and female images
    source_male = "data/male/"
    source_female = "data/female/"

    # Define target directories for training and validation sets
    target_train_male = "data/train/male/"
    target_train_female = "data/train/female/"
    target_val_male = "data/validation/male/"
    target_val_female = "data/validation/female/"

    # Ensure target directories exist
    os.makedirs(target_train_male, exist_ok=True)
    os.makedirs(target_train_female, exist_ok=True)
    os.makedirs(target_val_male, exist_ok=True)
    os.makedirs(target_val_female, exist_ok=True)

    # Split ratio for train and validation
    train_ratio = 0.8

    # Process male images
    male_images = os.listdir(source_male)
    random.shuffle(male_images)
    split_index = int(len(male_images) * train_ratio)
    train_male = male_images[:split_index]
    val_male = male_images[split_index:]

    for img in train_male:
        shutil.copy(os.path.join(source_male, img), target_train_male)
    for img in val_male:
        shutil.copy(os.path.join(source_male, img), target_val_male)

    # Process female images
    female_images = os.listdir(source_female)
    random.shuffle(female_images)
    split_index = int(len(female_images) * train_ratio)
    train_female = female_images[:split_index]
    val_female = female_images[split_index:]

    for img in train_female:
        shutil.copy(os.path.join(source_female, img), target_train_female)
    for img in val_female:
        shutil.copy(os.path.join(source_female, img), target_val_female)

    print("Dataset organized successfully.")

if __name__ == "__main__":
    organize_dataset()
