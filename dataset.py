import os
import pandas as pd
import shutil

# --- Configuration: Set your paths here ---
source_image_folder = 'C:/Users/Luigi/Downloads/test_images-20250628T161927Z-1-001/test_images'
csv_file_path = 'C:/Users/Luigi/Downloads/test_labels.csv'
# This is where the new, sorted folders will be created.
target_root_folder = 'C:/Users/Luigi/Desktop/code/Thesis/efficientnet-pytorch/assets/val'
# --- End of Configuration ---

# Read the CSV file
# Assumes the first column is the filename and the second is the class label.
try:
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['image_filename', 'label']
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    exit()

print(f"Starting to organize {len(df)} images...")

# Loop over each row in the CSV
for index, row in df.iterrows():
    filename = str(row['image_filename'])
    class_label = str(row['label'])
    
    # 1. Create the full path to the source image
    source_path = os.path.join(source_image_folder, filename)
    
    # 2. Create the target class folder if it doesn't exist
    target_class_folder = os.path.join(target_root_folder, class_label)
    os.makedirs(target_class_folder, exist_ok=True)
    
    # 3. Create the full path for the destination image
    destination_path = os.path.join(target_class_folder, filename)
    
    # 4. Copy the image from the source to the target folder
    if os.path.exists(source_path):
        # Using shutil.copy2 preserves file metadata
        shutil.copy2(source_path, destination_path)
    else:
        print(f"Warning: Image {filename} not found in {source_image_folder}")

print("Done! Your dataset is now organized in:", target_root_folder)