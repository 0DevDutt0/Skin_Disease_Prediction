import cv2
import os

def resize_images(dataset_path, size=(224, 224)):
    """
    Resizes all images in the dataset to the specified size.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Starting resize process for: {dataset_path}")
    print(f"Target size: {size}")

    count_processed = 0
    count_errors = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.dib', '.jp2')):
                filepath = os.path.join(root, file)
                
                try:
                    # Read image
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"Warning: Could not read {filepath}")
                        count_errors += 1
                        continue
                    
                    # Check if resize is needed
                    if img.shape[0] != size[0] or img.shape[1] != size[1]:
                        # Resize
                        img_resized = cv2.resize(img, size)
                        
                        # Overwrite
                        cv2.imwrite(filepath, img_resized)
                        count_processed += 1
                        
                        if count_processed % 100 == 0:
                            print(f"Processed {count_processed} images...")
                            
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    count_errors += 1

    print("\n--- Resize Completed ---")
    print(f"Total images resized: {count_processed}")
    print(f"Total errors: {count_errors}")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'SkinDisease', 'Train')
    test_dir = os.path.join(script_dir, 'SkinDisease', 'Test')
    
    resize_images(train_dir)
    resize_images(test_dir)
