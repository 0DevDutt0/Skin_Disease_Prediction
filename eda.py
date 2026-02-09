import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_dataset(dataset_path):
    """
    Traverse the dataset, load images, and display stats.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print(f"Processing dataset at: {dataset_path}")

    # Define sub-directories (Train/Test)
    subsets = ['Train', 'Test']
    
    class_stats = {}
    sample_images = []
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            print(f"Warning: Subset '{subset}' not found in {dataset_path}")
            continue
            
        print(f"\nScanning {subset} set...")
        
        # List all classes (subfolders)
        classes = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        classes.sort()
        
        print(f"Found {len(classes)} classes: {', '.join(classes)}")
        
        for class_name in classes:
            class_path = os.path.join(subset_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.dib', '.jp2'))]
            
            count = len(images)
            if class_name not in class_stats:
                class_stats[class_name] = {'Train': 0, 'Test': 0}
            class_stats[class_name][subset] = count
            
            # Load the first image as a sample if we don't have one for this class yet
            if count > 0 and len(sample_images) < 9: # Limit samples for grid
                img_path = os.path.join(class_path, images[0])
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        sample_images.append((class_name, img))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Print Statistics
    print("\n--- Class Statistics ---")
    print(f"{'Class Name':<25} | {'Train':<5} | {'Test':<5} | {'Total':<5}")
    print("-" * 46)
    total_imgs = 0
    for class_name, stats in class_stats.items():
        train_c = stats.get('Train', 0)
        test_c = stats.get('Test', 0)
        total_c = train_c + test_c
        total_imgs += total_c
        print(f"{class_name:<25} | {train_c:<5} | {test_c:<5} | {total_c:<5}")
    print("-" * 46)
    print(f"{'Total':<25} | {'':<5} | {'':<5} | {total_imgs:<5}")

    # Visualize Samples
    if sample_images:
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle('Sample Images from Skin Disease Dataset', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < len(sample_images):
                label, img = sample_images[i]
                ax.imshow(img)
                ax.set_title(label)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        output_file = 'sample_visualization.jpg'
        plt.savefig(output_file)
        print(f"\nSample visualization saved to '{output_file}'")
    else:
        print("\nNo images found for visualization.")

if __name__ == "__main__":
    # Adjust this path based on where the script is located relative to the dataset
    # The user view_file showed: d:/Prog/Datasets/SkinDisease/SkinDisease -> contains Train/Test
    # Script location: d:/Prog/Datasets/SkinDisease/process_skin_data.py
    # So the dataset root is ./SkinDisease
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, 'SkinDisease')
    
    process_dataset(dataset_root)
