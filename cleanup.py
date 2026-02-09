import os
import shutil

def filter_dataset(dataset_path):
    """
    Keep only the specified classes and remove others.
    """
    KEEP_CLASSES = {'Acne', 'Psoriasis', 'Vitiligo', 'Eczema', 'Warts'}
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    subsets = ['Train', 'Test']
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            continue
            
        print(f"\nScanning {subset}...")
        dirs = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        
        for d in dirs:
            if d not in KEEP_CLASSES:
                full_path = os.path.join(subset_path, d)
                print(f"Deleting: {d}")
                try:
                    shutil.rmtree(full_path)
                except Exception as e:
                    print(f"Error deleting {d}: {e}")
            else:
                print(f"Keeping: {d}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, 'SkinDisease')
    
    # Confirm prompt (just in case run manually)
    print("WARNING: This script will DELETE folders not in the 'keep list'.")
    # For automation, we skip interactive input, assuming the user already approved the plan.
    
    filter_dataset(dataset_root)
    print("\nFiltering Complete.")
