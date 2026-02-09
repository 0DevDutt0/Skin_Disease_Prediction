import cv2
import os

def check_dimensions(directory):
    print(f"Checking dimensions in: {directory}")
    errors = 0
    checked = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    if h != 224 or w != 224:
                        print(f"Incorrect size: {path} ({w}x{h})")
                        errors += 1
                checked += 1
                if checked >= 50: # Check just 50 images per dir to be fast
                    break
        if checked >= 50:
            break
            
    if errors == 0:
        print("Verification Successful: All checked images are 224x224.")
    else:
        print(f"Verification Failed: {errors} images have incorrect dimensions.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'SkinDisease', 'Train')
    test_dir = os.path.join(script_dir, 'SkinDisease', 'Test')
    
    check_dimensions(train_dir)
    check_dimensions(test_dir)
