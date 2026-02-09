import cv2
import numpy as np
import os
import shutil

def test_formats():
    test_dir = 'format_test_data'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    print(f"Created test directory: {test_dir}")

    # formats to test
    formats = ['.bmp', '.tif', '.webp', '.png', '.jpg']
    
    # Create dummy images
    for fmt in formats:
        # Create a random 100x100 image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filename = f"test_img{fmt}"
        path = os.path.join(test_dir, filename)
        success = cv2.imwrite(path, img)
        if success:
            print(f"Created {filename}")
        else:
            print(f"Failed to create {filename}")

    # Run resize logic
    print("\nRunning resize logic...")
    # Import the resize function from our script
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import resize_images

    resize_images(test_dir)

    # Verify
    print("\nVerifying dimensions...")
    all_passed = True
    for fmt in formats:
        filename = f"test_img{fmt}"
        path = os.path.join(test_dir, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                if h == 224 and w == 224:
                    print(f"PASS: {filename} resized to {w}x{h}")
                else:
                    print(f"FAIL: {filename} is {w}x{h}")
                    all_passed = False
            else:
                print(f"FAIL: Could not read {filename} after resize")
                all_passed = False
        else:
            print(f"FAIL: {filename} missing")
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All formats processed correctly.")
    else:
        print("\nFAILURE: Some formats failed.")

    # Cleanup
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_formats()
