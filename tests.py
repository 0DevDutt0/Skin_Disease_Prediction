import requests
import os

def test_analyze():
    url = 'http://127.0.0.1:5000/analyze'
    
    # Find a sample image
    base_dir = r'd:/Prog/Datasets/Project_1/SkinDisease/Train'
    found_image = None
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    found_image = os.path.join(root, file)
                    break
            if found_image:
                break
    
    if not found_image:
        print("No sample image found in dataset. Creating a dummy image for testing.")
        import numpy as np
        import cv2
        dummy_path = 'dummy_test_image.jpg'
        # Create a random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(dummy_path, img)
        found_image = dummy_path

    print(f"Testing with image: {found_image}")
    
    try:
        with open(found_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
        
        if response.status_code == 200 and response.json().get('success'):
            data = response.json()
            print("\nSUCCESS: API is working correctly.")
            print(f"Prediction: {data.get('prediction')}")
            if data.get('explanation_url'):
                print(f"Explanation URL: {data.get('explanation_url')}")
            else:
                print("WARNING: No explanation URL returned.")
        else:
            print("\nFAILURE: API returned unexpected result.")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running?")

if __name__ == "__main__":
    test_analyze()
