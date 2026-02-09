import requests
import os
import shutil

def test_app_predictions():
    url = 'http://127.0.0.1:5000/analyze'
    
    # Generate a dummy image
    dummy_img = 'test_pred.jpg'
    import cv2
    import numpy as np
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(dummy_img, img)

    allowed = {'Acne', 'Psoriasis', 'Vitiligo', 'Eczema (Atopic Dermatitis)', 'Warts'}
    
    print("Testing predictions...")
    try:
        with open(dummy_img, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"Full Response: {data}")
            pred = data.get('prediction')
            print(f"Prediction: {pred}")
            
            if pred in allowed:
                print("PASS: Prediction is in allowed list.")
            else:
                print(f"FAIL: Prediction '{pred}' is NOT in allowed list.")
        else:
            print(f"FAIL: Status code {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(dummy_img):
            os.remove(dummy_img)

if __name__ == "__main__":
    test_app_predictions()
