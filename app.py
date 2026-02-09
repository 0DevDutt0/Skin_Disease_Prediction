import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Use absolute paths for folders
script_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, 'temp_uploads')
app.config['EXPLAIN_FOLDER'] = os.path.join(script_dir, 'static', 'explanations')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPLAIN_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'dib', 'jp2'}

# Import LIME explainer
import explain

# Load Model
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'skin_disease_model_v2.h5')
INDICES_PATH = os.path.join(script_dir, 'class_indices.json')
model = None
class_labels = {}

def load_inference_model():
    global model, class_labels
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading ResNet50V2 model...")
            model = load_model(MODEL_PATH)
            print("Model loaded.")
        else:
            print("Error: Model file not found.")

        if os.path.exists(INDICES_PATH):
            with open(INDICES_PATH, 'r') as f:
                indices = json.load(f)
                class_labels = {v: k for k, v in indices.items()}
            print(f"Loaded classes: {class_labels}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load on startup
load_inference_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(filepath):
    """
    Analyze using the real model.
    """
    global model
    if model is None:
        return {'success': False, 'error': 'Model not loaded'}

    try:
        # 1. Preprocess for ResNet50V2
        # Resize to 224x224
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # ResNetV2 specific preprocessing (NOT 1./255)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        
        # 2. Predict
        preds = model.predict(x)
        predicted_class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]) * 100)
        predicted_class = class_labels.get(predicted_class_index, 'Unknown')
        
        # 3. Basic Metrics (OpenCV) for display
        img_cv = cv2.imread(filepath)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        avg_color = np.mean(img_cv, axis=(0, 1))

        # 4. Generate LIME Explanation
        explanation_file = explain.get_explanation(
            model, 
            filepath, 
            app.config['EXPLAIN_FOLDER'], 
            os.path.basename(filepath)
        )
        
        return {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'explanation_url': f"/static/explanations/{explanation_file}" if explanation_file else None,
            'metrics': {
                'sharpness': round(laplacian_var, 2),
                'avg_red': round(avg_color[2], 2), # BGR to RGB
                'avg_green': round(avg_color[1], 2),
                'avg_blue': round(avg_color[0], 2)
            }
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print(f"Received request: {request.files}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        print(f"Processing: {file.filename}")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_image(filepath)
        
        try:
            os.remove(filepath)
        except:
            pass
            
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process image'}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Use threaded=False for TensorFlow compatibility in some envs, 
    # but normally threaded=True is fine.
    app.run(debug=True, port=5000)
