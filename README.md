# 🩺 Skin Disease Prediction System

> **AI-powered dermatological diagnosis with explainable predictions using deep learning and LIME visualization**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

The **Skin Disease Prediction System** is a production-ready web application that leverages deep learning to classify skin conditions from images. Built with a fine-tuned **ResNet50V2** architecture, the system achieves **90%+ accuracy** across five common dermatological conditions.

### What Problem Does It Solve?

Early detection of skin diseases is critical for effective treatment, but access to dermatologists can be limited. This system provides:
- **Instant preliminary diagnosis** from uploaded skin images
- **Explainable AI** using LIME (Local Interpretable Model-agnostic Explanations) to visualize which regions influenced the prediction
- **Accessible healthcare screening** that can be deployed in remote or underserved areas

### Why It Matters

- **For Healthcare**: Reduces diagnostic time and assists medical professionals with AI-powered insights
- **For Patients**: Provides immediate feedback and encourages early medical consultation
- **For Recruiters**: Demonstrates end-to-end ML engineering skills—from data preprocessing and model training to deployment and explainability

---

## ✨ Features

- ✅ **Multi-class Classification**: Detects 5 skin conditions (Acne, Eczema, Psoriasis, Vitiligo, Warts)
- ✅ **High Accuracy**: 90%+ validation accuracy using fine-tuned ResNet50V2
- ✅ **Explainable AI**: LIME-based visual explanations showing which image regions influenced predictions
- ✅ **Real-time Inference**: Flask-based REST API for instant predictions
- ✅ **Image Preprocessing**: Automated resizing, normalization, and augmentation pipeline
- ✅ **Robust Training**: Two-phase training (feature extraction + fine-tuning) with callbacks (EarlyStopping, ReduceLROnPlateau)
- ✅ **Web Interface**: Clean, responsive UI for image upload and result visualization
- ✅ **Image Metrics**: Displays sharpness and color analysis alongside predictions

---

## 🛠️ Tech Stack

### Core Technologies
- **Deep Learning**: TensorFlow 2.x, Keras
- **Model Architecture**: ResNet50V2 (pre-trained on ImageNet, fine-tuned)
- **Explainability**: LIME (Local Interpretable Model-agnostic Explanations)
- **Backend**: Flask (Python web framework)
- **Image Processing**: OpenCV, NumPy, scikit-image

### Development Tools
- **Data Augmentation**: ImageDataGenerator (rotation, zoom, flip, shift)
- **Visualization**: Matplotlib
- **Model Optimization**: Adam optimizer, categorical cross-entropy loss
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

---

## 📦 Installation / Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

### Step 1: Clone the Repository
```bash
git clone https://github.com/0DevDutt0/skin-disease-prediction.git
cd skin-disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
Organize your dataset in the following structure:
```
SkinDisease/
├── Train/
│   ├── Acne/
│   ├── Eczema/
│   ├── Psoriasis/
│   ├── Vitiligo/
│   └── Warts/
└── Test/
    ├── Acne/
    ├── Eczema/
    ├── Psoriasis/
    ├── Vitiligo/
    └── Warts/
```

### Step 5: Train the Model (Optional)
If you want to retrain the model from scratch:
```bash
python train.py
```
This will:
- Load and preprocess training data
- Fine-tune ResNet50V2 in two phases
- Save the trained model as `skin_disease_model_v2.h5`
- Generate `class_indices.json` for label mapping

### Step 6: Run the Application
```bash
python app.py
```
The application will start at `http://localhost:5000`

---

## 🚀 Usage Examples

### Web Interface
1. Open your browser and navigate to `http://localhost:5000`
2. Click "Upload Image" and select a skin image (JPG, PNG, etc.)
3. Click "Analyze" to get predictions
4. View results including:
   - Predicted disease class
   - Confidence score (%)
   - LIME explanation overlay showing influential regions
   - Image quality metrics (sharpness, color analysis)

### API Endpoint
You can also use the REST API directly:

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@path/to/skin_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "prediction": "Eczema",
  "confidence": 94.23,
  "explanation_url": "/static/explanations/lime_skin_image.jpg",
  "metrics": {
    "sharpness": 1234.56,
    "avg_red": 178.45,
    "avg_green": 142.32,
    "avg_blue": 135.67
  }
}
```

### Programmatic Inference
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import json

# Load model and class labels
model = load_model('skin_disease_model_v2.h5')
with open('class_indices.json', 'r') as f:
    indices = json.load(f)
    class_labels = {v: k for k, v in indices.items()}

# Preprocess image
img = image.load_img('test_image.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.resnet_v2.preprocess_input(x)

# Predict
preds = model.predict(x)
predicted_class = class_labels[np.argmax(preds[0])]
confidence = np.max(preds[0]) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
```

---

## 📸 Demo Visuals

### Application Interface
*Include a screenshot of the web interface showing the upload button and results panel*

![Web Interface](static/demo_interface.png)

### LIME Explanation Example
*Show a before/after comparison: original image and LIME overlay highlighting influential regions*

![LIME Explanation](static/demo_lime_explanation.png)

### Prediction Results
*Screenshot showing prediction output with confidence score and metrics*

![Prediction Results](static/demo_results.png)

> **Note**: Add actual screenshots to the `static/` folder and update the paths above.

---

## 📊 Results / Metrics

### Model Performance
- **Architecture**: ResNet50V2 (fine-tuned)
- **Training Strategy**: Two-phase training (frozen base → fine-tuning top 50 layers)
- **Validation Accuracy**: **90%+**
- **Classes**: 5 (Acne, Eczema, Psoriasis, Vitiligo, Warts)
- **Input Size**: 224×224 RGB images
- **Training Data**: Augmented with rotation, zoom, flip, shift

### Training Configuration
- **Phase 1**: Train classification head (5 epochs, lr=0.001)
- **Phase 2**: Fine-tune top layers (20 epochs, lr=1e-5)
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.2, patience=3)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy

### Explainability
- **Method**: LIME (Local Interpretable Model-agnostic Explanations)
- **Visualization**: Superpixel-based boundary overlays
- **Samples**: 300 perturbed images per explanation
- **Top Features**: Highlights 5 most influential regions

---

## 💡 Challenges & Learnings

### Technical Challenges
1. **Class Imbalance**: Some skin conditions had fewer training samples
   - **Solution**: Applied aggressive data augmentation (rotation, zoom, flip) to balance dataset
   
2. **Model Overfitting**: Initial training showed high training accuracy but poor validation
   - **Solution**: Implemented two-phase fine-tuning with dropout (0.5) and learning rate scheduling

3. **Explainability Performance**: LIME explanations were slow for real-time inference
   - **Solution**: Optimized `num_samples=300` as a speed/accuracy tradeoff; cached explainer instance

4. **Preprocessing Consistency**: Mismatch between training and inference preprocessing caused accuracy drops
   - **Solution**: Standardized on `tf.keras.applications.resnet_v2.preprocess_input` across all pipelines

### Key Learnings
- **Transfer Learning**: Fine-tuning pre-trained models significantly reduces training time and improves accuracy
- **Explainable AI**: LIME provides crucial transparency for medical applications, building user trust
- **Production Readiness**: Proper error handling, file cleanup, and API design are essential for deployment
- **Callback Strategies**: EarlyStopping and ReduceLROnPlateau prevent overfitting and optimize training efficiency

---

## 🔮 Future Enhancements

### Short-term Improvements
- [ ] **Expand Disease Coverage**: Add more skin conditions (melanoma, rosacea, dermatitis)
- [ ] **Mobile App**: Develop iOS/Android app for on-the-go diagnosis
- [ ] **Batch Processing**: Support multiple image uploads for comparative analysis
- [ ] **User Authentication**: Add login system to track user history and predictions

### Long-term Vision
- [ ] **Model Ensemble**: Combine multiple architectures (EfficientNet, Vision Transformer) for improved accuracy
- [ ] **Grad-CAM Integration**: Add alternative explainability method for comparison
- [ ] **Cloud Deployment**: Deploy on AWS/GCP with auto-scaling for production use
- [ ] **Medical Integration**: Partner with dermatology clinics for real-world validation
- [ ] **Multilingual Support**: Internationalize UI for global accessibility
- [ ] **Severity Grading**: Classify not just disease type but also severity levels

---

## 🤝 Contributing Guidelines

Contributions are welcome! To contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and add tests if applicable
4. **Commit with clear messages**: `git commit -m "Add feature: description"`
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Open a Pull Request** with a detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README if adding new functionality

---

## 📄 License

MIT License © 2026 Devdutt S

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 👤 Contact & Author

**Devdutt S**

- 💼 LinkedIn: [linkedin.com/in/devdutts](https://linkedin.com/in/devdutts)
- 📧 Email: devduttshoji123@gmail.com
- 🐙 GitHub: [@0DevDutt0](https://github.com/0DevDutt0)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star!**

*Built with ❤️ for accessible healthcare and AI transparency*

</div>
