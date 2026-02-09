import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Global explainer instance to avoid re-init overhead
explainer = lime_image.LimeImageExplainer()

def get_explanation(model, img_path, output_dir, file_basename):
    """
    Generates a LIME explanation image (superpixels) for the given image path 
    using the provided model.
    """
    try:
        # Preprocess for ResNet50V2 (Must match app.py logic)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        # Expand dims for prediction
        img_batch = np.expand_dims(img_array, axis=0)
        # Preprocess (resnet_v2 specific)
        processed_img = tf.keras.applications.resnet_v2.preprocess_input(img_batch.copy())

        # Prediction function wrapper for LIME
        # LIME passes a batch of images
        def predict_fn(images):
            # images come in as standard arrays, need preprocessing
            # Note: LIME might pass float64 images, we ensure float32 for TF
            imgs = tf.keras.applications.resnet_v2.preprocess_input(images.astype('float32'))
            return model.predict(imgs)

        # Generate Explanation
        # num_samples=300 is a tradeoff for speed. Higher = more accurate but slower.
        explanation = explainer.explain_instance(
            img_array.astype('double'), 
            predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=300 
        )

        # Get the top label (the predicted class)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )

        # Create overlay image
        # mark_boundaries returns float 0-1 image. Scale to 255.
        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        out_filename = f"lime_{file_basename}"
        out_path = os.path.join(output_dir, out_filename)
        
        print(f"LIME: Saving explanation to {out_path}...")
        
        # Save using matplotlib
        plt.imsave(out_path, img_boundry)
        print("LIME: Saved successfully.")
        
        return out_filename

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"LIME Error: {e}")
        return None
