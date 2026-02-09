import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

def train_fine_tuned_model():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'SkinDisease', 'Train')
    test_dir = os.path.join(script_dir, 'SkinDisease', 'Test')
    
    # Parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    # Preprocessing function required for ResNet50V2
    preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input
    
    # Data Generators with Aggressive Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    
    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    CLASSES = len(train_generator.class_indices)
    
    # Base Model (ResNet50V2)
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Phase 1: Train Head only
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # Consolidate features
    predictions = Dense(CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Phase 1: Training Head...")
    history1 = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator
    )
    
    # Phase 2: Fine Tuning
    print("\nPhase 2: Fine Tuning...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    # ResNet50V2 has 190 layers. Unfreeze top ~50.
    fine_tune_at = 140
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5), # Low learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    
    history2 = model.fit(
        train_generator,
        epochs=20, # More epochs
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr]
    )
    
    # Save
    model_path = os.path.join(script_dir, 'skin_disease_model_v2.h5')
    model.save(model_path)
    print(f"Verified Model saved to {model_path}")
    
    # Save Class Indices (just in case they changed sequence)
    import json
    class_indices = train_generator.class_indices
    indices_path = os.path.join(script_dir, 'class_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(class_indices, f)

if __name__ == "__main__":
    train_fine_tuned_model()
