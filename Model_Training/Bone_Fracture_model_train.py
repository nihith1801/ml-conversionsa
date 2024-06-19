# bone_fracture_model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from Data_Loading.loader import load_data, load_model_and_labels, combine_sample_data, model_paths
from Data_Loading.data_sampler import combine_sample_data
import numpy as np

# Load data
data_dir = 'path/to/bone_fracture_dataset'
train_gen, val_gen = load_data(data_dir, binary=True)

# Convert generators to numpy arrays for resampling
X_train, y_train = next(train_gen)
X_val, y_val = next(val_gen)

# Perform combined over- and under-sampling
X_train_combined, y_train_combined = combine_sample_data(X_train, y_train)

# Load the pre-trained ResNet50 model and fine-tune
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_combined, y_train_combined, validation_data=(X_val, y_val), epochs=50)

# Save the model
model.save('my_model/Bone_Fracture.h5')

# Save the labels
labels = train_gen.class_indices
with open('my_model/bone_fracture_labels.txt', 'w') as f:
    for label, index in labels.items():
        f.write(f"{index} {label}\n")
