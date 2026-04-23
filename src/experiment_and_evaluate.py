import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix

# Experiment configuration
LEARNING_RATE = 0.0001     # Try 0.001, 0.0001, 0.01
DROPOUT_RATE = 0.5        # Try 0.3, 0.5, 0.7
ADD_EXTRA_LAYER = True   # Set to True to make the network deeper
EPOCHS = 15               # Increase if loss is still dropping

DATASET_PATH = 'data/dataset_split'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH = os.path.join(DATASET_PATH, 'val')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32

def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=True
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False
    )
    
    print("--- Loading Test Data (Hidden Set) ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def build_tunable_model(num_classes):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Experimental Block (Added if ADD_EXTRA_LAYER is True)
    if ADD_EXTRA_LAYER:
        print(">>> Adding extra convolutional layer for this experiment")
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    
    # Tunable Dropout
    model.add(layers.Dropout(DROPOUT_RATE))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('final_confusion_matrix.png')
    print("Confusion Matrix saved as 'final_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = load_data()
    num_classes = len(class_names)

    # 2. Build Model (with experiments)
    model = build_tunable_model(num_classes)
    
    # Compile with tunable Learning Rate
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Train
    print(f"\n--- Starting Experiment: LR={LEARNING_RATE}, Dropout={DROPOUT_RATE} ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # 4. Final Evaluation on Test Set
    print("\n--- Evaluating on Hidden Test Set ---")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # 5. Generate Confusion Matrix
    # We need to get predictions for the whole test set
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels from the dataset
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    plot_confusion_matrix(y_true, y_pred, class_names)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))