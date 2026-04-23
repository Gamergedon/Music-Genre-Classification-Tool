import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import splitfolders

# Path for folder containing .au files
SOURCE_AUDIO_PATH = 'data/genres'
# Path for generated spectrogram images
SPECTROGRAM_PATH = 'data/spectograms_all'
# Path for final split datasets
FINAL_SPLIT_PATH = 'data/dataset_split'

# Some default image settings
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Method for loading audio, calc Mel spectrogram, and save as image without axes/whitespaces
def create_spectrogram(file_path, save_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(mel_signal, ref=np.max)

        plt.figure(figsize=(10,4))
        plt.axis('off')
        librosa.display.specshow(spectrogram, sr=sr)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Method for iterating through all genre folders and processing .au files
def process_all_genres():
    print("----- Starting Spectogram Generation -----")
    if not os.path.exists(SOURCE_AUDIO_PATH):
        print(f"Error: Source path '{SOURCE_AUDIO_PATH}' not found.")
        return
    for root, dirs, files in os.walk(SOURCE_AUDIO_PATH):
        for file in files:
            if file.endswith('.au'):
                file_path = os.path.join(root, file)
                genre = os.path.basename(root)
                genre_out_path = os.path.join(SPECTROGRAM_PATH, genre)
                os.makedirs(genre_out_path, exist_ok=True)
                out_file = file.replace('.au', '.png')
                save_path = os.path.join(genre_out_path, out_file)
                if not os.path.exists(save_path):
                    print(f"Processing: {genre}/{file}")
                    create_spectrogram(file_path, save_path)
                else:
                    print(f"Skipping {file} (already exists)")
    print("----- Spectrogram Generation Complete -----")

# Method for splitting images into train/test/val folders
# Ratio is 80% train, 10% val, and 10% test
def split_data():
    print("----- Splitting Data Into {FINAL_SPLIT_PATH} -----")
    splitfolders.ratio(SPECTROGRAM_PATH, output=FINAL_SPLIT_PATH, seed=42, ratio=(.8, .1, .1), group_prefix=None)
    print("----- Data Split Complete -----")

if __name__ == "__main__":
    process_all_genres()
    split_data()
    