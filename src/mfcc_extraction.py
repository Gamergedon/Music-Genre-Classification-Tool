import os
import numpy as np
import librosa

# Paths (to match Jonny's structure)
SOURCE_AUDIO_PATH = 'data/genres'
MFCC_SAVE_DIR = 'data/mfcc'

# MFCC coefficient
N_MFCC = 20

# Genre label mapping
GENRE_TO_LABEL = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

def extract_mean_mfcc(file_path, n_mfcc=N_MFCC):
    """Load audio file -> compute MFCC -> return mean vector."""
    try:
        # load an audio file 
        y, sr = librosa.load(file_path, sr=22050)
        # MFCC matrix
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # mean for vector
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_mfcc_dataset():
    X = []
    y = []

    # Keeping terminal output format consistent
    print("----- Starting MFCC Extraction -----")

    for genre in GENRE_TO_LABEL.keys():
        genre_dir = os.path.join(SOURCE_AUDIO_PATH, genre)

        # Edge case: the folder doesn't exist
        if not os.path.exists(genre_dir):
            print(f"Warning: missing genre folder '{genre_dir}'")
            continue

        # Parse all files in the 'genre' directory.
        for file in os.listdir(genre_dir):
            if file.endswith('.au'):
                file_path = os.path.join(genre_dir, file)
                label = GENRE_TO_LABEL[genre]

                print(f"Processing: {genre}/{file}")

                # Extract feature vector
                mfcc_vec = extract_mean_mfcc(file_path)

                # Store if successful
                if mfcc_vec is not None:
                    X.append(mfcc_vec)
                    y.append(label)

    # Convert each list to an array
    X = np.array(X)
    y = np.array(y)

    # If theres no folder, make one
    os.makedirs(MFCC_SAVE_DIR, exist_ok=True)

    # Save the dataset so it doesnt take forever to test 
    np.save(os.path.join(MFCC_SAVE_DIR, "X_mfcc.npy"), X)
    np.save(os.path.join(MFCC_SAVE_DIR, "y_mfcc.npy"), y)
    #                       ^^^^
    # (feel free to comment this out if you want to avoid caching it on your end)

    print("----- MFCC Extraction Complete -----")
    print(f"Saved X_mfcc.npy with shape {X.shape}")
    print(f"Saved y_mfcc.npy with shape {y.shape}")

    return X, y

if __name__ == "__main__":
    build_mfcc_dataset()
