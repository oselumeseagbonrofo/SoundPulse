import os
import glob
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
AUDIO_DIR = "raw/FSDK50_dataset/FSD50K.eval_audio"  # Source .wav folder
OUTPUT_DIR = "npy_fsd50k_eval/"    # Destination .npy folder
TARGET_SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_file(file_path):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.npy")
    
    # Skip if already processed (allows safe script resumption)
    if os.path.exists(output_path):
        return True
        
    try:
        # Load and automatically resample to 16kHz mono
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Compute Mel Spectrogram (Matching FSD50K baseline settings)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=TARGET_SR, 
            n_fft=1024, 
            hop_length=512, 
            win_length=1024, 
            n_mels=96
        )
        
        # Convert to log amplitude (dB scale)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Compress to float16 to save disk space and I/O overhead
        mel_float16 = mel_db.astype(np.float16)
        
        # Save to disk
        np.save(output_path, mel_float16)
        return True
    except Exception as e:
        print(f"\nError processing {file_id}: {str(e)}")
        return False

if __name__ == "__main__":
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    print(os.path.dirname(os.path.abspath(__file__)))
    num_files = len(audio_files)
    print(f"Found {num_files} files. Processing with {cpu_count()} CPU workers...")

    # Parallelize extraction across all available CPU cores
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, audio_files), 
            total=num_files, 
            desc="Extracting Mels via Librosa"
        ))
        
    print(f"\nSuccessfully processed {sum(results)}/{num_files} files.")