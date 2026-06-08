import os
import glob
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
AUDIO_DIR = "raw/FSDK50_dataset/FSD50K.dev_audio"  # Source .wav folder
OUTPUT_DIR = "npy_fsd50k_dev_patches/"    # Destination .npy folder
TARGET_SR = 16000
PATCH_SECONDS = 1.0
MIN_CLIP_DURATION = int(TARGET_SR * PATCH_SECONDS)  # 16000 samples
HOP_SAMPLES = MIN_CLIP_DURATION // 2  # For 50% overlap

os.makedirs(OUTPUT_DIR, exist_ok=True)

def replicate_patch_if_needed(x, min_clip_duration=MIN_CLIP_DURATION):
    if len(x) < min_clip_duration:
        tile_size = (min_clip_duration // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:min_clip_duration]
    return x

def iter_patches(y, min_clip_duration=MIN_CLIP_DURATION, hop_samples=HOP_SAMPLES):
    """Generate 1-second patches from waveform, padding the last one if needed."""
    if len(y) < min_clip_duration:
        y = replicate_patch_if_needed(y, min_clip_duration)
    for start in range(0, max(len(y) - min_clip_duration + 1, 1), hop_samples):
        patch = y[start:start + min_clip_duration]
        # Pad the last patch if it's shorter than MIN_CLIP_DURATION
        patch = replicate_patch_if_needed(patch, min_clip_duration)
        yield start, patch

def process_single_file(file_path):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    patch_count = 0
    
    try:
        # Load and automatically resample to 16kHz mono
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Process each 1-second patch
        for start_sample, patch in iter_patches(y):
            # Compute Mel Spectrogram for this patch
            mel_spec = librosa.feature.melspectrogram(
                y=patch,
                sr=TARGET_SR,
                n_fft=512,
                hop_length=160,
                win_length=480,
                n_mels=96
            )
            
            # Convert to log amplitude (dB scale)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (101 time_frame, 96 n_mels) and Compress to float16
            mel_float16 = mel_db.T.astype(np.float16)
            
            # Save with patch index: filename_000000.npy, filename_001000.npy, etc.
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{file_id}_{start_sample:06d}.npy"
            )
            np.save(output_path, mel_float16)
            patch_count += 1
        
        return True, patch_count
    except Exception as e:
        print(f"\nError processing {file_id}: {str(e)}")
        return False, 0

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
            desc="Extracting 1-sec Mel patches via Librosa"
        ))
    
    total_patches = sum(count for success, count in results if success)
    successful_files = sum(1 for success, _ in results if success)
    print(f"\nSuccessfully processed {successful_files}/{num_files} files.")
    print(f"Generated {total_patches} total 1-second patches.")