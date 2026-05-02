import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from collections import Counter
import warnings
import subprocess
import tempfile
import shutil
import argparse

warnings.filterwarnings('ignore')

# === CONFIG ===
DATASET_PATH = '//DATASET'
CONVERTED_PATH = '//DATASET_WAV'
TARGET_SAMPLE_RATE = 16000
N_MFCC = 40
OUTPUT_CSV = 'results/enhanced_dataset.csv'
MIN_DURATION = 0.5  # Minimum audio duration in seconds
MAX_DURATION = 10.0  # Maximum audio duration in seconds

# Supported input formats
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.mp4', '.avi', '.mov', '.mkv', '.m4a', '.aac', '.ogg', '.wma']
CONVERSION_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.m4a', '.aac', '.ogg', '.wma']

# Enhanced Emotion Mapping for all datasets
EMOTION_MAP = {
    # CREMA-D emotions
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy',
    'NEU': 'neutral', 'SAD': 'sad', 'SUR': 'surprised',

    # EmoDB emotions
    'W': 'angry', 'L': 'boredom', 'E': 'disgust', 'A': 'fearful',
    'F': 'happy', 'T': 'sad', 'N': 'neutral',

    # ESD emotions (folder names)
    'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral',
    'sad': 'sad', 'surprise': 'surprised',

    # IEMOCAP emotions
    'ang': 'angry', 'hap': 'happy', 'neu': 'neutral',
    'sad': 'sad', 'exc': 'happy', 'fru': 'angry',

    # RAVDESS emotions (based on numbering)
    '01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',

    # SAVEE emotions
    'a': 'angry', 'd': 'disgust', 'f': 'fearful', 'h': 'happy',
    'n': 'neutral', 'sa': 'sad', 'su': 'surprised',

    # TESS emotions
    'pleasant_surprise': 'surprised', 'Pleasant_surprise': 'surprised',
    'Surprise': 'surprised', 'fear': 'fearful', 'disgust': 'disgust',

    # General mappings
    'anger': 'angry', 'happiness': 'happy', 'sadness': 'sad',
    'fear': 'fearful', 'surprise': 'surprised', 'contempt': 'disgust'
}


def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_wav(input_path, output_path):
    """Convert audio/video file to WAV using FFmpeg"""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FFmpeg command to extract audio and convert to WAV
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', str(TARGET_SAMPLE_RATE),  # Set sample rate
            '-ac', '1',  # Mono
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            '-y',  # Overwrite output file
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stderr

    except Exception as e:
        return False, str(e)


def batch_convert_files(input_dir, output_dir):
    """Convert all supported files to WAV format"""
    print("🔄 Starting batch conversion...")

    if not check_ffmpeg():
        print("❌ FFmpeg not found. Please install FFmpeg first.")
        print("   Download from: https://ffmpeg.org/download.html")
        return False

    # Find files to convert
    files_to_convert = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in CONVERSION_FORMATS:
                input_path = os.path.join(root, file)

                # Create corresponding output path
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_path = os.path.splitext(output_path)[0] + '.wav'

                files_to_convert.append((input_path, output_path))

    if not files_to_convert:
        print("ℹ️  No files found that need conversion.")
        return True

    print(f"Found {len(files_to_convert)} files to convert")

    # Convert files
    successful = 0
    failed = 0

    for input_path, output_path in tqdm(files_to_convert, desc="Converting files"):
        # Skip if output already exists
        if os.path.exists(output_path):
            successful += 1
            continue

        success, error = convert_to_wav(input_path, output_path)

        if success:
            successful += 1
        else:
            failed += 1
            print(f"❌ Failed to convert {os.path.basename(input_path)}: {error}")

    print(f"✅ Conversion complete! Successful: {successful}, Failed: {failed}")
    return failed == 0


def copy_existing_audio_files(input_dir, output_dir):
    """Copy existing WAV, MP3, FLAC files to output directory"""
    print("📁 Copying existing audio files...")

    existing_formats = ['.wav', '.mp3', '.flac']
    files_copied = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in existing_formats:
                input_path = os.path.join(root, file)

                # Create corresponding output path
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                # Skip if output already exists
                if os.path.exists(output_path):
                    continue

                # Create output directory and copy file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(input_path, output_path)
                files_copied += 1

    print(f"📁 Copied {files_copied} existing audio files")


def load_audio_with_conversion(file_path, use_temp_conversion=False):
    """Load audio file, converting to WAV if necessary"""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.wav', '.mp3', '.flac']:
        # Directly supported by librosa
        try:
            return librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    elif file_ext in CONVERSION_FORMATS and use_temp_conversion:
        # Convert to temporary WAV file
        temp_dir = tempfile.mkdtemp()
        temp_wav = os.path.join(temp_dir, 'temp_audio.wav')

        success, error = convert_to_wav(file_path, temp_wav)
        if success:
            try:
                y, sr = librosa.load(temp_wav, sr=TARGET_SAMPLE_RATE)
                # Clean up temp file
                os.remove(temp_wav)
                os.rmdir(temp_dir)
                return y, sr
            except Exception as e:
                print(f"Error loading converted file: {e}")
                return None, None
        else:
            print(f"Error converting {file_path}: {error}")
            return None, None

    else:
        print(f"Unsupported format or conversion disabled: {file_ext}")
        return None, None


# Dataset-specific emotion extraction functions
def extract_emotion_crema_d(filename):
    """Extract emotion from CREMA-D filename: 1001_DFA_ANG_XX.wav"""
    parts = filename.split('_')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, None)
    return None


def extract_emotion_emodb(filename):
    """Extract emotion from EmoDB filename: 03a01Wa.wav"""
    if len(filename) >= 6:
        emotion_code = filename[5].upper()  # 6th character
        return EMOTION_MAP.get(emotion_code, None)
    return None


def extract_emotion_esd(filepath):
    """Extract emotion from ESD folder structure"""
    path_parts = filepath.split(os.sep)
    for part in path_parts:
        if part.lower() in ['angry', 'happy', 'neutral', 'sad', 'surprise']:
            return EMOTION_MAP.get(part.lower(), None)
    return None


def extract_emotion_iemocap(filename):
    """Extract emotion from IEMOCAP filename or use dataset info"""
    filename_lower = filename.lower()
    for emotion in ['ang', 'hap', 'neu', 'sad', 'exc', 'fru']:
        if emotion in filename_lower:
            return EMOTION_MAP.get(emotion, None)
    return None


def extract_emotion_ravdess(filename):
    """Extract emotion from RAVDESS filename: 03-01-06-01-02-01-12.wav"""
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, None)
    return None


def extract_emotion_savee(filename):
    """Extract emotion from SAVEE filename: DC_a01.wav"""
    if '_' in filename:
        emotion_part = filename.split('_')[1]
        if len(emotion_part) >= 2:
            emotion_code = emotion_part[:2] if emotion_part[:2] == 'sa' else emotion_part[0]
            return EMOTION_MAP.get(emotion_code, None)
    return None


def extract_emotion_tess(filename):
    """Extract emotion from TESS filename"""
    filename_lower = filename.lower()
    for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'pleasant_surprise', 'surprise']:
        if emotion in filename_lower:
            return EMOTION_MAP.get(emotion, None)
    return None


def extract_emotion_meld(filepath):
    """Extract emotion from MELD based on folder structure or filename"""
    path_parts = filepath.split(os.sep)
    filename = os.path.basename(filepath).lower()

    # Check common emotion words in filename or path
    for emotion in ['angry', 'happy', 'sad', 'neutral', 'surprise', 'fear', 'disgust']:
        if emotion in filename or any(emotion in part.lower() for part in path_parts):
            return EMOTION_MAP.get(emotion, None)
    return None


def extract_emotion(file_path):
    """Master emotion extraction function"""
    filename = os.path.basename(file_path)
    filepath_lower = file_path.lower()

    # Determine dataset based on path
    if 'crema' in filepath_lower:
        return extract_emotion_crema_d(filename)
    elif 'emodb' in filepath_lower:
        return extract_emotion_emodb(filename)
    elif 'esd' in filepath_lower:
        return extract_emotion_esd(file_path)
    elif 'iemocap' in filepath_lower:
        return extract_emotion_iemocap(filename)
    elif 'ravdess' in filepath_lower:
        return extract_emotion_ravdess(filename)
    elif 'savee' in filepath_lower:
        return extract_emotion_savee(filename)
    elif 'tess' in filepath_lower:
        return extract_emotion_tess(filename)
    elif 'meld' in filepath_lower:
        return extract_emotion_meld(file_path)

    # Fallback: try general extraction
    filename_lower = filename.lower()
    for key, label in EMOTION_MAP.items():
        if key.lower() in filename_lower:
            return label

    return None


def extract_features(file_path, use_temp_conversion=False):
    """Extract comprehensive audio features"""
    try:
        # Load audio
        y, sr = load_audio_with_conversion(file_path, use_temp_conversion)

        if y is None or sr is None:
            return None

        # Check duration
        duration = len(y) / sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            return None

        # Normalize audio
        y = librosa.util.normalize(y)

        # Extract features
        features = {}

        # 1. MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zero_crossing_rate)
        features['zcr_std'] = np.std(zero_crossing_rate)

        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)

        # 4. Mel-frequency cepstral coefficients
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        features['mel_mean'] = np.mean(mel_spectrogram, axis=1)
        features['mel_std'] = np.std(mel_spectrogram, axis=1)

        # 5. Tempo and rhythm
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        # Flatten all features
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.tolist())
            else:
                feature_vector.append(value)

        return np.array(feature_vector)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def preprocess_all_audio(dataset_path, use_temp_conversion=False):
    """Main preprocessing function"""
    data = []
    emotion_counts = Counter()
    failed_files = []
    format_counts = Counter()

    print("🔍 Scanning for audio/video files...")

    # Collect all supported files
    audio_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                audio_files.append(os.path.join(root, file))
                format_counts[file_ext] += 1

    print(f"Found {len(audio_files)} supported files")
    print("File format distribution:")
    for fmt, count in format_counts.most_common():
        print(f"  {fmt}: {count}")

    # Process each file
    for file_path in tqdm(audio_files, desc="Processing files"):
        emotion = extract_emotion(file_path)

        if emotion:
            features = extract_features(file_path, use_temp_conversion)
            if features is not None:
                # Get dataset name from path
                dataset_name = 'unknown'
                path_parts = file_path.split(os.sep)
                for part in path_parts:
                    if part.lower() in ['crema-d', 'emodb', 'esd', 'iemocap', 'ravdess', 'savee', 'tess', 'meld']:
                        dataset_name = part.lower()
                        break

                row = [file_path, dataset_name, emotion] + features.tolist()
                data.append(row)
                emotion_counts[emotion] += 1
            else:
                failed_files.append((file_path, "Feature extraction failed"))
        else:
            failed_files.append((file_path, "Emotion not recognized"))

    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total processed files: {len(data)}")
    print(f"Failed files: {len(failed_files)}")
    print("\nEmotion distribution:")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion}: {count}")

    if failed_files:
        print(f"\n⚠️  First 10 failed files:")
        for file_path, reason in failed_files[:10]:
            print(f"  {os.path.basename(file_path)}: {reason}")

    return data


def save_dataset(data, output_path):
    """Save processed data to CSV"""
    if not data:
        print("❌ No data to save!")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"💾 Saving {len(data)} samples to {output_path}...")

    # Create column names
    feature_count = len(data[0]) - 3  # Subtract filepath, dataset, emotion
    feature_cols = [f'feature_{i + 1}' for i in range(feature_count)]
    columns = ['filepath', 'dataset', 'emotion'] + feature_cols

    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])}... (and {len(df.columns) - 5} more)")


def main():
    """Main execution function with command line options"""
    parser = argparse.ArgumentParser(description='Unified Audio Preprocessing for Emotion Recognition')
    parser.add_argument('--mode', choices=['convert', 'process', 'both'], default='both',
                        help='Mode: convert only, process only, or both (default: both)')
    parser.add_argument('--input-dir', default=DATASET_PATH, help='Input directory path')
    parser.add_argument('--output-dir', default=CONVERTED_PATH, help='Output directory for converted files')
    parser.add_argument('--temp-conversion', action='store_true',
                        help='Use temporary conversion (slower but saves disk space)')

    args = parser.parse_args()

    print("🚀 Unified Audio Preprocessing Started...")
    print(f"Mode: {args.mode}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"MFCC coefficients: {N_MFCC}")
    print(f"Duration limits: {MIN_DURATION}s - {MAX_DURATION}s")
    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return

    # Mode 1: Convert files
    if args.mode in ['convert', 'both']:
        print("\n" + "=" * 50)
        print("STEP 1: CONVERTING FILES TO WAV")
        print("=" * 50)

        # Batch convert files that need conversion
        conversion_success = batch_convert_files(args.input_dir, args.output_dir)

        # Copy existing audio files
        copy_existing_audio_files(args.input_dir, args.output_dir)

        if not conversion_success:
            print("⚠️  Some files failed to convert. Continuing with available files...")

    # Mode 2: Process files
    if args.mode in ['process', 'both']:
        print("\n" + "=" * 50)
        print("STEP 2: PROCESSING FILES FOR TRAINING")
        print("=" * 50)

        # Determine source directory
        if args.mode == 'both' and os.path.exists(args.output_dir):
            source_dir = args.output_dir
            print(f"Processing converted files from: {source_dir}")
        else:
            source_dir = args.input_dir
            print(f"Processing original files from: {source_dir}")
            if args.temp_conversion:
                print("Using temporary conversion for unsupported formats...")

        # Process all audio files
        processed_data = preprocess_all_audio(source_dir, args.temp_conversion)

        # Save results
        if processed_data:
            save_dataset(processed_data, OUTPUT_CSV)

            # Additional analysis
            df = pd.DataFrame(processed_data)
            print("\n📈 Final Dataset Analysis:")
            print(f"Total samples: {len(df)}")
            print(f"Emotion distribution:")
            emotion_counts = df.iloc[:, 2].value_counts()  # emotion column
            for emotion, count in emotion_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")

            print(f"\n🎉 Processing completed successfully!")
            print(f"Dataset saved as: {OUTPUT_CSV}")
        else:
            print("❌ No valid data was processed. Please check your dataset structure and file formats.")

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()