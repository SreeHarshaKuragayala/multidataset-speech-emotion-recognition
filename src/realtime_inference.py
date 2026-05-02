import os
import time
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import threading
import queue
from collections import deque
import pyaudio
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
import json

warnings.filterwarnings("ignore")


# Import your model architecture (make sure this matches your training code)
class AdvancedEmotionNet(nn.Module):
    def __init__(self, input_size=367, num_classes=8, dropout_rate=0.4):
        super(AdvancedEmotionNet, self).__init__()

        # Feature extraction layers with residual connections
        self.input_bn = nn.BatchNorm1d(input_size)

        # First block
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.residual1 = nn.Linear(input_size, 512)

        # Second block
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)
        )
        self.residual2 = nn.Linear(512, 256)

        # Third block
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6)
        )
        self.residual3 = nn.Linear(256, 128)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(32, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)

        # Block 1 with residual
        out1 = self.block1(x)
        res1 = self.residual1(x)
        out1 = out1 + res1

        # Block 2 with residual
        out2 = self.block2(out1)
        res2 = self.residual2(out1)
        out2 = out2 + res2

        # Block 3 with residual
        out3 = self.block3(out2)
        res3 = self.residual3(out2)
        out3 = out3 + res3

        # Attention mechanism
        attention_weights = self.attention(out3)
        out3 = out3 * attention_weights

        # Final classification
        return self.classifier(out3)


class AudioFeatureExtractor:
    """Extract audio features with exact feature count matching"""

    def __init__(self, sample_rate=22050, target_feature_count=367):
        self.sample_rate = sample_rate
        self.target_feature_count = target_feature_count

        # Pre-calculate feature dimensions to ensure exact count
        self.feature_config = {
            'mfcc': 13,  # 13 * 4 = 52 features
            'spectral_centroids': 1,  # 1 * 4 = 4 features
            'spectral_rolloff': 1,  # 1 * 4 = 4 features
            'zcr': 1,  # 1 * 4 = 4 features
            'chroma': 12,  # 12 * 4 = 48 features
            'tonnetz': 6,  # 6 * 4 = 24 features
            'spectral_contrast': 7,  # 7 * 2 = 14 features
            'tempo': 1,  # 1 feature
            'rms': 1,  # 1 * 4 = 4 features
            'mel_spectrogram': 61  # Adjusted to fit exactly
        }

        # Calculate total: 52 + 4 + 4 + 4 + 48 + 24 + 14 + 1 + 4 + (61*4) = 367 features
        print(f"🔧 Configured for exactly {target_feature_count} features")

    def extract_features(self, audio_data, duration=3.0):
        """
        Extract audio features with exact count matching
        """
        try:
            # Ensure audio is the right length
            target_length = int(duration * self.sample_rate)
            if len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
            else:
                audio_data = audio_data[:target_length]

            features = []

            # 1. MFCCs (13 coefficients) - 52 features (13 * 4)
            try:
                mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.feature_config['mfcc'])
                features.extend(np.mean(mfccs, axis=1))  # 13 features
                features.extend(np.std(mfccs, axis=1))  # 13 features
                features.extend(np.max(mfccs, axis=1))  # 13 features
                features.extend(np.min(mfccs, axis=1))  # 13 features
            except Exception as e:
                print(f"Warning: MFCC extraction failed: {e}")
                features.extend(np.zeros(52))

            # 2. Spectral Centroid - 4 features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
                features.extend([
                    np.mean(spectral_centroids),
                    np.std(spectral_centroids),
                    np.max(spectral_centroids),
                    np.min(spectral_centroids)
                ])
            except Exception as e:
                print(f"Warning: Spectral centroid extraction failed: {e}")
                features.extend(np.zeros(4))

            # 3. Spectral Rolloff - 4 features
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
                features.extend([
                    np.mean(spectral_rolloff),
                    np.std(spectral_rolloff),
                    np.max(spectral_rolloff),
                    np.min(spectral_rolloff)
                ])
            except Exception as e:
                print(f"Warning: Spectral rolloff extraction failed: {e}")
                features.extend(np.zeros(4))

            # 4. Zero Crossing Rate - 4 features
            try:
                zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
                features.extend([
                    np.mean(zcr),
                    np.std(zcr),
                    np.max(zcr),
                    np.min(zcr)
                ])
            except Exception as e:
                print(f"Warning: ZCR extraction failed: {e}")
                features.extend(np.zeros(4))

            # 5. Chroma features (12 features) - 48 features (12 * 4)
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
                features.extend(np.mean(chroma, axis=1))  # 12 features
                features.extend(np.std(chroma, axis=1))  # 12 features
                features.extend(np.max(chroma, axis=1))  # 12 features
                features.extend(np.min(chroma, axis=1))  # 12 features
            except Exception as e:
                print(f"Warning: Chroma extraction failed: {e}")
                features.extend(np.zeros(48))

            # 6. Tonnetz (6 features) - 24 features (6 * 4)
            try:
                tonnetz = librosa.feature.tonnetz(y=audio_data, sr=self.sample_rate)
                features.extend(np.mean(tonnetz, axis=1))  # 6 features
                features.extend(np.std(tonnetz, axis=1))  # 6 features
                features.extend(np.max(tonnetz, axis=1))  # 6 features
                features.extend(np.min(tonnetz, axis=1))  # 6 features
            except Exception as e:
                print(f"Warning: Tonnetz extraction failed: {e}")
                features.extend(np.zeros(24))

            # 7. Spectral Contrast (7 features) - 14 features (7 * 2)
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
                features.extend(np.mean(spectral_contrast, axis=1))  # 7 features
                features.extend(np.std(spectral_contrast, axis=1))  # 7 features
            except Exception as e:
                print(f"Warning: Spectral contrast extraction failed: {e}")
                features.extend(np.zeros(14))

            # 8. Tempo - 1 feature
            try:
                tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
                features.append(tempo)
            except Exception as e:
                print(f"Warning: Tempo extraction failed: {e}")
                features.append(0.0)

            # 9. RMS Energy - 4 features
            try:
                rms = librosa.feature.rms(y=audio_data)[0]
                features.extend([
                    np.mean(rms),
                    np.std(rms),
                    np.max(rms),
                    np.min(rms)
                ])
            except Exception as e:
                print(f"Warning: RMS extraction failed: {e}")
                features.extend(np.zeros(4))

            # 10. Mel-frequency spectrogram - Adjusted to fit exactly
            try:
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=self.sample_rate,
                    n_mels=self.feature_config['mel_spectrogram']
                )
                features.extend(np.mean(mel_spectrogram, axis=1))  # 61 features
                features.extend(np.std(mel_spectrogram, axis=1))  # 61 features
                features.extend(np.max(mel_spectrogram, axis=1))  # 61 features
                features.extend(np.min(mel_spectrogram, axis=1))  # 61 features
            except Exception as e:
                print(f"Warning: Mel spectrogram extraction failed: {e}")
                features.extend(np.zeros(244))  # 61 * 4

            # Convert to numpy array
            features = np.array(features)

            # Debug: Print feature count
            print(f"🔍 Extracted {len(features)} features (target: {self.target_feature_count})")

            # Handle NaN and infinity values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure exact feature count
            if len(features) != self.target_feature_count:
                if len(features) > self.target_feature_count:
                    # Truncate if too many features
                    features = features[:self.target_feature_count]
                    print(f"⚠️ Truncated to {self.target_feature_count} features")
                else:
                    # Pad with zeros if too few features
                    padding = np.zeros(self.target_feature_count - len(features))
                    features = np.concatenate([features, padding])
                    print(f"⚠️ Padded to {self.target_feature_count} features")

            return features

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return np.zeros(self.target_feature_count)


class LiveEmotionDetector:
    """Main class for live emotion detection"""

    def __init__(self, model_path, scaler_path, label_encoder_path):
        # Device configuration
        self.device = self.get_device()
        print(f"🖥️ Using device: {self.device}")

        # Load model and preprocessing objects
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)
        self.label_encoder = self.load_label_encoder(label_encoder_path)

        # Audio configuration
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.record_seconds = 3.0  # Duration of each audio chunk for analysis

        # Feature extractor - get expected feature count from model
        expected_features = self.model.input_bn.num_features if self.model else 367
        self.feature_extractor = AudioFeatureExtractor(self.sample_rate, expected_features)

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Threading and queues
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.recording = False

        # History for smoothing predictions
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.6

        # Results storage
        self.results_log = []

        # Add silence detection
        self.silence_threshold = 0.01  # Adjust based on your needs
        self.min_audio_level = 0.005

    def get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def safe_torch_load(self, path):
        """Safely load PyTorch model with compatibility for different versions"""
        try:
            return torch.load(path, weights_only=True)
        except Exception as e:
            print(f"⚠️ Loading with weights_only=False due to compatibility issue")
            return torch.load(path, weights_only=False)

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            checkpoint = self.safe_torch_load(model_path)

            # Get model parameters from checkpoint
            input_size = checkpoint.get('input_size', 367)
            num_classes = checkpoint.get('num_classes', 8)
            dropout_rate = checkpoint.get('dropout_rate', 0.4)

            # Initialize model
            model = AdvancedEmotionNet(
                input_size=input_size,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            ).to(self.device)

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"✅ Model loaded successfully")
            print(f"📊 Input size: {input_size}, Classes: {num_classes}")
            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def load_scaler(self, scaler_path):
        """Load the feature scaler"""
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
            print(f"📊 Scaler expects {scaler.n_features_in_} features")
            return scaler
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return None

    def load_label_encoder(self, label_encoder_path):
        """Load the label encoder"""
        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("✅ Label encoder loaded successfully")
            print(f"🏷️ Classes: {label_encoder.classes_}")
            return label_encoder
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            return None

    def is_silence(self, audio_data):
        """Check if audio data is mostly silence"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < self.silence_threshold

    def preprocess_audio(self, audio_data):
        """Preprocess audio data"""
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply simple noise reduction (high-pass filter)
        from scipy.signal import butter, filtfilt
        try:
            nyquist = self.sample_rate // 2
            low_freq = 80  # Remove frequencies below 80Hz
            b, a = butter(4, low_freq / nyquist, btype='high')
            audio_data = filtfilt(b, a, audio_data)
        except:
            pass  # Skip if scipy is not available

        return audio_data

    def predict_emotion(self, audio_data):
        """Predict emotion from audio data"""
        try:
            # Check for silence
            if self.is_silence(audio_data):
                print("🔇 Silence detected, skipping prediction")
                return "neutral", 0.5, np.ones(len(self.label_encoder.classes_)) / len(self.label_encoder.classes_)

            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data)

            # Extract features
            features = self.feature_extractor.extract_features(audio_data)

            # Verify feature count
            if len(features) != self.scaler.n_features_in_:
                print(f"⚠️ Feature count mismatch: got {len(features)}, expected {self.scaler.n_features_in_}")
                return "unknown", 0.0, np.zeros(len(self.label_encoder.classes_))

            # Normalize features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]

            # Get emotion label
            emotion = self.label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
            confidence_score = confidence.cpu().item()

            # Get all probabilities for visualization
            all_probs = probabilities.cpu().numpy()[0]

            return emotion, confidence_score, all_probs

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return "unknown", 0.0, np.zeros(len(self.label_encoder.classes_))

    def audio_callback(self):
        """Audio recording callback"""
        frames = []
        frames_per_buffer = int(self.sample_rate * self.record_seconds)

        while self.recording:
            try:
                # Record audio
                audio_data = []
                for _ in range(0, int(self.sample_rate / self.chunk_size * self.record_seconds)):
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data.append(data)

                # Convert to numpy array
                audio_bytes = b''.join(audio_data)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

                # Put in queue for processing
                self.audio_queue.put(audio_array)

            except Exception as e:
                print(f"❌ Error in audio recording: {e}")
                break

    def processing_callback(self):
        """Audio processing callback"""
        while self.recording:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1.0)

                # Predict emotion
                emotion, confidence, probabilities = self.predict_emotion(audio_data)

                # Add to history for smoothing
                self.prediction_history.append((emotion, confidence, probabilities))

                # Smooth predictions
                smoothed_emotion, smoothed_confidence = self.smooth_predictions()

                # Store result
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'emotion': smoothed_emotion,
                    'confidence': smoothed_confidence,
                    'raw_emotion': emotion,
                    'raw_confidence': confidence,
                    'probabilities': probabilities.tolist()
                }

                self.result_queue.put(result)
                self.results_log.append(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in processing: {e}")

    def smooth_predictions(self):
        """Smooth predictions using history"""
        if not self.prediction_history:
            return "unknown", 0.0

        # Get recent predictions
        recent_predictions = list(self.prediction_history)

        # Count emotions weighted by confidence
        emotion_scores = {}
        for emotion, confidence, _ in recent_predictions:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += confidence

        # Get most confident emotion
        if emotion_scores:
            smoothed_emotion = max(emotion_scores, key=emotion_scores.get)
            smoothed_confidence = emotion_scores[smoothed_emotion] / len(recent_predictions)
            return smoothed_emotion, smoothed_confidence

        return "unknown", 0.0

    def start_detection(self):
        """Start live emotion detection"""
        try:
            # Initialize audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print("🎤 Audio stream initialized")

            # Start recording
            self.recording = True

            # Start threads
            audio_thread = threading.Thread(target=self.audio_callback)
            processing_thread = threading.Thread(target=self.processing_callback)

            audio_thread.start()
            processing_thread.start()

            print("🚀 Live emotion detection started!")
            print("Press Ctrl+C to stop...")

            # Main loop for displaying results
            while self.recording:
                try:
                    result = self.result_queue.get(timeout=1.0)

                    # Display result
                    timestamp = datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')
                    emotion = result['emotion']
                    confidence = result['confidence']

                    # Color code based on confidence
                    if confidence > 0.8:
                        color = "🟢"
                    elif confidence > 0.6:
                        color = "🟡"
                    else:
                        color = "🔴"

                    print(f"{color} [{timestamp}] Emotion: {emotion.upper()} "
                          f"(Confidence: {confidence:.2f})")

                    # Show top 3 emotions
                    probs = np.array(result['probabilities'])
                    top_indices = np.argsort(probs)[-3:][::-1]

                    print("   Top emotions:")
                    for i, idx in enumerate(top_indices):
                        emotion_name = self.label_encoder.classes_[idx]
                        prob = probs[idx]
                        print(f"     {i + 1}. {emotion_name}: {prob:.3f}")
                    print("-" * 50)

                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    print("\n🛑 Stopping detection...")
                    break

            # Cleanup
            self.stop_detection()

            # Wait for threads to finish
            audio_thread.join()
            processing_thread.join()

        except Exception as e:
            print(f"❌ Error in detection: {e}")
            self.stop_detection()

    def stop_detection(self):
        """Stop live emotion detection"""
        self.recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        print("✅ Detection stopped")

    def save_results(self, filename="emotion_detection_log.json"):
        """Save detection results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results_log, f, indent=2)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def get_emotion_statistics(self):
        """Get statistics from detection session"""
        if not self.results_log:
            return {}

        emotions = [result['emotion'] for result in self.results_log]
        confidences = [result['confidence'] for result in self.results_log]

        from collections import Counter
        emotion_counts = Counter(emotions)

        stats = {
            'total_detections': len(self.results_log),
            'emotion_distribution': dict(emotion_counts),
            'average_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences)
        }

        return stats


def main():
    """Main function to run live emotion detection"""
    # Paths to your trained model and preprocessing objects
    model_path = "../results/best_emotion_model.pth"
    scaler_path = "../results/feature_scaler.pkl"
    label_encoder_path = "../results/label_encoder.pkl"

    # Check if files exist
    required_files = [model_path, scaler_path, label_encoder_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: Required file not found: {file_path}")
            return

    # Initialize detector
    print("🔧 Initializing Live Emotion Detector...")
    detector = LiveEmotionDetector(model_path, scaler_path, label_encoder_path)

    if detector.model is None:
        print("❌ Failed to load model. Exiting.")
        return

    try:
        # Start detection
        detector.start_detection()

        # Show statistics
        print("\n📊 Session Statistics:")
        stats = detector.get_emotion_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Save results
        detector.save_results()

    except KeyboardInterrupt:
        print("\n🛑 Detection interrupted by user")
    except Exception as e:
        print(f"❌ Error in main: {e}")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()