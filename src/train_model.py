import os
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import random
from collections import Counter
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Device Configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_device()
print(f"🖥️ Using device: {device}")


# Enhanced Model Architecture
class AdvancedEmotionNet(nn.Module):
    def __init__(self, input_size=367, num_classes=8, dropout_rate=0.4):  # Updated input size
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


# Enhanced Dataset Class with advanced augmentation
class EmotionDataset(Dataset):
    def __init__(self, features, labels, augment=False, noise_factor=0.005):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise
            if random.random() < 0.5:
                x += torch.randn_like(x) * self.noise_factor

            # Random scaling
            if random.random() < 0.3:
                scale = 1.0 + (random.random() - 0.5) * 0.1
                x *= scale

            # Random masking (dropout some features)
            if random.random() < 0.2:
                mask = torch.rand_like(x) > 0.1
                x *= mask

        return x, y


# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Enhanced plotting functions
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Acc', linewidth=2)
    plt.plot(val_accuracies, label='Val Acc', linewidth=2)
    plt.title("Accuracy Curve", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1', linewidth=2)
    plt.plot(val_f1s, label='Val F1', linewidth=2)
    plt.title("F1 Score Curve", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Count'})
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


# Learning rate scheduler with warmup
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def format_time(seconds):
    """Format seconds into a readable time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def safe_torch_load(path):
    """Safely load PyTorch model with compatibility for different versions"""
    try:
        # First try with weights_only=True (secure mode)
        return torch.load(path, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        if "weights_only" in str(e).lower() or "unpickling" in str(e).lower():
            print(f"⚠️  Loading with weights_only=False due to compatibility issue")
            # Add safe globals for numpy objects commonly saved in checkpoints
            torch.serialization.add_safe_globals([
                'numpy.core.multiarray.scalar',
                'numpy.dtype',
                'numpy.ndarray',
                'builtins.tuple',
                'builtins.dict',
                'builtins.list',
                'builtins.int',
                'builtins.float',
                'builtins.str',
                'builtins.bool'
            ])
            return torch.load(path, weights_only=False)
        else:
            raise e


# Enhanced training function
def train_model(csv_path,
                model_save_path='best_emotionvoice_model.pth',
                batch_size=128,
                epochs=200,
                dropout_rate=0.4,
                lr=0.001,
                patience=25,
                use_focal_loss=True,
                k_fold=5):
    os.makedirs("../results", exist_ok=True)

    # Load and prepare data
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_path)

    # Extract features and labels from the CSV
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    features = df[feature_columns].values
    emotions = df['emotion'].values

    print(f"📈 Dataset shape: {features.shape}")
    print(f"🔧 Feature columns: {len(feature_columns)}")

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(emotions)

    print(f"🎯 Classes: {label_encoder.classes_}")
    print(f"📋 Class Distribution: {Counter(emotions)}")

    # Handle class imbalance - remove very rare classes or combine them
    class_counts = Counter(emotions)
    min_samples = 100  # Minimum samples per class

    # Filter out classes with too few samples
    valid_indices = []
    for i, emotion in enumerate(emotions):
        if class_counts[emotion] >= min_samples:
            valid_indices.append(i)

    if len(valid_indices) < len(features):
        print(f"🔄 Filtering out {len(features) - len(valid_indices)} samples with rare classes")
        features = features[valid_indices]
        emotions = emotions[valid_indices]
        labels = label_encoder.fit_transform(emotions)

    # Split data
    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42, stratify=labels
    )

    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Normalize features
    print("⚖️ Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save preprocessing objects
    with open("../results/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    with open("../results/feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Validation set: {X_val.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")

    # Create datasets with augmentation
    print("🎨 Creating datasets with augmentation...")
    train_dataset = EmotionDataset(X_train, y_train, augment=True)
    val_dataset = EmotionDataset(X_val, y_val, augment=False)
    test_dataset = EmotionDataset(X_test, y_test, augment=False)

    # Create weighted sampler for balanced batches
    class_weights = 1.0 / np.array([np.sum(y_train == i) for i in range(len(label_encoder.classes_))])
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders (set num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Initialize model
    print("🏗️ Initializing model...")
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = AdvancedEmotionNet(input_size=input_size, num_classes=num_classes,
                               dropout_rate=dropout_rate).to(device)

    # Loss function
    if use_focal_loss:
        print("🎯 Using Focal Loss for class imbalance")
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        # Class weights for standard CrossEntropyLoss
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, max_epochs=epochs)

    # Training loop
    best_val_f1 = 0
    best_val_acc = 0
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []

    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("=" * 80)

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"🔥 Epoch {epoch:03d}/{epochs}",
                          leave=False, ncols=120, position=0)

        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())

            # Update progress bar
            current_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        val_pbar = tqdm(val_loader, desc=f"✅ Validation",
                        leave=False, ncols=120, position=0)

        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

                # Update progress bar
                current_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

        # Update learning rate
        current_lr = scheduler.step(epoch)

        # Store metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Calculate time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time

        # Estimate remaining time
        avg_epoch_time = elapsed_time / epoch
        remaining_epochs = epochs - epoch
        est_remaining_time = avg_epoch_time * remaining_epochs

        # Save best model
        improvement = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'input_size': input_size,
                'num_classes': num_classes,
                'dropout_rate': dropout_rate
            }, os.path.join("../results", model_save_path))
            early_stop_counter = 0
            improvement = "✅ NEW BEST!"
        else:
            early_stop_counter += 1

        # Progress reporting with cleaner output
        progress_percentage = (epoch / epochs) * 100
        print(f"\r📊 Epoch {epoch:03d}/{epochs} ({progress_percentage:.1f}%) | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"F1: {val_f1:.3f} | Time: {format_time(epoch_time)} | "
              f"ETA: {format_time(est_remaining_time)} {improvement}", end='')

        # Print newline for important epochs or improvements
        if epoch % 10 == 0 or improvement or epoch == 1:
            print()  # New line for clean output

        # Early stopping
        if early_stop_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch} (patience: {patience})")
            break

    # Training complete
    total_training_time = time.time() - training_start_time
    print("=" * 80)
    print(f"✅ Training complete! Total time: {format_time(total_training_time)}")
    print(f"🏆 Best validation F1: {best_val_f1:.3f}, Accuracy: {best_val_acc:.2f}%")

    # Load best model for final evaluation using safe loading
    print("📂 Loading best model for evaluation...")
    checkpoint = safe_torch_load(os.path.join("../results", model_save_path))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    print("\n🧪 Evaluating on test set...")
    model.eval()
    test_preds = []
    test_targets = []

    test_pbar = tqdm(test_loader, desc="🧪 Testing", ncols=120)
    with torch.no_grad():
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())

    test_acc = 100.0 * np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')

    print(f"\n🎯 Final Test Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 Score: {test_f1:.3f}")

    # Generate plots
    print("\n📊 Generating plots...")
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s)
    plot_confusion_matrix(test_targets, test_preds, label_encoder.classes_)

    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=label_encoder.classes_))

    # Save detailed results
    precision, recall, f1, support = precision_recall_fscore_support(test_targets, test_preds, average=None)
    results_df = pd.DataFrame({
        'emotion': label_encoder.classes_,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })
    results_df.to_csv('results/classification_report.csv', index=False)

    print(f"\n💾 Model saved as: {model_save_path}")
    print(f"📁 All results saved in 'results/' directory")


if __name__ == "__main__":
    csv_path = "../results/enhanced_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
    else:
        train_model(
            csv_path=csv_path,
            model_save_path='best_emotion_model.pth',
            batch_size=128,
            epochs=200,
            dropout_rate=0.4,
            lr=0.001,
            patience=25,
            use_focal_loss=True
        )