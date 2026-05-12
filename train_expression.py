"""
Training Script for Facial Expression Recognition (PyTorch)
Pre-trained MobileNetV2 + CBAM on FER2013

GPU-only. Usage: python train_expression.py
"""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from expression_model import build_expression_model

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FER_DIR = os.path.join(BASE_PATH, "fer2013")
MODEL_DIR = os.path.join(BASE_PATH, "models")
TRAIN_DIR = os.path.join(FER_DIR, "train")
TEST_DIR = os.path.join(FER_DIR, "test")

IMG_SIZE = 224   # MobileNetV2 expects 224x224
EPOCHS = 50      # Pretrained model converges faster
LEARNING_RATE = 0.0005
BATCH_SIZE = 64  # Default, overridden for GPU


# =========================================================================
# GPU Setup
# =========================================================================

def setup_device():
    """Configure PyTorch for GPU training. Exits if no GPU found."""
    print("=" * 60)
    print("  GPU Diagnostics")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n" + "!" * 60)
        print("  ERROR: No GPU detected!")
        print("!" * 60)
        print("""
  TO FIX - install PyTorch with CUDA:

  1. Go to: https://pytorch.org/get-started/locally/
  2. Select your OS, CUDA version, and pip
  3. Run the generated command, e.g.:
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  Then verify:
     python -c "import torch; print(torch.cuda.is_available())"
""")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  [OK] GPU: {gpu_name}")
    print(f"  [OK] VRAM: {gpu_mem:.1f} GB")

    # Set batch size based on VRAM
    if gpu_mem >= 8:
        batch_size = 128
    elif gpu_mem >= 4:
        batch_size = 64
    else:
        batch_size = 32

    print(f"  [OK] Batch size: {batch_size}")
    print("=" * 60)

    device = torch.device('cuda')
    return device, batch_size, gpu_name


def verify_dataset():
    """Verify FER2013 is downloaded."""
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Dataset not found. Run: python download_fer2013.py")
        return False
    print("\nDataset Statistics:")
    total = 0
    for emotion in sorted(os.listdir(TRAIN_DIR)):
        tp = os.path.join(TRAIN_DIR, emotion)
        if os.path.isdir(tp):
            count = len(os.listdir(tp))
            total += count
            print(f"  {emotion:10s}: {count:5d} images")
    print(f"  {'TOTAL':10s}: {total:5d}")
    return True


def create_data_loaders(batch_size):
    """Create train/val/test data loaders with augmentation."""

    # FER2013 is grayscale 48x48 — we resize to 224x224 and convert to 3-channel
    # for MobileNetV2, using ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Grayscale -> 3ch
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    # Split into train/val (85/15)
    val_size = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

    # Val set should use test transforms (no augmentation)
    val_set_clean = datasets.ImageFolder(TRAIN_DIR, transform=test_transform)
    val_indices = val_set.indices
    val_set = torch.utils.data.Subset(val_set_clean, val_indices)

    test_set = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    class_names = full_train.classes  # ['angry', 'disgust', ...]
    return train_loader, val_loader, test_loader, class_names, full_train


def train():
    device, batch_size, gpu_name = setup_device()

    print(f"\n{'='*60}")
    print(f"  Facial Expression Recognition - Training")
    print(f"  Model: MobileNetV2 + CBAM | Dataset: FER2013")
    print(f"  Device: {gpu_name} | Batch Size: {batch_size}")
    print(f"{'='*60}")

    if not verify_dataset():
        sys.exit(1)

    # Data
    train_loader, val_loader, test_loader, class_names, full_train = create_data_loaders(batch_size)
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Class weights for imbalance
    all_labels = [label for _, label in full_train.samples]
    cw = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.FloatTensor(cw).to(device)
    print(f"Class weights: { {class_names[i]: f'{w:.2f}' for i, w in enumerate(cw)} }")

    # Build model
    model = build_expression_model(num_classes=num_classes, pretrained=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable\n")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "expression_model.pth")
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15

    print(f"Training for up to {EPOCHS} epochs (early stopping patience={patience})...")
    print(f"Estimated time on GPU: ~5-15 minutes\n")

    for epoch in range(EPOCHS):
        start = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        # Metrics
        t_loss = train_loss / train_total
        t_acc = train_correct / train_total
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        elapsed = time.time() - start

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"{elapsed:.1f}s")

        scheduler.step(v_loss)

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (val_acc: {v_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Save class mapping
    class_map = {i: name.capitalize() for i, name in enumerate(class_names)}
    with open(os.path.join(MODEL_DIR, "expression_classes.pkl"), 'wb') as f:
        pickle.dump(class_map, f)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("  Final Evaluation on Test Set")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += images.size(0)

    test_acc = test_correct / test_total
    print(f"\nTest Accuracy: {test_acc*100:.1f}%")
    print(f"Best Val Accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    train()
