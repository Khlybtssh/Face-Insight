"""
Evaluation Script for Expression Recognition Model (PyTorch)

Usage: python evaluate_expression.py
"""

import os, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from expression_model import ExpressionModel, EMOTION_LABELS

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models", "expression_model.pth")
CLASSES_PATH = os.path.join(BASE_PATH, "models", "expression_classes.pkl")
TEST_DIR = os.path.join(BASE_PATH, "fer2013", "test")


def evaluate():
    print("=" * 60)
    print("  Expression Recognition - Evaluation (PyTorch)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Train the model first: python train_expression.py")
        return

    if not os.path.exists(TEST_DIR):
        print(f"Test data not found: {TEST_DIR}")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    with open(CLASSES_PATH, 'rb') as f:
        class_map = pickle.load(f)
    num_classes = len(class_map)

    model = ExpressionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # Test data
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    # Predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_labels = test_set.classes

    # Classification Report
    print("=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"{'':12s}", end="")
    for label in class_labels:
        print(f"{label[:7]:>9s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_labels[i]:12s}", end="")
        for val in row:
            print(f"{val:9d}", end="")
        print()

    accuracy = (all_preds == all_labels).mean()
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    print(f"Total test samples: {len(all_labels)}")


if __name__ == '__main__':
    evaluate()
