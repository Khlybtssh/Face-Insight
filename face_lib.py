import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pickle

# PyTorch for expression model
import torch
import torchvision.transforms as T
from expression_model import ExpressionModel, EMOTION_EMOJIS

class FaceSystem:
    def __init__(self, data_dir="data", model_dir="models"):
        # Use absolute paths relative to this script file
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_path, data_dir)
        self.model_dir = os.path.join(base_path, model_dir)
        
        self.model_path = os.path.join(self.model_dir, "face_model.keras")
        self.classes_path = os.path.join(self.model_dir, "classes.pkl")
        
        # Expression model paths (PyTorch)
        self.expr_model_path = os.path.join(self.model_dir, "expression_model.pth")
        self.expr_classes_path = os.path.join(self.model_dir, "expression_classes.pkl")
        
        # PyTorch device (CPU for real-time inference)
        self.torch_device = torch.device('cpu')
        
        # Expression preprocessing (same as training)
        self.expr_transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=3),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Ensure directories exist
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directories: {e}")
        
        # Load Face Detector (Haar Cascade is faster/lighter for CPU than MTCNN)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Identity model
        self.model = None
        self.class_names = {}
        
        # Expression model
        self.expr_model = None
        self.expr_class_names = {}
        
        self.load_resources()

    def load_resources(self):
        """Try to load existing identity model and expression model."""
        # Load Identity Model
        if os.path.exists(self.model_path) and os.path.exists(self.classes_path):
            try:
                self.model = load_model(self.model_path)
                with open(self.classes_path, 'rb') as f:
                    self.class_names = pickle.load(f)
                print("Identity model and classes loaded.")
            except Exception as e:
                print(f"Error loading identity model: {e}")
        else:
            print("No trained identity model found.")
        
        # Load Expression Model (PyTorch)
        if os.path.exists(self.expr_model_path) and os.path.exists(self.expr_classes_path):
            try:
                with open(self.expr_classes_path, 'rb') as f:
                    self.expr_class_names = pickle.load(f)
                num_classes = len(self.expr_class_names)
                self.expr_model = ExpressionModel(num_classes=num_classes, pretrained=False)
                self.expr_model.load_state_dict(
                    torch.load(self.expr_model_path, map_location=self.torch_device, weights_only=True)
                )
                self.expr_model.to(self.torch_device)
                self.expr_model.eval()
                print("Expression model loaded (PyTorch).")
            except Exception as e:
                print(f"Error loading expression model: {e}")
        else:
            print("No trained expression model found.")

    def detect_faces(self, frame):
        """Returns list of (x, y, w, h) for detected faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces

    def save_face(self, frame, face_box, name):
        """Save a cropped face image for a specific user."""
        x, y, w, h = face_box
        # Add a little padding if possible
        padding = 10
        h_img, w_img, _ = frame.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        face_img = frame[y1:y2, x1:x2]
        
        user_dir = os.path.join(self.data_dir, name)
        os.makedirs(user_dir, exist_ok=True)
        
        count = len(os.listdir(user_dir))
        filename = os.path.join(user_dir, f"{name}_{count}.jpg")
        cv2.imwrite(filename, face_img)
        return True

    def train(self, epochs=10):
        """Train the MobileNetV2 model on collected data."""
        # check if we have data
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print("No data found.")
            return "No data to train on."

        # Setup data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        try:
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )
        except Exception as e:
            return f"Error setting up data generators: {e}. Make sure you have enough data."

        num_classes = train_generator.num_classes
        self.class_names = {v: k for k, v in train_generator.class_indices.items()}
        
        # Build Model
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base layers first
        for layer in base_model.layers:
            layer.trainable = False
            
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Fine-tune
        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size + 1,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size + 1,
            epochs=epochs
        )
        
        # Save
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model_path)
        with open(self.classes_path, 'wb') as f:
            pickle.dump(self.class_names, f)
            
        return "Training Complete!"

    def predict(self, face_img):
        """Predict the identity of a face image."""
        if self.model is None:
            return "Unknown", 0.0
            
        # Preprocess
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        preds = self.model.predict(img, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        
        if idx in self.class_names:
            return self.class_names[idx], confidence
        else:
            return "Unknown", confidence

    def predict_expression(self, face_img):
        """
        Predict the facial expression/emotion of a face image.
        
        Args:
            face_img: BGR face crop from webcam (numpy array)
            
        Returns:
            (emotion_name, confidence, emoji) tuple
        """
        if self.expr_model is None:
            return "N/A", 0.0, ""
        
        # Preprocess: BGR -> RGB, then apply torchvision transforms
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_tensor = self.expr_transform(rgb)            # (3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0)              # (1, 3, 224, 224)
        img_tensor = img_tensor.to(self.torch_device)
        
        with torch.no_grad():
            outputs = self.expr_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, idx = probs.max(1)
            confidence = confidence.item()
            idx = idx.item()
        
        if idx in self.expr_class_names:
            emotion = self.expr_class_names[idx]
        else:
            emotion = "Unknown"
        
        emoji = EMOTION_EMOJIS.get(emotion, "")
        return emotion, confidence, emoji

    def predict_combined(self, face_img):
        """
        Run BOTH identity recognition and expression recognition on a face.
        
        Args:
            face_img: BGR face crop from webcam
            
        Returns:
            dict with keys: name, name_conf, emotion, emotion_conf, emoji, label
        """
        # Identity prediction
        name, name_conf = self.predict(face_img)
        
        # Expression prediction
        emotion, emotion_conf, emoji = self.predict_expression(face_img)
        
        # Build combined label: "Ahmed, Happy 😊"
        if name == "Unknown" or name_conf < 0.5:
            name = "Unknown"
        
        label = f"{name}, {emotion} {emoji}"
        
        return {
            'name': name,
            'name_conf': name_conf,
            'emotion': emotion,
            'emotion_conf': emotion_conf,
            'emoji': emoji,
            'label': label
        }
