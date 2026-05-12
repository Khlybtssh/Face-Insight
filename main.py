import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
from face_lib import FaceSystem
import threading

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition + Expression Detection (CNN + CBAM)")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1a1a2e")
        
        # Initialize Backend
        self.fs = FaceSystem()
        
        # State variables
        self.is_recognizing = False
        self.is_collecting = False
        self.collect_name = ""
        self.collect_count = 0
        self.collect_limit = 50  # Number of images to collect
        
        # Emotion color mapping for bounding boxes
        self.emotion_colors = {
            'Angry':    (0, 0, 220),     # Red
            'Disgust':  (0, 140, 0),     # Dark Green
            'Fear':     (180, 0, 180),   # Purple
            'Happy':    (0, 220, 0),     # Green
            'Sad':      (220, 140, 0),   # Blue-ish
            'Surprise': (0, 220, 220),   # Yellow
            'Neutral':  (180, 180, 180), # Gray
        }
        
        # UI Layout
        self.setup_ui()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Start Video Loop
        self.update_video()

    def setup_ui(self):
        # Title Bar
        title_frame = tk.Frame(self.root, bg="#16213e", height=50)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🎭 Face Identity + Expression Recognition",
            font=("Segoe UI", 16, "bold"),
            bg="#16213e", fg="#e94560"
        )
        title_label.pack(expand=True)
        
        # Top Frame for Video
        self.video_frame = tk.Frame(self.root, bg="black", bd=2, relief=tk.GROOVE)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)
        
        # Info Frame (shows model status)
        info_frame = tk.Frame(self.root, bg="#16213e", height=30)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        identity_status = "✅ Loaded" if self.fs.model else "❌ Not trained"
        expr_status = "✅ Loaded" if self.fs.expr_model else "❌ Not trained"
        
        self.model_info = tk.Label(
            info_frame,
            text=f"Identity Model: {identity_status}  |  Expression Model: {expr_status}",
            font=("Consolas", 10),
            bg="#16213e", fg="#a8a8a8"
        )
        self.model_info.pack(side=tk.LEFT, padx=10, pady=2)
        
        # Bottom Frame for Controls
        self.control_frame = tk.Frame(self.root, bg="#1a1a2e", height=80)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        btn_style = {
            'font': ("Segoe UI", 11, "bold"),
            'height': 2, 'width': 18,
            'relief': tk.FLAT, 'cursor': 'hand2'
        }
        
        self.btn_register = tk.Button(
            self.control_frame, text="📷 Register User",
            command=self.start_registration,
            bg="#0f3460", fg="white", activebackground="#1a5276",
            **btn_style
        )
        self.btn_register.pack(side=tk.LEFT, padx=8)
        
        self.btn_train = tk.Button(
            self.control_frame, text="🧠 Train Identity",
            command=self.start_training,
            bg="#0f3460", fg="white", activebackground="#1a5276",
            **btn_style
        )
        self.btn_train.pack(side=tk.LEFT, padx=8)
        
        self.btn_recognize = tk.Button(
            self.control_frame, text="▶ Start Recognition",
            command=self.toggle_recognition,
            bg="#533483", fg="white", activebackground="#6c3e99",
            **btn_style
        )
        self.btn_recognize.pack(side=tk.LEFT, padx=8)
        
        self.status_label = tk.Label(
            self.control_frame, text="Status: Ready",
            font=("Segoe UI", 12, "bold"),
            bg="#1a1a2e", fg="#e94560"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def start_registration(self):
        name = simpledialog.askstring("Input", "Enter Name for the user:")
        if name:
            self.collect_name = name.strip()
            self.collect_count = 0
            self.is_collecting = True
            self.is_recognizing = False
            self.status_label.config(text=f"Status: Collecting data for {name}...")
            
    def start_training(self):
        self.status_label.config(text="Status: Training... Please wait.")
        self.root.update()
        
        # Run training in separate thread to not freeze UI
        def train_task():
            res = self.fs.train(epochs=5) # 5 epochs for quick demo
            self.root.after(0, lambda: self.status_label.config(text=f"Status: {res}"))
            self.fs.load_resources() # Reload new model
            self.root.after(0, self._update_model_info)
            
        threading.Thread(target=train_task, daemon=True).start()

    def _update_model_info(self):
        """Update the model status display."""
        identity_status = "✅ Loaded" if self.fs.model else "❌ Not trained"
        expr_status = "✅ Loaded" if self.fs.expr_model else "❌ Not trained"
        self.model_info.config(
            text=f"Identity Model: {identity_status}  |  Expression Model: {expr_status}"
        )

    def toggle_recognition(self):
        if self.is_recognizing:
            self.is_recognizing = False
            self.btn_recognize.config(text="▶ Start Recognition", bg="#533483")
            self.status_label.config(text="Status: Ready")
        else:
            # Allow recognition if at least one model is loaded
            if self.fs.model is None and self.fs.expr_model is None:
                messagebox.showerror("Error", "No models loaded!\n\n"
                    "• For identity: Register users and Train\n"
                    "• For expression: Run train_expression.py")
                return
            self.is_recognizing = True
            self.is_collecting = False
            self.btn_recognize.config(text="⏹ Stop Recognition", bg="#e94560")
            self.status_label.config(text="Status: Recognizing...")

    def draw_combined_label(self, frame, x, y, w, h, result):
        """Draw bounding box and combined label on frame."""
        # Choose color based on emotion
        emotion = result.get('emotion', 'Neutral')
        color = self.emotion_colors.get(emotion, (180, 180, 180))
        
        # Red box if identity is unknown
        name = result.get('name', 'Unknown')
        if name == "Unknown":
            box_color = (0, 0, 255)  # Red for unknown
        else:
            box_color = color
        
        # Draw face rectangle with rounded look (thick + thin)
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        
        # Build label text
        emoji_text = result.get('emoji', '')
        name_conf = result.get('name_conf', 0)
        emotion_conf = result.get('emotion_conf', 0)
        
        # Line 1: Name + confidence
        line1 = f"{name} ({name_conf:.0%})"
        # Line 2: Emotion + confidence
        line2 = f"{emotion} ({emotion_conf:.0%})"
        
        # Get text sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        
        max_w = max(w1, w2) + 10
        total_h = h1 + h2 + 15
        
        # Background rectangle above face box
        bg_y1 = max(0, y - total_h - 5)
        bg_y2 = y
        cv2.rectangle(frame, (x, bg_y1), (x + max_w, bg_y2), box_color, -1)
        
        # Draw text lines
        cv2.putText(frame, line1, (x + 5, bg_y1 + h1 + 3), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, line2, (x + 5, bg_y1 + h1 + h2 + 10), font, font_scale, (255, 255, 255), thickness)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Face Detection
            faces = self.fs.detect_faces(frame)
            
            # Logic based on state
            if self.is_collecting:
                for (x, y, w, h) in faces:
                    # Draw rectangle green
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if self.collect_count < self.collect_limit:
                        self.fs.save_face(frame, (x,y,w,h), self.collect_name)
                        self.collect_count += 1
                        self.status_label.config(text=f"Collected {self.collect_count}/{self.collect_limit}")
                    else:
                        self.is_collecting = False
                        self.status_label.config(text=f"Status: Collection for {self.collect_name} Done!")
                        messagebox.showinfo("Info", "Data Collection Complete!")
                    # Only collect one face per frame to avoid duplicates/confusion
                    break 

            elif self.is_recognizing:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Skip tiny faces
                    if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                        continue
                    
                    # Combined prediction: identity + expression
                    result = self.fs.predict_combined(face_roi)
                    
                    # Draw the combined label
                    self.draw_combined_label(frame, x, y, w, h, result)

            else:
                # Just draw boxes (idle state)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 255), 2)

            # Convert to Tkinter Format
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
