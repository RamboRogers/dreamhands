# ===== STEP 1: Install Dependencies =====
# pip install moondream  # Install dependencies in your project directory


# ===== STEP 2: Download Model =====
# Download model (1,733 MiB download size, 2,624 MiB memory usage)
# Use: wget (Linux and Mac) or curl.exe -O (Windows)
# wget https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz

import cv2
import moondream as md
from PIL import Image
import numpy as np
import time
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM
import os

class HandDetector:
    def __init__(self):
        print("Initializing Moondream model...")
        
        try:
            # Check for MPS availability
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("MPS (Metal Performance Shaders) is available! Using Apple Silicon acceleration...")
                self.device = torch.device("mps")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "vikhyatk/moondream2",
                        revision="2025-01-09",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map={"": self.device}
                    )
                except Exception as e:
                    print(f"Failed to initialize MPS model: {e}")
                    print("Falling back to CPU model...")
                    self.fallback_to_cpu()
            else:
                print("MPS device not found. Using CPU.")
                self.fallback_to_cpu()
        except Exception as e:
            print(f"Error during initialization: {e}")
            print("Falling back to CPU model...")
            self.fallback_to_cpu()
        
        self.last_hand_count = 0
        self.last_detection_time = time.time()
        self.is_processing = False
        self.last_message = ""
        print("Model initialized successfully!")
    
    def fallback_to_cpu(self):
        """Fallback to CPU-based model"""
        model_path = './moondream-2b-int8.mf.gz'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please download the model using:")
            print("wget https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = md.vl(model=model_path)

    def analyze_frame(self, image_pil):
        """Detailed analysis of the frame"""
        try:
            print("Processing frame...")
            # Don't try to move the encoded image to device - moondream handles this internally
            encoded_image = self.model.encode_image(image_pil)
            
            # Get both a hand count and a detailed description
            hand_query = "How many hands are raised in the air? Reply with just a number."
            scene_query = "Describe what you see in this image in one short sentence."
            
            hand_response = self.model.query(encoded_image, hand_query)["answer"]
            scene_response = self.model.query(encoded_image, scene_query)["answer"]
            
            print(f"Hand count response: {hand_response}")
            print(f"Scene description: {scene_response}")
            
            number = ''.join(filter(str.isdigit, hand_response))
            count = int(number) if number else 0
            
            return count, scene_response
        except Exception as e:
            print(f"Analysis error: {e}")
            return 0, f"Analysis failed: {str(e)}"

def create_overlay(frame, hand_count, fps, is_processing, message=""):
    """Create a beautiful overlay with hand count and FPS"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay at the bottom
    overlay = frame.copy()
    overlay_height = 150 if message else 100
    cv2.rectangle(overlay, (0, height-overlay_height), (width, height), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Calculate text sizes for better positioning
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    thickness = 1
    
    # Add hand count (left side)
    count_text = f"Hands in air: {hand_count} "
    status_text = "READY (Press SPACE to analyze)" if not is_processing else "PROCESSING..."
    status_color = (0, 255, 0) if not is_processing else (0, 255, 255)
    
    # Get text sizes
    count_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]
    
    # Position text
    count_x = 20
    status_x = count_x + count_size[0] + 10  # Add some padding
    text_y = height - 60
    
    # Draw texts
    cv2.putText(frame, count_text, (count_x, text_y), 
                font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, status_text, (status_x, text_y),
                font, font_scale, status_color, thickness)
    
    # Add analysis message if present
    if message:
        cv2.putText(frame, message, (20, height-20),
                    font, 0.7, (255, 255, 255), 1)
    
    # Add timestamp (top right)
    timestamp = datetime.now().strftime("%H:%M:%S")
    timestamp_size = cv2.getTextSize(timestamp, font, 0.7, 1)[0]
    timestamp_x = width - timestamp_size[0] - 20
    cv2.putText(frame, timestamp, (timestamp_x, height-100),
                font, 0.7, (255, 255, 255), 1)
    
    return frame

def main():
    print("Starting application...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    
    # Set resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = HandDetector()
    
    processing_frame = False
    last_message = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Check for spacebar press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not processing_frame:
            processing_frame = True
            # Convert frame to PIL Image for Moondream
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Perform detailed analysis
            detector.last_hand_count, last_message = detector.analyze_frame(pil_image)
            processing_frame = False
        elif key == ord('q'):
            break
        
        # Add overlay with current hand count and status
        frame = create_overlay(frame, detector.last_hand_count, 0, processing_frame, last_message)
        
        # Display the frame
        cv2.imshow('Hand Detection', frame)
    
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
