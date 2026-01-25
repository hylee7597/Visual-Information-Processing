"""
Real-Time Face Mask Detection System
Detects masks in real-time from webcam/video streams
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import time
import argparse
import urllib.request
from datetime import datetime

class MaskDetector:
    # Initialize mask detector with trained model and face detector
    def __init__(self, model_path, cascade_path=None, enhance_type=None, face_detector='dnn'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Incorrect Mask', 'With Mask', 'Without Mask']
        self.enhance_type = enhance_type
        self.detector_type = face_detector
        
        # Load face detector based on type
        # DNN-based face detector (more accurate, better for masks)
        if face_detector == 'dnn':
            prototxt = "deploy.prototxt"
            caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Try to load from current directory, otherwise download
            if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
                print("[INFO] DNN model files not found. Downloading...")
                self._download_dnn_models()
            
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print(f"[INFO] Loaded DNN face detector (ResNet-10 SSD)")

        # YuNet face detector (OpenCV's newest, very fast)           
        elif face_detector == 'yunet':
            model_path_yunet = "face_detection_yunet_2023mar.onnx"
            if not os.path.exists(model_path_yunet):
                print("[INFO] YuNet model not found. Downloading...")
                self._download_yunet_model()
            
            self.face_net = cv2.FaceDetectorYN.create(model_path_yunet, "", (0, 0))
            print(f"[INFO] Loaded YuNet face detector")
        # MTCNN face detector (Multi-task CNN, with landmarks, but very low FPS)
        elif face_detector == 'mtcnn':
            try:
                from mtcnn import MTCNN
                self.face_detector_mtcnn = MTCNN()
                print(f"[INFO] Loaded MTCNN face detector")
            except ImportError:
                print("[ERROR] MTCNN not installed. Install: pip install mtcnn")
                raise
            
        else:  # haar cascade (default fallback)
            if cascade_path is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError(f"Could not load cascade from {cascade_path}")
            print(f"[INFO] Loaded Haar Cascade face detector")
        
        print(f"[INFO] Loaded mask detection model: {model_path}")
    
    # Function to download DNN models (if not present)
    def _download_dnn_models(self):
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        files = {
            "deploy.prototxt": base_url + "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": 
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"[INFO] Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"[INFO] Downloaded {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to download {filename}: {e}")
                    print(f"[INFO] Please download manually from: {url}")
    
    # Function to download YuNet model (if not present)
    def _download_yunet_model(self):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        filename = "face_detection_yunet_2023mar.onnx"
        
        try:
            print(f"[INFO] Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"[INFO] Downloaded {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            print(f"[INFO] Please download manually from: {url}")
    
    # Function to detect faces in frame using selected detector
    def detect_faces(self, frame):
        """
        Args:
            frame: Input frame (BGR)
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """

        # DNN face detector
        if self.detector_type == 'dnn':
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections (confidence > 0.5)
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Convert to (x, y, w, h) format
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            
            return faces
        
        # YuNet face detector
        elif self.detector_type == 'yunet':
            (h, w) = frame.shape[:2]
            self.face_net.setInputSize((w, h))
            _, detections = self.face_net.detect(frame)
            
            faces = []
            if detections is not None:
                for detection in detections:
                    x, y, w, h = detection[:4].astype(int)
                    faces.append((x, y, w, h))
            
            return faces
        
        # MTCNN face detector
        elif self.detector_type == 'mtcnn':
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.face_detector_mtcnn.detect_faces(frame_rgb)
            
            faces = []
            for detection in detections:
                if detection['confidence'] > 0.9:  # High confidence threshold
                    x, y, w, h = detection['box']
                    # Ensure positive coordinates
                    x = max(0, x)
                    y = max(0, y)
                    faces.append((x, y, w, h))
            
            return faces
        
        else:  # haar
            # Haar Cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            return faces
    
    # Function to preprocess face image for model prediction
    def preprocess_face(self, face_roi):
        """
        Args:
            face_roi: Face region of interest (numpy array)
        Returns:
            Preprocessed image array
        """
        # Resize to model input size (224x224)
        resized = cv2.resize(face_roi, (224, 224))
        # Keep pixel values in 0-255 range (no normalization) to match training
        # Add batch dimension
        batch = np.expand_dims(resized, axis=0)
        return batch
    
    # Function to classify mask status on face images
    def predict(self, face_roi):
        """
        Args:
            face_roi: Face region of interest 
        Returns:
            Tuple of (class_name, confidence, class_idx)
        """
        processed = self.preprocess_face(face_roi)
        predictions = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = self.class_names[class_idx]
        
        return class_name, confidence, class_idx
    
    # Function to get bounding box color and mask status based on prediction
    def get_color_and_status(self, class_idx):
        """
        Args:
            class_idx: Predicted class index (0=Incorrect, 1=With Mask, 2=Without) 
        Returns:
            Tuple of (color (BGR), status_text)
        """
        if class_idx == 1:  # With Mask
            return (0, 255, 0), "ALLOWED"  # Green
        else:  # Incorrect or Without Mask
            return (0, 0, 255), "DENIED"  # Red

# Main function for real-time mask detection
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-Time Face Mask Detection')
    # Model to be used for mask status detection
    parser.add_argument('--model', type=str, default='mask_detector_efficientnet.h5',
                        help='Path to trained model (.h5 file)')
    # Video source (default is webcam)
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: 0 for webcam, or path to video file')
    # Confidence threshold for predictions (default 0.5)
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    # Display FPS counter (default: True)
    parser.add_argument('--fps-display', action='store_true', default=True,
                        help='Display FPS counter')
    # Save output video path (default: None = do not save)
    parser.add_argument('--save-output', type=str, default=None,
                        help='Path to save output video (e.g., output.mp4)')
    # Face detector type (default: haar)
    parser.add_argument('--face-detector', type=str, default='haar',
                        choices=['dnn', 'haar', 'yunet', 'mtcnn'],
                        help='Face detection method: dnn (recommended), yunet (fastest), mtcnn (most accurate but slow), haar (legacy)')
    args = parser.parse_args()
    
    # Load model for mask detection
    try:
        if not os.path.exists(args.model):
            print(f"[ERROR] Model not found: {args.model}")
            print("[INFO] Available models in current directory:")
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    print(f"  - {f}")
            return
        
        # Determine model name for saving frames
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        if model_name == 'mask_detector_efficientnet':
            model_name = 'efficientnet'
        elif model_name == 'mask_detector_mobilenetv2':
            model_name = 'mobilenetv2'
        
        # Initialize mask detector
        detector = MaskDetector(args.model, face_detector=args.face_detector)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize detector: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    # Open video source (webcam or video file)
    source = 0 if args.source == '0' else args.source
    if isinstance(source, str) and not os.path.exists(source):
        print(f"[ERROR] Video source not found: {source}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(source if isinstance(source, int) else source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # Setup video writer if saving output
    out = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_output, fourcc, fps_cap, (frame_width, frame_height))
        print(f"[INFO] Output video will be saved to: {args.save_output}")
    
    print("[INFO] Starting real-time detection. Press 'q' to quit, 's' to save frame.")
    
    # Main real-time processing loop
    frame_count = 0
    fps_list = []
    
    while True:
        # Read frame from video source
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timer for FPS calculation
        start_time = time.time()
        frame_count += 1
        
        # Detect faces using configured detector
        faces = detector.detect_faces(frame)
        
        detections = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # ROI = Region of Interest (face area)
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict mask status
            class_name, confidence, class_idx = detector.predict(face_roi)
            
            # Get color and status
            color, status = detector.get_color_and_status(class_idx)
            
            # Store detection info in list
            detections.append({
                'bbox': (x, y, w, h),
                'class_name': class_name,
                'confidence': confidence,
                'class_idx': class_idx,
                'color': color,
                'status': status
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label with background (eg. "With Mask: 0.95")
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw status (eg. "ALLOWED" or "DENIED")
            cv2.putText(frame, status, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Calculate FPS (Frames Per Second) - the higher the better (faster processing)
        elapsed = time.time() - start_time
        fps = 1 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)

        # Keep last 30 FPS values for averaging
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = np.mean(fps_list)
        
        # Display statistics
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count masks
        with_mask = sum(1 for d in detections if d['class_idx'] == 1)
        cv2.putText(frame, f"With Mask: {with_mask}/{len(detections)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video
        if out:
            out.write(frame)
        
        # Display frame
        cv2.imshow('Face Mask Detection', frame)
        
        # Handle keyboard input (s to save frame into image, q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Exiting...")
            break
        elif key == ord('s'):
            filename = f"mask_detection_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Frame saved: {filename}")
    
    # Cleanup and release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"[INFO] Processed {frame_count} frames")
    print(f"[INFO] Average FPS: {np.mean(fps_list):.1f}")

# Main entry point
if __name__ == '__main__':
    main()
