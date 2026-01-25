A. MaskDetector Class (Core Engine)
1. __init__() - Initialization

Loads Keras mask classification model (MobileNetV2 or EfficientNet)
Initializes one of 4 face detectors:
DNN (ResNet-10, default) - Auto-downloads 2 files if missing
YuNet (newest, fastest) - Auto-downloads 1 file if missing
MTCNN (accurate but slow) - Requires pip install mtcnn
Haar Cascade (legacy, inaccurate for masks)
Sets up enhancement type if specified
2. _download_dnn_models() & _download_yunet_model()

Auto-downloads face detector models from GitHub
Handles network errors gracefully
3. detect_faces(frame) - Face Detection Stage

Input: Webcam frame (BGR image)
Process: Runs selected face detector algorithm
Output: List of face bounding boxes [(x, y, w, h), ...]
Different logic per detector:
DNN: Creates blob → forward pass → filters by confidence > 0.5
YuNet: Sets input size → detects
MTCNN: Converts BGR→RGB → detects with confidence > 0.9
Haar: Converts to grayscale → cascade detection
4. preprocess_face(face_roi) - Preprocessing

Input: Cropped face region
Process:
Resize to 224×224 (model input size)
Keep 0-255 range (no normalization to match training)
Add batch dimension
Output: Ready-for-model array
5. predict(face_roi) - Mask Classification

Input: Face region
Process: Preprocess → model.predict()
Output: (class_name, confidence, class_idx)
class_idx: 0=Incorrect, 1=With Mask, 2=Without
6. get_color_and_status(class_idx) - UI Mapping

Input: Class index
Output: (color, status_text)
With Mask → Green + "ALLOWED"
Without/Incorrect → Red + "DENIED"
B. main() Function (Application Loop)
1. Argument Parsing

2. Initialization

Validates model exists
Creates MaskDetector instance
Opens video source (webcam or file)
Sets up video writer if saving
3. Main Loop (Real-time Processing)

4. Cleanup

Release camera
Close video writer
Print final statistics