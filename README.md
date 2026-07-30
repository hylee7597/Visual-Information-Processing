# Real-Time Face Mask Detection

This project detects faces from a webcam or video file and predicts whether a person is wearing a mask correctly, wearing no mask, or wearing an incorrect mask.

## Project files
- `real_time_mask_detector.py` - main real-time detector script
- `mask_detector_efficientnet.keras` / `mask_detector_mobilenetv2.keras` - trained model files
- `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel` - face detector files
- `Dataset/` - image dataset used for training
- `VIP_Project_Group_10.ipynb` - notebook with training/evaluation workflow

## Requirements
The environment is defined in `CDS6334.yml`.

### Create and activate the environment
On Windows with Anaconda/Miniconda:

```powershell
conda env create -f CDS6334.yml
conda activate CDS6334
```

### Available face detectors:
- dnn (default)
- yunet
- mtcnn (Very low FPS) (just for demonstration purposes)
- haar (Low accuracy) (just for demonstration purposes)

Available mask detector models:
- mask_detector_efficientnet.h5 (default)
- mask_detector_mobilenetv2.h5

### Example with YuNet and the H5 model
```powershell
python real_time_mask_detector.py --face-detector yunet --model mask_detector_efficientnet.h5
```

### Controls while the app is running
- Press `q` to quit
- Press `s` to save the current frame as an image

## Notes
- If the required face-detector files are missing, the script can try to download them automatically.
- The default model is `mask_detector_efficientnet.keras`, which is recommended for good performance.
- If TensorFlow or OpenCV is not installed correctly, recreate the Conda environment from `CDS6334.yml`.
