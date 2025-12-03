# Recycling Detection - Inference Setup

## Quick Start

1. **Install Python 3.8 or higher**

2. **Install dependencies:**
```bash
pip install -r requirements_inference.txt
```

3. **Run webcam detection:**
```bash
python src/inference/webcam_demo.py
```

4. **Run image detection:**
```bash
# Edit IMAGE_PATH in the file, then run:
python src/inference/image_processor.py
```

## Files Needed
- `models/teacher/best.pt` - YOLO model weights
- `src/inference/detector.py` - Detection class
- `src/inference/webcam_demo.py` - Webcam script
- `src/inference/image_processor.py` - Image processing script

## Keyboard Controls
- **q** - Quit
- **s** - Save frame
- **p** - Pause/Resume (webcam only)
- **f** - Fullscreen toggle

## Troubleshooting

**ImportError: No module named 'ultralytics'**
```bash
pip install ultralytics
```

**Can't find model file:**
- Make sure `best.pt` is in `models/teacher/` folder
- Or specify path: `--model path/to/your/model.pt`
