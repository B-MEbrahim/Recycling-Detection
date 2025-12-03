import cv2 
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any 
import torch
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class RecyclingDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, inference_size: int = 416):
        """Initialize the detector with a Yolo model"""
        # Load model - works with both .pt and .torchscript
        self.model = YOLO(model_path, task='detect')
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        self.inference_size = inference_size  # Default 416 for speed (was 640)

    def detect(self, image: np.ndarray, resize_input: bool = True) -> List[Dict[str, Any]]:
        """Detect objects in an image"""
        original_height, original_width = image.shape[:2]
        
        # Resize input for faster inference (enabled by default)
        if resize_input and max(original_width, original_height) > self.inference_size:
            scale = self.inference_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized_image = image
            scale = 1.0
        
        # run inference
        results = self.model(
            resized_image,
            conf=self.confidence_threshold,
            imgsz=self.inference_size,
            verbose=False
        )[0]

        # DEBUG: Print raw results
        img_height, img_width = image.shape[:2]
        print(f"Image size: {img_width}x{img_height}")
        if results.boxes is not None:
            print(f"Number of detections: {len(results.boxes)}")
            for i, box in enumerate(results.boxes):
                print(f"Box {i}: xyxy={box.xyxy[0].cpu().numpy()}")
                print(f"Box {i}: xywh={box.xywh[0].cpu().numpy()}")

        # parse results 
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get coordinates - xyxy format gives pixel coordinates directly
                bbox = box.xyxy[0].cpu().numpy()
                
                # Scale coordinates back to original image size
                x1 = float(max(0, min(bbox[0] / scale, original_width)))
                y1 = float(max(0, min(bbox[1] / scale, original_height)))
                x2 = float(max(0, min(bbox[2] / scale, original_width)))
                y2 = float(max(0, min(bbox[3] / scale, original_height)))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid box coordinates: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.class_names[int(box.cls[0])]
                }
                detections.append(detection)
                
                logger.debug(f"Detection: {detection['class_name']} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                           f"(conf: {detection['confidence']:.3f})")
        
        return detections
    
    def detect_from_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect objects from image bytes"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.detect(image)
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on image"""
        img_copy = image.copy()
        img_height, img_width = img_copy.shape[:2]
        
        # Scale font and thickness based on image size
        scale_factor = max(img_width, img_height) / 1000.0
        font_scale = max(0.8, 0.5 * scale_factor)
        font_thickness = max(2, int(2 * scale_factor))
        box_thickness = max(3, int(3 * scale_factor))

        for det in detections:
            # Get bounding box coordinates (already in pixel format)
            bbox = det['bbox']
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Skipping invalid box: ({x1}, {y1}, {x2}, {y2})")
                continue
            
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                logger.warning(f"Box outside image bounds: ({x1}, {y1}, {x2}, {y2}) "
                             f"for image size ({img_width}, {img_height})")
            
            confidence = det['confidence']
            class_name = det['class_name']

            # draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, box_thickness)

            # draw label with background
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label above the box, or inside if at top of image
            padding = 10
            if y1 - label_height - padding > 0:
                # Above the box
                label_y_top = y1 - padding
                label_y_text = y1 - padding - 5
            else:
                # Inside the box at top
                label_y_top = y1
                label_y_text = y1 + label_height + 5
            
            label_x1 = max(0, x1)
            label_x2 = min(img_width, x1 + label_width + padding * 2)
            
            # Draw label background with semi-transparency effect
            cv2.rectangle(img_copy, 
                         (label_x1, label_y_top - label_height - padding),
                         (label_x2, label_y_top + 5),
                         color, -1)
            
            # Draw label text in white for better contrast
            cv2.putText(img_copy, label, (label_x1 + padding, label_y_text),
                       font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
        return img_copy
    
    def process_upload(self, file) -> Tuple[np.ndarray, List[Dict]]:
        """Process uploaded file"""
        # read image
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # convert RGB to BGR for opencv
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # detect objects 
        detections = self.detect(image_bgr)

        return image_bgr, detections
    
    def display_detections(self, image: np.ndarray, detections: List[Dict], window_name: str = "Detections") -> None:
        """
        Display detections in OpenCV window
        
        Args:
            image: Input image
            detections: List of detections
            window_name: Name of the display window
        """
        # Draw detections on image
        annotated_image = self.draw_detections(image, detections)
        
        # Get image dimensions
        img_height, img_width = annotated_image.shape[:2]
        
        # Create resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set fixed window size to 480x480
        window_width = 480
        window_height = 480
        
        cv2.resizeWindow(window_name, window_width, window_height)
        
        # Display image
        cv2.imshow(window_name, annotated_image)
        
        print(f"\nDisplaying {len(detections)} detections")
        print(f"Image: {img_width}x{img_height}, Window: {window_width}x{window_height}")
        print("Press 'q' to quit, 's' to save image, 'f' for fullscreen")
        
        fullscreen = False
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
            elif key == ord('s'):
                # Save image
                timestamp = cv2.getTickCount()
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_image)
                print(f"Saved image to {filename}")
            elif key == ord('f'):
                # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Fullscreen ON")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, window_width, window_height)
                    print("Fullscreen OFF")
        
        # Close window
        cv2.destroyAllWindows()
    
    def detect_and_display(self, image: np.ndarray, window_name: str = "Recycling Detection") -> List[Dict]:
        """
        Detect objects and display results
        
        Args:
            image: Input image
            window_name: Name of the display window
            
        Returns:
            List of detections
        """
        # Run detection
        detections = self.detect(image)
        
        # Display results
        self.display_detections(image, detections, window_name)
        
        return detections

