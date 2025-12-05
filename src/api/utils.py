import time
import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    """Simple timer context manager"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed_time:.3f} seconds")

def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Convert OpenCV image to base64 string
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to bytes
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(image_base64: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
        
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        raise

def validate_image_file(file_content: bytes) -> bool:
    """
    Validate if the uploaded file is a valid image
    """
    try:
        # Try to decode as image
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False
        
        # Check image dimensions are reasonable
        if img.shape[0] > 5000 or img.shape[1] > 5000:
            logger.warning(f"Image too large: {img.shape}")
            return False
            
        return True
        
    except Exception:
        return False

def format_detection_results(detections: List[Dict]) -> Dict:
    """
    Format detection results for API response
    """
    if not detections:
        return {
            "total_detections": 0,
            "detections_by_class": {},
            "average_confidence": 0
        }
    
    # Group by class
    detections_by_class = {}
    total_confidence = 0
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        
        if class_name not in detections_by_class:
            detections_by_class[class_name] = {
                "count": 0,
                "avg_confidence": 0,
                "confidences": []
            }
        
        detections_by_class[class_name]["count"] += 1
        detections_by_class[class_name]["confidences"].append(confidence)
        total_confidence += confidence
    
    # Calculate average confidence per class
    for class_name, stats in detections_by_class.items():
        confidences = stats["confidences"]
        stats["avg_confidence"] = sum(confidences) / len(confidences)
        del stats["confidences"]  # Remove raw data for cleaner response
    
    return {
        "total_detections": len(detections),
        "detections_by_class": detections_by_class,
        "average_confidence": total_confidence / len(detections)
    }

def resize_image_if_needed(image: np.ndarray, max_dimension: int = 1280) -> np.ndarray:
    """
    Resize image if it's too large, maintaining aspect ratio
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dimension:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    return resized

def create_detection_thumbnail(image: np.ndarray, detections: List[Dict], 
                              thumbnail_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Create a thumbnail with detections for quick preview
    """
    # Create a copy of the image
    thumbnail = image.copy()
    
    # Resize to thumbnail size
    thumbnail = cv2.resize(thumbnail, thumbnail_size, interpolation=cv2.INTER_AREA)
    
    # Scale bounding boxes for thumbnail
    scale_x = thumbnail_size[0] / image.shape[1]
    scale_y = thumbnail_size[1] / image.shape[0]
    
    # Draw scaled bounding boxes
    for det in detections[:5]:  # Limit to 5 detections for thumbnail
        x1, y1, x2, y2 = det['bbox']
        
        # Scale coordinates
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        # Draw rectangle
        cv2.rectangle(thumbnail, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return thumbnail