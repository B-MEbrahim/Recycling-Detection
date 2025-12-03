import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import os

# Direct import when running as script
from detector import RecyclingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process images for recycling detection"""
    
    def __init__(self, detector: RecyclingDetector = None):
        """
        Initialize image processor
        
        Args:
            detector: RecyclingDetector instance (optional)
        """
        self.detector = detector or RecyclingDetector()
        
        # Color palette for different classes
        self.color_palette = {
            'plastic': (255, 0, 0),      # Red
            'paper': (0, 255, 0),        # Green
            'glass': (0, 0, 255),        # Blue
            'metal': (255, 255, 0),      # Yellow
            'cardboard': (255, 0, 255),  # Magenta
            'trash': (0, 255, 255),      # Cyan
            'default': (255, 165, 0)     # Orange
        }
        
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy.ndarray: Loaded image or None if failed
        """
        try:
            if isinstance(image_path, Path):
                image_path = str(image_path)
            
            # Read image using OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load image from bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            numpy.ndarray: Loaded image or None if failed
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image from bytes")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess image for detection
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        processed = image.copy()
        
        # Resize if target size specified
        if target_size:
            width, height = target_size
            processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_AREA)
        
        # Normalize (if needed by model)
        # Note: YOLO handles normalization internally
        
        return processed
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get color for a class
        
        Args:
            class_name: Name of the class
            
        Returns:
            tuple: BGR color tuple
        """
        class_lower = class_name.lower()
        
        # Check for keywords in class name
        for key, color in self.color_palette.items():
            if key in class_lower:
                return color
        
        return self.color_palette['default']
    
    def draw_detections_pil(self, image: np.ndarray, detections: List[Dict]) -> Image.Image:
        """
        Draw detections using PIL (better text rendering)
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            PIL.Image: Image with drawn detections
        """
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get color for this class
            color = self.get_class_color(class_name)
            color_rgb = color[::-1]  # BGR to RGB
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(label_bbox, fill=color_rgb)
            
            # Draw label text
            draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
        
        return pil_image
    
    def draw_detections_opencv(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detections using OpenCV (faster)
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            numpy.ndarray: Image with drawn detections
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get color for this class
            color = self.get_class_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def process_single_image(self, image_path: Union[str, Path], 
                            save_output: bool = True,
                            output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Process a single image file
        
        Args:
            image_path: Path to image file
            save_output: Whether to save output images
            output_dir: Directory to save outputs
            
        Returns:
            dict: Processing results
        """
        results = {
            "success": False,
            "image_path": str(image_path),
            "detections": [],
            "output_path": None,
            "error": None
        }
        
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                results["error"] = "Failed to load image"
                return results
            
            # Run detection
            detections = self.detector.detect(image)
            results["detections"] = detections
            
            # Draw detections
            annotated_image = self.draw_detections_opencv(image, detections)
            
            # Save output if requested
            if save_output:
                output_path = self._save_processed_image(
                    image_path, annotated_image, detections, output_dir
                )
                results["output_path"] = str(output_path)
            
            results["success"] = True
            results["image_size"] = f"{image.shape[1]}x{image.shape[0]}"
            results["detection_count"] = len(detections)
            
            logger.info(f"Processed {image_path}: {len(detections)} detections")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error processing {image_path}: {e}")
        
        return results
    
    def process_batch(self, image_paths: List[Union[str, Path]],
                     output_dir: str = "outputs/batch") -> List[Dict[str, Any]]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save outputs
            
        Returns:
            list: List of processing results
        """
        results = []
        
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(
                image_path, 
                save_output=True,
                output_dir=output_dir
            )
            
            results.append(result)
        
        # Generate summary
        successful = sum(1 for r in results if r["success"])
        total_detections = sum(len(r["detections"]) for r in results if r["success"])
        
        logger.info(f"Batch processing complete: {successful}/{len(image_paths)} successful, "
                   f"{total_detections} total detections")
        
        return results
    
    def _save_processed_image(self, original_path: Union[str, Path], 
                            annotated_image: np.ndarray,
                            detections: List[Dict],
                            output_dir: str) -> Path:
        """
        Save processed image and detection data
        
        Args:
            original_path: Original image path
            annotated_image: Annotated image
            detections: List of detections
            output_dir: Output directory
            
        Returns:
            Path: Path to saved image
        """
        # Create output directory
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        original_path = Path(original_path)
        timestamp = original_path.stem
        output_image_path = output_dir_path / f"{timestamp}_detected.jpg"
        
        # Save annotated image
        cv2.imwrite(str(output_image_path), annotated_image)
        
        # Save detection data
        if detections:
            output_data_path = output_dir_path / f"{timestamp}_detections.txt"
            with open(output_data_path, 'w') as f:
                f.write(f"Image: {original_path.name}\n")
                f.write(f"Detections: {len(detections)}\n")
                f.write("=" * 50 + "\n")
                
                for det in detections:
                    f.write(f"Class: {det['class_name']}\n")
                    f.write(f"Confidence: {det['confidence']:.3f}\n")
                    f.write(f"BBox: {det['bbox']}\n")
                    f.write("-" * 30 + "\n")
        
        logger.info(f"Saved output to {output_image_path}")
        return output_image_path
    
    def image_to_base64(self, image: np.ndarray, format: str = "JPEG") -> str:
        """
        Convert image to base64 string
        
        Args:
            image: Input image
            format: Image format (JPEG, PNG)
            
        Returns:
            str: Base64 encoded image
        """
        # Convert BGR to RGB if needed
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
    
    def create_detection_summary_image(self, detections: List[Dict], 
                                      size: Tuple[int, int] = (400, 300)) -> Image.Image:
        """
        Create a summary image showing detection statistics
        
        Args:
            detections: List of detections
            size: Output image size
            
        Returns:
            PIL.Image: Summary image
        """
        # Create blank image
        summary_img = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(summary_img)
        
        # Try to load font
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            text_font = ImageFont.truetype("arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Draw title
        draw.text((20, 20), "Detection Summary", fill=(0, 0, 0), font=title_font)
        
        # Count detections by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Draw statistics
        y_offset = 70
        for i, (class_name, count) in enumerate(class_counts.items()):
            color = self.get_class_color(class_name)
            color_rgb = color[::-1]  # BGR to RGB
            
            # Draw color square
            draw.rectangle([20, y_offset, 40, y_offset + 20], fill=color_rgb)
            
            # Draw text
            text = f"{class_name}: {count}"
            draw.text((50, y_offset), text, fill=(0, 0, 0), font=text_font)
            
            y_offset += 30
        
        # Draw total
        draw.text((20, y_offset + 20), f"Total: {len(detections)}", 
                 fill=(0, 0, 0), font=text_font)
        
        return summary_img

def main():
    """Main function for standalone image processing"""
    
    # ===== CONFIGURATION - Edit these values =====
    MODEL_PATH = "models/teacher/best.pt"
    IMAGE_PATH = r"images\trash\trash119_jpg.rf.e630cbc5784875924175a4660ac8b205.jpg"  
    DISPLAY_MODE = True  # True = display with OpenCV, False = save to file
    OUTPUT_DIR = "outputs"
    # ============================================
    
    # Create processor
    detector = RecyclingDetector(model_path=MODEL_PATH)
    processor = ImageProcessor(detector)
    
    # Normalize path
    image_path = Path(IMAGE_PATH.replace('\\', os.sep))
    
    # Check if path exists
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Absolute path attempted: {image_path.absolute()}")
        return
    
    # Load image
    image = processor.load_image(image_path)
    if image is None:
        print("Failed to load image")
        return
    
    if DISPLAY_MODE:
        # Display mode - show with OpenCV
        print(f"Processing and displaying: {image_path.name}")
        detections = detector.detect_and_display(image, f"Detection: {image_path.name}")
        
        if detections:
            print(f"\nFound {len(detections)} detections:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']}: {det['confidence']:.3f}")
        else:
            print("No detections found")
    else:
        # Save mode - process and save
        print(f"Processing and saving: {image_path.name}")
        result = processor.process_single_image(image_path, output_dir=OUTPUT_DIR)
        
        if result["success"]:
            print(f"\n{'='*50}")
            print("PROCESSING COMPLETE")
            print(f"{'='*50}")
            print(f"Input: {result['image_path']}")
            print(f"Size: {result['image_size']}")
            print(f"Detections: {result['detection_count']}")
            print(f"Output: {result['output_path']}")
            
            if result["detections"]:
                print(f"\nFound {len(result['detections'])} detections:")
                for i, det in enumerate(result["detections"], 1):
                    print(f"  {i}. {det['class_name']}: {det['confidence']:.3f}")
            print(f"{'='*50}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()