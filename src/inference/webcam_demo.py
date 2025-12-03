import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

# Direct import when running as script
from detector import RecyclingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamDetector:
    """Real-time webcam detection with YOLO model"""
    
    def __init__(self, model_path: str = "models/teacher/best.pt",  # Restored to best.pt
                 confidence_threshold: float = 0.25,
                 webcam_id: int = 0,
                 inference_size: int = 416):  # Fast inference size
        """
        Initialize webcam detector
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            webcam_id: Webcam device ID (0 for default)
            inference_size: YOLO inference size (416=fast, 640=accurate)
        """
        self.detector = RecyclingDetector(model_path, confidence_threshold, inference_size)
        self.webcam_id = webcam_id
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
        # Detection statistics
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "detections_per_class": {}
        }
        
    def start_camera(self, width: int = 640, height: int = 480) -> bool:
        """
        Start webcam capture
        
        Args:
            width: Frame width (default 640 for speed)
            height: Frame height (default 480 for speed)
            
        Returns:
            bool: True if camera started successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.webcam_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open webcam {self.webcam_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Webcam started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop webcam capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Webcam stopped")
    
    def capture_frames(self):
        """Capture frames from webcam in a separate thread"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop frame if queue is full
                pass
            
            # Update FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
    
    def process_frames(self):
        """Process frames with YOLO detector in a separate thread"""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Run detection with resize enabled for speed
                detections = self.detector.detect(frame, resize_input=True)
                
                # Draw detections
                annotated_frame = self.detector.draw_detections(frame.copy(), detections)
                
                # Get frame height to position text at bottom
                frame_height = annotated_frame.shape[0]
                
                # Add FPS text at bottom
                cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", 
                           (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add detection count at bottom
                cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                           (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update statistics
                self._update_stats(detections)
                
                # Put result in queue
                try:
                    self.result_queue.put((annotated_frame, detections), block=False)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    def _update_stats(self, detections: list):
        """Update detection statistics"""
        self.stats["total_frames"] += 1
        self.stats["total_detections"] += len(detections)
        
        for det in detections:
            class_name = det["class_name"]
            if class_name not in self.stats["detections_per_class"]:
                self.stats["detections_per_class"][class_name] = 0
            self.stats["detections_per_class"][class_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current detection statistics"""
        return {
            **self.stats,
            "fps": self.fps,
            "elapsed_time": time.time() - self.start_time if self.start_time > 0 else 0
        }
    
    def run_single_thread(self, window_name: str = "Recycling Detector", fullscreen: bool = False):
        """
        Run detection in single thread (simpler but slower)
        
        Args:
            window_name: Name of the display window
            fullscreen: Whether to display in fullscreen mode
        """
        if not self.start_camera():
            return
        
        logger.info("Starting single-threaded webcam detection")
        logger.info("Press 'q' to quit, 's' to save frame, 'p' to pause")
        
        # Create window and set size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 480)
        
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        paused = False
        annotated_frame = None
        
        try:
            while self.is_running:
                if not paused:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Run detection
                    start_time = time.time()
                    detections = self.detector.detect(frame)
                    inference_time = time.time() - start_time
                    
                    # Draw detections
                    annotated_frame = self.detector.draw_detections(frame, detections)
                    
                    # Get frame height to position text at bottom
                    frame_height = annotated_frame.shape[0]
                    
                    # Add info overlay at bottom
                    cv2.putText(annotated_frame, f"FPS: {1/inference_time:.1f}" if inference_time > 0 else "FPS: 0", 
                               (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                               (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                if annotated_frame is not None:
                    cv2.imshow(window_name, annotated_frame if not paused else frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('s'):  # Save frame
                    self._save_frame(annotated_frame if not paused else frame, detections)
                elif key == ord('p'):  # Pause/Unpause
                    paused = not paused
                    logger.info(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('r'):  # Reset stats
                    self.stats = {
                        "total_frames": 0,
                        "total_detections": 0,
                        "detections_per_class": {}
                    }
                    logger.info("Statistics reset")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
    
    def _save_frame(self, frame: np.ndarray, detections: list):
        """Save current frame with detections to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs/webcam")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        # Save detection data
        if detections:
            data_file = output_dir / f"detection_{timestamp}.txt"
            with open(data_file, 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Detections: {len(detections)}\n")
                for det in detections:
                    f.write(f"  {det['class_name']}: {det['confidence']:.3f}\n")
        
        logger.info(f"Saved frame to {filepath}")
    
    def run_multi_thread(self, window_name: str = "Recycling Detector", fullscreen: bool = False):
        """
        Run detection with multiple threads (faster)
        
        Args:
            window_name: Name of the display window
            fullscreen: Whether to display in fullscreen mode
        """
        if not self.start_camera():
            return
        
        logger.info("Starting multi-threaded webcam detection")
        logger.info("Press 'q' to quit, 's' to save frame")
        
        # Create window and set size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        
        capture_thread.start()
        process_thread.start()
        
        try:
            while self.is_running:
                try:
                    # Get processed frame from queue
                    annotated_frame, detections = self.result_queue.get(timeout=0.1)
                    
                    # Display frame
                    cv2.imshow(window_name, annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Quit
                        break
                    elif key == ord('s'):  # Save frame
                        self._save_frame(annotated_frame, detections)
                
                except queue.Empty:
                    continue
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop_camera()
            self.is_running = False
            
            # Wait for threads to finish
            capture_thread.join(timeout=1)
            process_thread.join(timeout=1)
            
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final detection statistics"""
        logger.info("\n" + "="*50)
        logger.info("DETECTION STATISTICS")
        logger.info("="*50)
        logger.info(f"Total frames processed: {self.stats['total_frames']}")
        logger.info(f"Total detections: {self.stats['total_detections']}")
        logger.info(f"Average FPS: {self.fps:.1f}")
        
        if self.stats['detections_per_class']:
            logger.info("\nDetections by class:")
            for class_name, count in self.stats['detections_per_class'].items():
                logger.info(f"  {class_name}: {count}")
        
        logger.info("="*50)

def main():
    """Main function to run webcam demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Webcam recycling detection demo")
    parser.add_argument("--model", type=str, default="models/teacher/best.pt",  # Restored to best.pt
                       help="Path to YOLO model")
    parser.add_argument("--webcam", type=int, default=0,
                       help="Webcam device ID")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--multi-thread", action="store_true",
                       help="Use multi-threaded processing")
    parser.add_argument("--width", type=int, default=640,
                       help="Frame width (640 for speed, 1280 for quality)")
    parser.add_argument("--height", type=int, default=480,
                       help="Frame height (480 for speed, 720 for quality)")
    parser.add_argument("--fullscreen", action="store_true",
                       help="Display in fullscreen mode")
    parser.add_argument("--inference-size", type=int, default=416,
                       help="Inference size (320=fastest, 416=fast, 640=accurate)")
    
    args = parser.parse_args()
    
    # Create webcam detector
    detector = WebcamDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        webcam_id=args.webcam,
        inference_size=args.inference_size
    )
    
    # Run detection
    if args.multi_thread:
        detector.run_multi_thread(fullscreen=args.fullscreen)
    else:
        detector.run_single_thread(fullscreen=args.fullscreen)

if __name__ == "__main__":
    main()