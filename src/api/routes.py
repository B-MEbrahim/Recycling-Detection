from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import io
import json
from PIL import Image
import base64
import time

from .schemas import DetectionResponse, BatchDetectionRequest
from src.inference.detector import RecyclingDetector

router = APIRouter()

# Global detector instance (in production, use dependency injection)
detector = None

def get_detector():
    """Dependency to get detector instance"""
    global detector
    if detector is None:
        detector = RecyclingDetector(model_path="models/teacher/best.pt")
    return detector

@router.post("/detect/upload", response_model=DetectionResponse)
async def detect_upload(
    file: UploadFile = File(...),
    detector: RecyclingDetector = Depends(get_detector)
):
    """
    Upload an image for recycling object detection
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect objects
        detections = detector.detect(image)
        
        # Draw detections on image
        annotated_image = detector.draw_detections(image, detections)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return DetectionResponse(
            detections=detections,
            annotated_image=annotated_base64,
            image_size=f"{image.shape[1]}x{image.shape[0]}",
            model_used="teacher"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/batch")
async def detect_batch(
    request: BatchDetectionRequest,
    detector: RecyclingDetector = Depends(get_detector)
):
    """
    Process multiple images (base64 encoded)
    """
    results = []
    
    for i, image_base64 in enumerate(request.images):
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64.split(",")[-1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect objects
            detections = detector.detect(image)
            
            results.append({
                "image_id": i,
                "detections": detections,
                "count": len(detections)
            })
            
        except Exception as e:
            results.append({
                "image_id": i,
                "error": str(e),
                "detections": []
            })
    
    return {
        "total_images": len(request.images),
        "results": results,
        "model_used": "teacher"
    }

class DetectRequest(BaseModel):
    """Request model for base64 image detection"""
    image: str

@router.post("/detect", response_model=DetectionResponse)
async def detect_from_base64(
    request: DetectRequest,
    detector: RecyclingDetector = Depends(get_detector)
):
    """
    Detect objects from base64 encoded image (for web frontend)
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect objects
        detections = detector.detect(image)
        
        # Draw detections on image
        annotated_image = detector.draw_detections(image, detections)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        inference_time = time.time() - start_time
        
        return DetectionResponse(
            detections=detections,
            annotated_image=f"data:image/jpeg;base64,{annotated_base64}",
            image_size=f"{image.shape[1]}x{image.shape[0]}",
            model_used="teacher",
            inference_time=inference_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models():
    """
    Get list of available models
    """
    import os
    
    models = []
    models_dir = "models"
    
    for model_type in ["teacher", "student", "distilled"]:
        model_path = os.path.join(models_dir, model_type, "best.pt")
        if os.path.exists(model_path):
            models.append({
                "name": model_type,
                "path": model_path,
                "status": "available"
            })
        else:
            models.append({
                "name": model_type,
                "status": "not_available"
            })
    
    return {"models": models}

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "recycling-detection-api"
    }

@router.get("/stats")
async def get_detection_stats():
    """
    Get basic detection statistics
    """
    # In production, this would query a database
    # For now, return mock data
    return {
        "total_detections": 1500,
        "most_common_class": "plastic_bottle",
        "average_confidence": 0.78,
        "api_version": "1.0"
    }