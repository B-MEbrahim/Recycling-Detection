from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

class Detection(BaseModel):
    """Single detection result"""
    bbox: List[float]
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    """Response model for detection endpoint"""
    detections: List[Detection]
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image")
    image_size: Optional[str] = Field(None, description="Original image dimensions")
    model_used: str = Field("teacher", description="Model used for inference")
    inference_time: Optional[float] = Field(None, description="Inference time in seconds")

class BatchDetectionRequest(BaseModel):
    """Request model for batch detection"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    confidence_threshold: Optional[float] = Field(0.25, ge=0, le=1)

class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    total_images: int
    results: List[Dict[str, Any]]
    model_used: str

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    path: Optional[str]
    status: str
    parameters: Optional[int]
    size_mb: Optional[float]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str
    version: str = "1.0.0"