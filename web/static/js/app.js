// Mode switching
function switchMode(mode) {
    const uploadMode = document.getElementById('uploadMode');
    const webcamMode = document.getElementById('webcamMode');
    const uploadBtn = document.getElementById('uploadModeBtn');
    const webcamBtn = document.getElementById('webcamModeBtn');

    if (mode === 'upload') {
        uploadMode.classList.add('active');
        webcamMode.classList.remove('active');
        uploadBtn.classList.add('active');
        webcamBtn.classList.remove('active');
        stopWebcam();
    } else {
        uploadMode.classList.remove('active');
        webcamMode.classList.add('active');
        uploadBtn.classList.remove('active');
        webcamBtn.classList.add('active');
        resetUpload();
    }
}

// Upload Mode Functions
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadedImage = document.getElementById('uploadedImage');
const previewSection = document.getElementById('previewSection');
const uploadResults = document.getElementById('uploadResults');

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileUpload(file);
    }
});

uploadArea.addEventListener('click', (e) => {
    // Prevent double-trigger when clicking on the label or input
    if (e.target.tagName === 'LABEL' || e.target.tagName === 'INPUT') {
        return;
    }
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.src = e.target.result;
        previewSection.style.display = 'block';
        uploadResults.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function detectFromUpload() {
    const loading = document.getElementById('uploadLoading');
    const detectBtn = document.getElementById('detectBtn');
    
    loading.style.display = 'flex';
    detectBtn.disabled = true;

    try {
        // Convert image to base64
        const imageData = uploadedImage.src;
        
        // Send to API
        const response = await fetch('/api/v1/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Detection result:', result);
        
        // Display results
        displayResults(result, 'upload');
        
    } catch (error) {
        console.error('Detection error:', error);
        alert(`Error detecting objects: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        detectBtn.disabled = false;
    }
}

function resetUpload() {
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadResults.style.display = 'none';
}

// Webcam Mode Functions
let webcamStream = null;
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamResults = document.getElementById('webcamResults');

async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        webcamVideo.srcObject = webcamStream;
        
        document.getElementById('startWebcamBtn').style.display = 'none';
        document.getElementById('stopWebcamBtn').style.display = 'inline-block';
        document.getElementById('captureBtn').style.display = 'inline-block';
        
    } catch (error) {
        console.error('Webcam error:', error);
        alert('Unable to access webcam. Please check permissions.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamVideo.srcObject = null;
        webcamStream = null;
    }
    
    document.getElementById('startWebcamBtn').style.display = 'inline-block';
    document.getElementById('stopWebcamBtn').style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
    webcamResults.style.display = 'none';
}

async function captureFrame() {
    const loading = document.getElementById('webcamLoading');
    
    // Capture frame from video
    const context = webcamCanvas.getContext('2d');
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    context.drawImage(webcamVideo, 0, 0);
    
    const imageData = webcamCanvas.toDataURL('image/jpeg');
    
    loading.style.display = 'flex';
    
    try {
        // Send to API
        const response = await fetch('/api/v1/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Detection result:', result);
        
        // Display results
        displayResults(result, 'webcam');
        
    } catch (error) {
        console.error('Detection error:', error);
        alert(`Error detecting objects: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// Display results
function displayResults(result, mode) {
    const prefix = mode === 'upload' ? 'upload' : 'webcam';
    const resultsSection = document.getElementById(`${prefix}Results`);
    const resultImage = document.getElementById(`${prefix}ResultImage`);
    const detectionsList = document.getElementById(`${prefix}DetectionsList`);
    
    // Debug: Check if elements exist
    if (!resultsSection || !resultImage || !detectionsList) {
        console.error('Missing elements:', {
            resultsSection: `${prefix}Results`,
            resultImage: `${prefix}ResultImage`,
            detectionsList: `${prefix}DetectionsList`
        });
        alert('Error: UI elements not found');
        return;
    }
    
    // Show annotated image
    if (result.annotated_image) {
        resultImage.src = result.annotated_image;
    } else {
        console.error('No annotated image in response');
    }
    
    // Show detections list
    detectionsList.innerHTML = '';
    
    if (result.detections && result.detections.length > 0) {
        result.detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <span class="class-name">${det.class_name}</span>
                <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
            `;
            detectionsList.appendChild(item);
        });
    } else {
        detectionsList.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 20px;">No objects detected</p>';
    }
    
    resultsSection.style.display = 'block';
    
    // Show stats if available
    if (result.inference_time) {
        console.log(`Inference time: ${(result.inference_time * 1000).toFixed(0)}ms`);
    }
    if (result.image_size) {
        console.log(`Image size: ${result.image_size}`);
    }
}
