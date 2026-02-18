# V500 RetinaGuard - Retinitis Pigmentosa Detection System

## Setup Instructions

### 1. Install Python Dependencies

#### Option A: Using the batch file (Windows)
```bash
install_dependencies.bat
```

#### Option B: Manual installation
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 2. Choose Your Deep Learning Framework

The `requirements.txt` includes TensorFlow by default. If you prefer PyTorch:

```bash
# For PyTorch (uncomment in requirements.txt or install separately)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Directory Structure

Create these folders for your models:
```
V500/
├── app.py                  # Flask AI server
├── server.js              # Node.js database server (existing)
├── requirements.txt       # Python dependencies
├── models/               # Your trained ML models
│   ├── vessels_model.h5
│   ├── macula_model.h5
│   ├── drusen_model.h5
│   ├── nerves_model.h5
│   ├── bleeds_model.h5
│   ├── scar_model.h5
│   └── verdict_model.h5
├── uploads/              # Temporary image uploads (auto-created)
└── public/
    └── index.html        # Frontend UI
```

### 4. Running the System

You need to run TWO servers:

#### Terminal 1 - Node.js Database Server (MongoDB)
```bash
node server.js
# Runs on http://localhost:5000
```

#### Terminal 2 - Flask AI Server
```bash
python app.py
# Runs on http://localhost:5001
```

### 5. Integration with Frontend

Update the frontend to call Flask API instead of using static data. Change in `index.html`:

```javascript
// In handleFileSelect() function, after file is loaded:
async function callFlaskAPI(base64Image) {
    const response = await fetch('http://localhost:5001/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: base64Image,
            patientId: currentPatientName
        })
    });
    const data = await response.json();
    return data;
}
```

## Package Descriptions

### Core Framework
- **Flask**: Python web framework for AI API
- **flask-cors**: Handle cross-origin requests from frontend

### Deep Learning
- **TensorFlow**: Primary ML framework (or PyTorch)
- **Keras**: High-level API (included in TensorFlow 2.x)

### Image Processing
- **opencv-python**: Computer vision operations
- **Pillow**: Image file handling
- **numpy**: Numerical operations
- **scikit-image**: Advanced image processing

### Medical Imaging
- **pydicom**: DICOM medical image format support
- **SimpleITK**: Advanced medical image processing

### Data Science
- **pandas**: Data manipulation
- **scikit-learn**: ML utilities and preprocessing

## Model Training Resources

For Retinitis Pigmentosa detection, consider:

1. **Datasets**:
   - ODIR (Ocular Disease Intelligent Recognition)
   - DRIVE (Digital Retinal Images for Vessel Extraction)
   - Custom RP-specific datasets from medical institutions

2. **Key Features to Detect**:
   - Bone spicule pigmentation
   - Vascular attenuation (narrowed blood vessels)
   - Optic disc pallor
   - Retinal atrophy
   - Loss of peripheral vision indicators

3. **Model Architectures**:
   - ResNet50/101 for classification
   - U-Net for segmentation
   - EfficientNet for efficiency
   - Vision Transformers (ViT) for state-of-art

## Testing

Test the Flask API:
```bash
curl http://localhost:5001/api/health
```

Expected response:
```json
{
  "status": "online",
  "message": "Flask AI Server Running"
}
```

## Troubleshooting

### Port Already in Use
```bash
# Change port in app.py:
app.run(host='0.0.0.0', port=5002, debug=True)
```

### CORS Errors
- Ensure both servers are running
- Check firewall settings
- Verify CORS is enabled in Flask

### Memory Issues with Large Models
```bash
# Use model quantization or compression
pip install tensorflow-model-optimization
```

## Production Considerations

1. **Security**: Add authentication/API keys
2. **Performance**: Use Gunicorn instead of Flask dev server
3. **Monitoring**: Add logging and error tracking
4. **Scaling**: Consider containerization (Docker)
5. **GPU**: Ensure CUDA/cuDNN for TensorFlow GPU support
