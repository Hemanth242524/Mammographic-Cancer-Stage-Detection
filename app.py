"""
Flask web application for Mammographic Cancer Stage Detection
Provides a user-friendly interface for uploading mammograms and getting stage predictions
"""
import os
import cv2
import numpy as np
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from utils.radiomics_utils import extract_basic_radiomics

# Try to import segmentation_models, but handle gracefully if it fails
try:
    import segmentation_models as sm
    SEGMENTATION_MODELS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: segmentation_models not available ({e}). U-Net functionality will be limited.")
    SEGMENTATION_MODELS_AVAILABLE = False
    sm = None
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Load configuration
cfg_path = 'configs/config.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = cfg.get('img_size', 256)

# Global variables for models (loaded on first use)
unet_model = None
stage_classifier_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_unet_model():
    """Load U-Net model for pectoral muscle removal"""
    global unet_model
    if unet_model is None:
        if not SEGMENTATION_MODELS_AVAILABLE:
            # Fallback: Create a simple U-Net-like model using TensorFlow/Keras
            print("Using fallback U-Net model (segmentation_models not available)")
            from tensorflow.keras import layers, Model
            from tensorflow.keras.applications import EfficientNetB0
            
            # Encoder
            base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
            base_model.trainable = False
            
            # Simple decoder for binary segmentation
            inputs = base_model.input
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(IMG_SIZE * IMG_SIZE, activation='sigmoid')(x)
            x = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)
            
            unet_model = Model(inputs, x)
            return unet_model
        
        try:
            sm.set_framework('tf.keras')
            ENCODER = 'efficientnetb0'
            unet_model = sm.Unet(
                backbone_name=ENCODER,
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                encoder_weights='imagenet',
                classes=1,
                activation='sigmoid'
            )
        except Exception as e:
            print(f"Error loading segmentation_models U-Net: {e}")
            print("Falling back to simple model")
            # Use the same fallback as above
            from tensorflow.keras import layers, Model
            from tensorflow.keras.applications import EfficientNetB0
            
            base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
            base_model.trainable = False
            
            inputs = base_model.input
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(IMG_SIZE * IMG_SIZE, activation='sigmoid')(x)
            x = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)
            
            unet_model = Model(inputs, x)
    return unet_model

def load_stage_classifier():
    """Load stage classifier model"""
    global stage_classifier_model
    if stage_classifier_model is None:
        model_path = 'models/stage_classifier.h5'
        if os.path.exists(model_path):
            stage_classifier_model = load_model(model_path)
        else:
            # Return None if model doesn't exist (for demo mode)
            return None
    return stage_classifier_model

def preprocess_image(img_path, size=IMG_SIZE):
    """Load and preprocess image for model"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    img_normalized = img_resized.astype('float32') / 255.0
    return img_normalized, img_rgb

def predict_pectoral_mask(model, img):
    """Predict pectoral muscle mask"""
    x = np.expand_dims(img, 0)
    pred = model.predict(x, verbose=0)[0, :, :, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred

def postprocess_mask(mask, threshold=0.4):
    """Post-process mask to remove small artifacts"""
    b = (mask >= threshold).astype('uint8') * 255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = b.shape
    keep = np.zeros_like(b)
    min_area = 0.001 * h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if y < h * 0.35:
            cv2.drawContours(keep, [cnt], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)
    return keep

def remove_pectoral(orig_img, mask_resized):
    """Remove pectoral muscle from original image"""
    H, W = orig_img.shape[:2]
    mask = cv2.resize(mask_resized, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bool = (mask > 0).astype('uint8')
    out = orig_img.copy()
    out[mask_bool == 255] = 0
    return out

def predict_stage(img_array, radiomics_features, clinical_features=None):
    """Predict cancer stage from processed image and features"""
    model = load_stage_classifier()
    if model is None:
        # Demo mode - return mock predictions
        return {
            'stage': 'Stage 0',
            'confidence': 0.75,
            'binary': 'Benign',
            'binary_confidence': 0.65,
            'demo': True
        }
    
    # Prepare tabular features
    if clinical_features is None:
        clinical_features = [50, 2]  # Default age=50, density=2
    
    tabular_features = np.concatenate([radiomics_features, clinical_features])
    tabular_features = np.expand_dims(tabular_features, 0)
    
    # Predict
    img_input = np.expand_dims(img_array, 0)
    predictions = model.predict([img_input, tabular_features], verbose=0)
    
    stage_probs = predictions[0][0]
    binary_prob = predictions[1][0][0]
    
    stage_idx = np.argmax(stage_probs)
    stage_names = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
    
    return {
        'stage': stage_names[stage_idx],
        'confidence': float(stage_probs[stage_idx]),
        'binary': 'Malignant' if binary_prob > 0.5 else 'Benign',
        'binary_confidence': float(binary_prob),
        'all_stage_probs': {stage_names[i]: float(stage_probs[i]) for i in range(5)},
        'demo': False
    }

def image_to_base64(img_array):
    """Convert numpy image array to base64 string"""
    # Handle different image formats
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype('uint8')
    else:
        img_array = img_array.astype('uint8')
    
    # Handle grayscale vs RGB
    if len(img_array.shape) == 2:
        img_pil = Image.fromarray(img_array, mode='L')
    elif len(img_array.shape) == 3:
        img_pil = Image.fromarray(img_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
    
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get clinical features from form
        age = request.form.get('age', type=int)
        density = request.form.get('density', type=int)
        clinical_features = [age if age else 50, density if density else 2]
        
        # Process image
        img_normalized, img_original = preprocess_image(filepath, IMG_SIZE)
        if img_normalized is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Step 1: Remove pectoral muscle
        unet = load_unet_model()
        mask_pred = predict_pectoral_mask(unet, img_normalized)
        mask_post = postprocess_mask(mask_pred)
        img_processed = remove_pectoral(img_original, mask_post)
        img_processed_normalized = img_processed.astype('float32') / 255.0
        img_processed_resized = cv2.resize(img_processed_normalized, (IMG_SIZE, IMG_SIZE))
        
        # Step 2: Extract radiomics features
        radiomics = extract_basic_radiomics(img_processed)
        
        # Step 3: Predict stage
        prediction = predict_stage(img_processed_resized, radiomics, clinical_features)
        
        # Convert images to base64 for display
        original_b64 = image_to_base64(img_normalized)
        processed_b64 = image_to_base64(img_processed_resized)
        mask_b64 = image_to_base64(mask_post.astype('float32') / 255.0)
        
        # Save results
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'processed_image': processed_b64,
            'mask_image': mask_b64,
            'prediction': prediction,
            'radiomics': {
                'contrast': float(radiomics[0]),
                'homogeneity': float(radiomics[1]),
                'energy': float(radiomics[2]),
                'entropy': float(radiomics[3])
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Mammographic Cancer Stage Detection Web Application...")
    print("Access the application at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

