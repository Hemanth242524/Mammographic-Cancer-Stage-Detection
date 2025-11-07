# Mammographic_Cancer_Stage_Detection
Modular pipeline for stage-wise mammographic cancer detection with pectoral muscle removal, self-supervised pretraining, multimodal fusion, temporal tracking and explainability.
Folder contains modular scripts/notebooks (Python scripts ready to convert to Colab notebooks) and utility modules.

## Structure
- 01_unet_muscle_removal.py           : Pre-trained U-Net inference (pectoral removal)
- 02_simclr_pretrain.py               : SimCLR self-supervised pretraining script (encoder)
- 03_multimodal_fusion_train.py       : Fine-tune encoder + multimodal fusion (radiomics + clinical)
- 04_temporal_tracking.py             : Siamese & sequence temporal models
- 05_explainability_eval.py           : Grad-CAM and SHAP utilities for explainability
- utils/radiomics_utils.py            : Radiomic feature extraction helpers (GLCM etc.)
- utils/data_prep.py                  : Data loading and patient-level split helpers
- configs/config.yaml                 : Configuration (paths, hyperparams)
- models/                             : place where model weights will be saved


## How to use (Colab ready)
1. Upload your raw mammogram images to the Colab session or mount Google Drive:
   - Put images in a folder and set `INPUT_DIR` in `configs/config.yaml`.
2. Run `01_unet_muscle_removal.py` to generate processed images (pectoral removed).
3. (Optional) Run `02_simclr_pretrain.py` to pretrain encoder on unlabeled mammograms.
   - If you skip, you can use ImageNet pretrained encoder in step 3.
4. Prepare CSV with clinical & radiomic features (see utils/data_prep.py).
5. Run `03_multimodal_fusion_train.py` to train stage classifier.
6. Run `04_temporal_tracking.py` if you have patient time-series data.
7. Run `05_explainability_eval.py` to generate Grad-CAM and SHAP outputs.


This package is intended to be modular so you can run each stage independently (Option B from the project plan).

## Web Application

A Flask-based web interface is available for easy interaction with the mammographic cancer detection system.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **If you encounter OpenCV/NumPy compatibility errors**, see [FIX_INSTALLATION.md](FIX_INSTALLATION.md) for troubleshooting steps.

2. (Optional) Train the stage classifier model:
   - Follow steps 1-5 in "How to use" section above
   - The trained model should be saved at `models/stage_classifier.h5`
   - If the model is not available, the web app will run in demo mode with mock predictions

3. Run the web application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

### Web App Features

- **Image Upload**: Upload mammogram images (PNG, JPG, JPEG, TIFF, DICOM)
- **Clinical Data Input**: Enter patient age and breast density category
- **Automatic Processing**:
  - Pectoral muscle removal using U-Net
  - Radiomic feature extraction
  - Cancer stage prediction
- **Results Display**:
  - Original, processed, and mask images
  - Stage classification (Stage 0-IV) with confidence scores
  - Binary classification (Benign/Malignant)
  - Radiomic features visualization
  - Stage probability distribution

### Directory Structure (Web App)

- `app.py` - Flask application main file (for local development)
- `templates/index.html` - Web interface template (Flask version)
- `static/css/style.css` - Custom styling
- `static/js/main.js` - Client-side JavaScript (Flask version)
- `public/` - Static files for Netlify deployment
- `netlify.toml` - Netlify configuration
- `netlify/functions/` - Serverless functions for Netlify
- `uploads/` - Temporary storage for uploaded images
- `processed/` - Processed images storage
- `results/` - Results storage

## Netlify Deployment

The web application can be deployed to Netlify for free hosting. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Quick Deploy to Netlify

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Netlify deployment"
   git push origin main
   ```

2. **Connect to Netlify:**
   - Go to [https://app.netlify.com](https://app.netlify.com)
   - Click "Add new site" → "Import an existing project"
   - Select your GitHub repository
   - Netlify will auto-detect settings from `netlify.toml`

3. **Deploy:**
   - Click "Deploy site"
   - Your site will be live in 1-2 minutes!

**Note:** The Netlify version uses a simplified processing pipeline due to serverless function limitations (10s timeout, 1.5GB memory). For full functionality with U-Net and trained models, consider deploying the Flask backend separately (see DEPLOYMENT.md for alternatives).
