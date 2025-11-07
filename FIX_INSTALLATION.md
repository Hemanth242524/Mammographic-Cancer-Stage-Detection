# Fixing Installation Issues

## OpenCV and NumPy Compatibility Error

If you encounter errors like:
- `AttributeError: _ARRAY_API not found`
- `ImportError: numpy.core.multiarray failed to import`

This is typically caused by version incompatibility between NumPy and OpenCV.

## Solution 1: Reinstall with Compatible Versions (Recommended)

1. **Uninstall conflicting packages:**
   ```bash
   pip uninstall opencv-python opencv-contrib-python numpy -y
   ```

2. **Reinstall with compatible versions:**
   ```bash
   pip install numpy==1.24.3
   pip install opencv-python==4.8.1.78
   pip install -r requirements.txt
   ```

## Solution 2: Use a Virtual Environment (Best Practice)

1. **Create a new virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

4. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Solution 3: If TensorFlow Installation Fails

If you get TensorFlow version errors, try:

1. **For Python 3.11+:**
   ```bash
   pip install tensorflow>=2.20.0
   ```

2. **For Python 3.10 or earlier:**
   ```bash
   pip install tensorflow==2.15.0
   ```
   Then update `requirements.txt` to use `tensorflow==2.15.0` instead of `>=2.20.0`

## Solution 4: Clean Install (Nuclear Option)

If nothing else works:

1. **Uninstall all packages:**
   ```bash
   pip freeze > installed_packages.txt
   pip uninstall -r installed_packages.txt -y
   ```

2. **Create fresh virtual environment:**
   ```bash
   python -m venv venv_new
   venv_new\Scripts\activate  # Windows
   # or
   source venv_new/bin/activate  # Linux/Mac
   ```

3. **Install from scratch:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Verify Installation

After installation, verify everything works:

```python
python -c "import cv2; import numpy; import tensorflow; print('All imports successful!')"
```

If this runs without errors, you're good to go!

## Common Issues

### Issue: "No module named 'cv2'"
**Solution:** Make sure you installed `opencv-python` (not just `opencv`)

### Issue: "Segmentation models import error"
**Solution:** 
```bash
pip install segmentation-models==1.0.1 --no-deps
pip install tensorflow keras-applications
```

### Issue: "TensorFlow not found"
**Solution:** Check your Python version. TensorFlow 2.20+ requires Python 3.11+

## Still Having Issues?

1. Check your Python version: `python --version`
2. Check pip version: `pip --version`
3. Try installing packages one by one to identify the problematic package
4. Check for conflicting packages: `pip list | grep -i opencv`

