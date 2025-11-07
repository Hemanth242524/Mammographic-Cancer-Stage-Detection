"""01_unet_muscle_removal.py
Run a pre-trained U-Net segmentation model (encoder pretrained on ImageNet) to infer pectoral muscle mask and remove it.
This script is Colab-ready. Change INPUT_DIR/OUTPUT_DIR paths in config or below.
"""
import os, cv2, numpy as np, argparse, yaml
from glob import glob
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import img_to_array

# Load config
cfg_path = 'configs/config.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

INPUT_DIR = cfg.get('input_dir', '/content/raw_mammograms')
OUTPUT_DIR = cfg.get('processed_dir', '/content/processed_mammograms')
IMG_SIZE = cfg.get('img_size', 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sm.set_framework('tf.keras')
sm.framework()

ENCODER = 'efficientnetb0'
model = sm.Unet(backbone_name=ENCODER, input_shape=(IMG_SIZE, IMG_SIZE, 3), encoder_weights='imagenet', classes=1, activation='sigmoid')

def read_and_resize(path, size=IMG_SIZE):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

def preprocess_for_model(img):
    arr = img.astype('float32')/255.0
    return arr

def predict_mask(model, img):
    x = preprocess_for_model(img)
    x = np.expand_dims(x,0)
    pred = model.predict(x)[0,:,:,0]
    pred = (pred - pred.min())/(pred.max()-pred.min()+1e-8)
    return pred

def postprocess_mask(mask, threshold=0.4):
    b = (mask>=threshold).astype('uint8')*255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = b.shape
    keep = np.zeros_like(b)
    min_area = 0.001*h*w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        if y < h*0.35:
            cv2.drawContours(keep, [cnt], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)
    return keep

def remove_pectoral_from_original(orig_img, mask_resized):
    H,W = orig_img.shape[:2]
    mask = cv2.resize(mask_resized, (W,H), interpolation=cv2.INTER_NEAREST)
    mask_bool = (mask>0).astype('uint8')
    out = orig_img.copy()
    out[mask_bool==255] = 0
    return out

def main():
    files = sorted(glob(os.path.join(INPUT_DIR, '*.*')))
    print('Found', len(files),'images. Processing to', OUTPUT_DIR)
    for fp in files:
        orig = cv2.imread(fp)
        if orig is None: continue
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        resized = read_and_resize(fp, IMG_SIZE)
        pred = predict_mask(model, resized)
        post = postprocess_mask(pred, threshold=0.4)
        result = remove_pectoral_from_original(orig_rgb, post)
        save_path = os.path.join(OUTPUT_DIR, os.path.basename(fp))
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print('Done. Processed images saved to', OUTPUT_DIR)

if __name__=='__main__':
    main()
