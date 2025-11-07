"""05_explainability_eval.py
Grad-CAM and SHAP examples for image and tabular explanations. Use with trained model file 'models/stage_classifier.h5'.
"""
import tensorflow as tf, numpy as np, cv2, yaml, os
from tensorflow.keras.models import load_model

cfg = yaml.safe_load(open('configs/config.yaml'))
IMG_SIZE = cfg.get('img_size', 256)

def grad_cam(model, img_array, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name; break
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    return heatmap

if __name__=='__main__':
    print('Explainability utilities ready. Use in notebooks to create overlays.') 
