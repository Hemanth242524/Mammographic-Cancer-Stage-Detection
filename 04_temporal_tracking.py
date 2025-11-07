"""04_temporal_tracking.py
Temporal models: Siamese (prior vs current) and sequence (GRU) example.
Requires patient-level paired CSV or sequence CSV prepared by utils/data_prep.py
"""
import tensorflow as tf, yaml, os, pandas as pd, numpy as np
from tensorflow.keras import layers, models, optimizers

cfg = yaml.safe_load(open('configs/config.yaml'))
IMG_SIZE = cfg.get('img_size', 256)
BATCH = cfg.get('batch_size', 16)
EPOCHS = 10

def get_feature_encoder():
    base = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    inp = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inp, x, name='feat_enc')

def siamese_model(feat_dim=1536, num_tab=10, num_classes=5):
    enc = get_feature_encoder()
    cur_in = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3), name='img_current')
    pri_in = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3), name='img_prior')
    feat_cur = enc(cur_in)
    feat_pri = enc(pri_in)
    absdiff = layers.Lambda(lambda x: tf.abs(x[0]-x[1]))([feat_cur, feat_pri])
    tab_cur = layers.Input(shape=(num_tab,), name='tab_current')
    tab_pri = layers.Input(shape=(num_tab,), name='tab_prior')
    combined = layers.Concatenate()([feat_cur, feat_pri, absdiff, tab_cur, tab_pri])
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(0.4)(x)
    out_stage = layers.Dense(num_classes, activation='softmax')(x)
    out_bin = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=[cur_in, pri_in, tab_cur, tab_pri], outputs=[out_stage, out_bin])
    model.compile(optimizer='adam', loss=['categorical_crossentropy','binary_crossentropy'], metrics=['accuracy'])
    return model

def sequence_model(seq_len=4, num_tab=10, num_classes=5):
    enc = get_feature_encoder()
    img_seq = layers.Input(shape=(seq_len, IMG_SIZE, IMG_SIZE, 3), name='img_seq')
    td = layers.TimeDistributed(enc)(img_seq)
    tab_seq = layers.Input(shape=(seq_len, num_tab), name='tab_seq')
    x = layers.Concatenate(axis=-1)([td, tab_seq])
    x = layers.GRU(256, return_sequences=False)(x)
    x = layers.Dense(256, activation='relu')(x)
    out_stage = layers.Dense(num_classes, activation='softmax')(x)
    out_bin = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=[img_seq, tab_seq], outputs=[out_stage, out_bin])
    model.compile(optimizer='adam', loss=['categorical_crossentropy','binary_crossentropy'])
    return model

if __name__=='__main__':
    print('Temporal models defined. Use your patient CSVs to train.')
