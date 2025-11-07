"""03_multimodal_fusion_train.py
Fine-tune encoder (optionally with SimCLR weights) and train multimodal fusion head that combines image features + radiomics + clinical tabular data.
Outputs: models/stage_classifier.h5
"""
import tensorflow as tf, yaml, os, pandas as pd, numpy as np
from tensorflow.keras import layers, models, optimizers
from glob import glob

cfg = yaml.safe_load(open('configs/config.yaml'))
IMG_SIZE = cfg.get('img_size', 256)
BATCH = cfg.get('batch_size', 16)
EPOCHS = cfg.get('epochs_finetune', 20)

# Load CSV prepared using utils/data_prep.py (must contain columns: img_path, stage_label (0-4), rad_* features, clinical features)
CSV_PATH = 'dataset_with_clinical_and_radiomics.csv'

def get_image_encoder():
    base = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    inp = tf.keras.Input((IMG_SIZE,IMG_SIZE,3), name='image_input')
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    return models.Model(inp, x, name='image_encoder')

def build_multimodal_model(num_tab=10, num_classes=5):
    img_enc = get_image_encoder()
    img_in = img_enc.input
    tab_in = layers.Input(shape=(num_tab,), name='tabular_input')
    img_feat = img_enc(img_in)
    t = layers.Dense(64, activation='relu')(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.2)(t)
    concat = layers.Concatenate()([img_feat, t])
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(concat)
    x = layers.Dropout(0.4)(x)
    out_stage = layers.Dense(num_classes, activation='softmax', name='stage_output')(x)
    out_bin = layers.Dense(1, activation='sigmoid', name='binary_output')(x)
    model = models.Model(inputs=[img_in, tab_in], outputs=[out_stage, out_bin])
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss={'stage_output':'categorical_crossentropy','binary_output':'binary_crossentropy'},
                  loss_weights={'stage_output':1.0,'binary_output':0.5},
                  metrics={'stage_output':'accuracy','binary_output':'accuracy'})
    return model

def data_generator(df, batch=BATCH, img_size=IMG_SIZE, tab_cols=None):
    import sklearn.preprocessing as prep
    idx = np.arange(len(df))
    while True:
        np.random.shuffle(idx)
        for i in range(0, len(df), batch):
            batch_idx = idx[i:i+batch]
            imgs = []
            tabs = []
            stages = []
            bins = []
            for j in batch_idx:
                row = df.iloc[j]
                img = tf.keras.preprocessing.image.load_img(row['img_path'], target_size=(img_size,img_size))
                img = tf.keras.preprocessing.image.img_to_array(img)/255.0
                imgs.append(img)
                tabs.append(row[tab_cols].values.astype('float32'))
                s = tf.keras.utils.to_categorical(int(row['stage_label']), num_classes=5)
                stages.append(s)
                bins.append(1 if int(row['stage_label'])>0 else 0)
            yield [np.array(imgs), np.array(tabs)], [np.array(stages), np.array(bins)]

if __name__=='__main__':
    df = pd.read_csv(CSV_PATH)
    tab_cols = [c for c in df.columns if c.startswith('rad_') or c in ['age','density']]
    print('Tabular columns used:', tab_cols)
    model = build_multimodal_model(num_tab=len(tab_cols))
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['stage_label'])
    train_gen = data_generator(train_df, tab_cols=tab_cols)
    val_gen = data_generator(val_df, tab_cols=tab_cols)
    steps_per_epoch = max(1, len(train_df)//BATCH)
    val_steps = max(1, len(val_df)//BATCH)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)]
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, callbacks=callbacks)
    model.save('models/stage_classifier.h5')
    print('Saved model to models/stage_classifier.h5')
