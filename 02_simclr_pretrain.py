"""02_simclr_pretrain.py
Self-supervised pretraining (SimCLR) for encoder on unlabeled mammograms.
This script is a simplified SimCLR implementation for demonstration.
"""
import tensorflow as tf, yaml, os, numpy as np
from glob import glob
from tensorflow.keras import layers, models, optimizers

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = cfg.get('img_size', 256)
BATCH = cfg.get('simclr_batch', 64)
EPOCHS = cfg.get('epochs_simclr', 10)

def augmenter():
    return tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.07),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomContrast(0.1)
    ])

def parse_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)/255.0
    return img

def two_view_dataset(file_paths, batch=BATCH):
    paths = tf.constant(file_paths)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(lambda x: parse_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    aug = augmenter()
    def two_views(x):
        return aug(x), aug(x)
    ds = ds.map(two_views, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def get_encoder():
    base = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    inp = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
    x = base(inp, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inp, x, name='encoder')

def projection_head(h_dim=128):
    return tf.keras.Sequential([layers.Dense(512, activation='relu'), layers.Dense(h_dim)])

# NT-Xent simplified (use with care)
def nt_xent(z1, z2, temperature=0.1):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    large = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(large, large, transpose_b=True)/temperature
    N = tf.shape(z1)[0]
    mask = tf.eye(2*N)
    exp_sim = tf.exp(sim) * (1-mask)
    denom = tf.reduce_sum(exp_sim, axis=1)
    positives = tf.concat([tf.reduce_sum(z1*z2, axis=1), tf.reduce_sum(z2*z1, axis=1)], axis=0)
    loss = -tf.reduce_mean(tf.math.log(tf.exp(positives/temperature)/denom))
    return loss

def train(file_list):
    ds = two_view_dataset(file_list)
    encoder = get_encoder()
    proj = projection_head()
    opt = optimizers.Adam(1e-4)
    for epoch in range(EPOCHS):
        for step, (x1, x2) in enumerate(ds):
            with tf.GradientTape() as tape:
                h1 = encoder(x1, training=True)
                h2 = encoder(x2, training=True)
                z1 = proj(h1, training=True)
                z2 = proj(h2, training=True)
                loss = nt_xent(z1, z2)
            vars = encoder.trainable_variables + proj.trainable_variables
            grads = tape.gradient(loss, vars)
            opt.apply_gradients(zip(grads, vars))
        print(f"Epoch {epoch+1}/{EPOCHS} loss={loss.numpy():.4f}")
    encoder.save_weights('models/encoder_simclr.h5')
    print('Saved encoder weights to models/encoder_simclr.h5')

if __name__=='__main__':
    import glob
    cfg = yaml.safe_load(open('configs/config.yaml'))
    folder = cfg.get('processed_dir', '/content/processed_mammograms')
    files = glob.glob(os.path.join(folder, '*.*'))[:1000]
    print('Using', len(files),'files for SimCLR pretraining (you can change config).')
    train(files)
