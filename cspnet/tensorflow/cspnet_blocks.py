import tensorflow as tf

# 3x3 conv, BN-tf.keras.layers.ReLU-conv-concat
def dense_block(x, K):
    inpt = x
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(K, 3, strides=1, padding='same')(x)
    x = tf.keras.layers.concatenate([inpt, x])
    return x


# 1x1 conv + 3x3 conv, BN-tf.keras.layers.ReLU-conv-concat
def dense_block_b(x, K):
    inpt = x
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(4*K, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(K, 3, strides=1, padding='same')(x)
    x = tf.keras.layers.concatenate([inpt, x])
    return x


# 1x1 conv + 2x2 avg pooling
def transition_block(x, theta):
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(int(theta*x.shape[-1]), 1, strides=1, padding='same')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
    return x

# 1x1 conv
def csp_transition_block(x, theta):
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(int(theta*x.shape[-1]), 1, strides=1, padding='same')(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = tf.keras.layers.Conv2D(n_filters, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    if activation:
        x = tf.keras.layers.ReLU()(x)
    return x

    