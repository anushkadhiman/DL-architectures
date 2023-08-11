import tensorflow as tf

#batch norm + relu + conv
def bn_rl_conv(x,filters,kernel=1,strides=1):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel, strides=strides,padding = 'same')(x)

    return x

def dense_block(x, repetition, filters):

    for _ in range(repetition):
        y = bn_rl_conv(x, 4*filters)
        y = bn_rl_conv(y, filters, 3)
        x = tf.keras.layers.concatenate([y,x])
    return x

def transition_layer(x, filters):
    print("x.shape[-1] //2: ",x.shape[-1] //2)
    x = bn_rl_conv(x,filters)
    x = tf.keras.layers.AvgPool2D(2, strides = 2, padding = 'same')(x)
    return x

