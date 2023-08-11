# Author - Anushka Dhiman

import tensorflow as tf
from densenet_block import dense_block,bn_rl_conv, transition_layer

def DenseNet121(input_shape,num_classes,filters=32):
    """
        Build the DenseNet121 model 

        Input:
            input_shape - tuple, shape of input tensor (height, width, channel)
            num_classes - integers, no. of classes 
            filters - no. of filter in convolution blocks
            
        Output:
            model - a Keras Model() instance

    """

    input_image = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(input_image)
    x = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x)

    for repetition in [6,12,24,16]:
        d = dense_block(x, repetition, filters)

    x = transition_layer(d,filters)

    x = tf.keras.layers.GlobalAveragePooling2D()(d)
    output = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)

    #Model
    model = tf.keras.models.Model(inputs = input_image, outputs = output)
    # Return the result
    return model

input_shape = (224,224,3)
num_classes = 3

densenet121_model = DenseNet121(input_shape,num_classes,filters=32)
print(densenet121_model.summary())
