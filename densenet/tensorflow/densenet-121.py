import tensorflow as tf
from densenet_block import dense_block,bn_rl_conv, transition_layer

def DenseNet121(input_shape,num_classes,filters=32):
    """
        Build the DenseNet121 model 

        Input:
            x - input tensor of shape (m, height, width, channel)
            stage - integer, one of the 5 stages that our networks is conceptually divided into 
                - stage names have the form: conv2_x, conv3_x ... conv5_x
            conv_identity_blocks - dictionary contains keys conv_block which is the list of no. of conv blocks 
                and identity_blocks which is the list of no. of identity blocks
            
        Output:
            X - tensor (m, height, width, channel)

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