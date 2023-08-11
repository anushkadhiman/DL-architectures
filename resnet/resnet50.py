# Author: Anushka Dhiman 

import tensorflow as tf

from resnet_block import conv_bn_relu, residual_block, identity_block

def ResNet50(input_shape, num_classes, conv_identity_blocks, conv_block_params, identity_block_params):
    """
        Build the ResNet50 model 

        Input:
            input_shape - tuple, shape of input tensor (height, width, channel)
            num_classes - integers, no. of classes 
            conv_identity_blocks - dictionary, contain keys conv_block which is the list of no. of conv blocks 
                and identity_blocks which is the list of no. of identity blocks
            conv_block_params - dictionary, contain keys 'f' - list of filters at each convolution block at each stages,
                            'k' - kernel size of convolution blocks at each stages and 's' - stride size of convolution blocks at each stages
            identity_block_params - dictionary, contain keys 'f' - list of filters at each identity block at each stages,
                            'k' - kernel size of identity blocks at each stages and 's' - stride size of identity blocks at each stages
                            
        Output:
            X - tensor (m, height, width, channel)

    """

    # tensor placeholder for the model's input
    x_input =  tf.keras.layers.Input(input_shape)
    # print("x_input: ",x_input)
    #stage 1
    # padding
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)

    # convolutional layer, followed by batch normalization and relu activation
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='conv1_1_1_nb')(x)
    x =  tf.keras.layers.Activation('relu')(x)

    # max pooling layer to halve the size coming from the previous layer
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # print("x: ",x)

    # conv_num is the no. of conv blocks used and idn_num is the no. of identity blocks used
    # in ith no. of stage.
    for i in range(1,stage):
        # print('stage i: ',i)

        conv_blocks_num = conv_identity_blocks['conv_blocks'][i-1]
        # print("conv_blocks_num: ",conv_blocks_num)
        x = residual_block(x, i, conv_blocks_num, conv_block_params)
        # print(x)
        
        idn_blocks_num = conv_identity_blocks['identity_blocks'][i-1]
        # print("idn_blocks_num: ",idn_blocks_num)
        x = identity_block(x, i, idn_blocks_num, identity_block_params)
        # print(x)    
          

    # Average Pooling layers
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    # Output layer
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax',
                kernel_initializer=initializer)(x)

    #Model
    model = tf.keras.models.Model(inputs = x_input, outputs = output)
    # Return the result
    return model

##resnet50
input_shape = (32,32,3)
num_classes = 10
stage = 5
conv_identity_blocks = {'conv_blocks' : [1,1,1,1], 'identity_blocks' : [2,3,5,2]}
conv_block_params = {'f': [[64, 64, 256],[128, 128, 512],[256, 256, 1024],[512, 512, 2048]], 'k': [1,3,1,1], 's': [1,2,2,2]}
identity_block_params = {'f': [[64, 64, 256],[128, 128, 512],[256, 256, 1024],[512, 512, 2048]], 'k': [1,3,1,1], 's': [1,2,2,2]}
initializer = tf.keras.initializers.GlorotNormal()

resnet50_model = ResNet50(input_shape,num_classes, conv_identity_blocks, conv_block_params,identity_block_params)

print(resnet50_model.summary())

