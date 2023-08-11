# Author: Anushka Dhiman

import tensorflow as tf

def conv_bn_relu(x,f,k,s,padding,i_block):
    """
    conv block contains 
    tf.keras.layers.Conv2D of nf num_filters, k kernel and s stride
    batch normalization 
    relu activation
    """

    x = tf.keras.layers.Conv2D(f, kernel_size=(k,k), strides=(s,s),\
                kernel_initializer=initializer, padding=padding)(x)

    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x =  tf.keras.layers.Activation("relu")(x)

    return x

def identity_block(x, stage, idn_blocks_num, block_param):
    """
        Identity block 

        Input:
            x - input tensor of shape (m, height, width, channel)
            stage - integer, one of the 5 stages that our networks is conceptually divided into 
            idn_blocks_num - integer, no. of identity blocks in a stage
            block_param - dictionary, contain keys 'f' - list of filters at each identity block at each stages,
                            'k' - kernel size of identity blocks at each stages and 's' - stride size of identity blocks at each stages
            
        Output:
            x - tensor (m, height, width, channel)

    """

    initializer = tf.keras.initializers.GlorotNormal()

    # Create skip connection
    x_skip = x

    #convolution blocks 
    for num_blocks in range(idn_blocks_num):
        # print("idn_blocks_num i: ",num_blocks)
        # print("block_param['f'][stage-1]: ",block_param['f'][stage-1])
        for nf in  block_param['f'][stage-1]:
            # print("nf: ",nf)
            # Perform the convolution layer, batch norm and relu 
            x = conv_bn_relu(x_skip,nf,1,1,"valid",num_blocks)

            x = conv_bn_relu(x,nf,3,1,"same",num_blocks)


        x = tf.keras.layers.Conv2D(block_param['f'][stage-1][-1], kernel_size=(1,1), strides=(1,1),\
            kernel_initializer=initializer, padding="valid")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)

        # Add the skip connection to the main mapping
        x =  tf.keras.layers.Add()([x, x_skip])

        # Nonlinearly function RELU at the end of the block
        x =  tf.keras.layers.Activation("relu")(x)
        # print(x)

    # Return the result
    return x


def residual_block(x, stage, conv_blocks_num, block_param):
    """
        Simple Residual block 

        Input:
            x - input tensor of shape (m, height, width, channel)
            stage - integer, one of the 5 stages that our networks is conceptually divided into 
            conv_blocks_num - integer, no. of convolution blocks in a stage
            block_param - dictionary, contain keys 'f' - list of filters at each convolution block at each stages,
                            'k' - kernel size of convolution blocks at each stages and 's' - stride size of convolution blocks at each stages
            
        Output:
            x - tensor (m, height, width, channel)


    """

    # Create skip connection
    x_skip = x
    # print("x_skip: ",x_skip)
    #no. of conv_bn_relu blocks
    for num_blocks in range(conv_blocks_num):
        # print("conv_blocks_num i: ",num_blocks)
        for nf in  block_param['f'][stage-1]:
            # print("nf: ",nf)
            # print("block_param['s'][stage-1]: ",block_param['s'][stage-1])
            # Perform the convolution layer
            x = conv_bn_relu(x_skip,nf,1,block_param['s'][stage-1],"valid",num_blocks)

            x = conv_bn_relu(x,nf,3,1,"same",num_blocks)

        x = tf.keras.layers.Conv2D(block_param['f'][stage-1][-1], kernel_size=(1,1),strides=(1,1),\
            kernel_initializer=initializer, padding="valid")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # print("x: ",x)

        # shortcut path
        x_skip = tf.keras.layers.Conv2D(block_param['f'][stage-1][-1], kernel_size=(1,1),strides=(block_param['s'][stage-1],block_param['s'][stage-1]),
                            padding='valid', kernel_initializer=initializer)(x_skip)
        x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
        # print("x_skip: ",x_skip)
        # Add the skip connection to the main mapping
        x =  tf.keras.layers.Add()([x, x_skip])

        # Nonlinearly function RELU at the end of the block
        x =  tf.keras.layers.Activation("relu")(x)
        # print(x)

    # Return the result
    return x

