import tensorflow as tf

def conv_bn_relu(x,f,k,s,padding,i_block):
    """
    conv block contains 
    tf.keras.layers.Conv2D of nf num_filters, k kernel and s stride
    batch normalization 
    relu activation
    """

    # print("x in conv_bn_relu: ",x,f,k,s,i_block)

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
            num_filters - a list on integers, each of them defining the number of filters in each convolutional layer
            stage - integer, one of the 5 stages that our networks is conceptually divided into 
                - stage names have the form: conv2_x, conv3_x ... conv5_x
            block_param - dictionary 
            
        Output:
            X - tensor (m, height, width, channel)

    """

    initializer = tf.keras.initializers.GlorotNormal()
    # block = len(block_param['f'][stage-1])
    # layers will be called conv{stage}_iden{block}_{convlayer_number_within_block}'
    # conv_name = f'conv{stage}_{idn_blocks_num}' + '_{layer}_{type}'

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
                - stage names have the form: conv2_x, conv3_x ... conv5_x
            block_param - 
                    a list on integers, each of them defining the number of filters in each convolutional layer
            
        Output:
            X - tensor (m, height, width, channel)

    """

    # layers will be called conv{stage}_iden{block}_{convlayer_number_within_block}'
    # conv_name = f'conv{stage}_{conv_blocks_num}' + '_{layer}_{type}'
    # block = len(block_param['f'][stage-1])

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

