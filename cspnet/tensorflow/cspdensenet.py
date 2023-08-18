#Author : Anushka Dhiman

import tensorflow as tf

from cspnet_blocks import dense_block, dense_block_b, transition_block, csp_transition_block, Conv_BN

def CSPDenseNet(input_shape,num_classes=80,backbone='dense121', bottleneck=False, filters=32, theta=1.0, modeltype='b'):
    """
        Build the CSPDenseNet model 

        Input:
            input_shape - tuple, shape of input tensor (height, width, channel)
            num_classes - integers, no. of classes 
            backbone - dictionary, type of densenet model used in cspdensnet block i.e. densenet121, densenet169, densenet201, 
            bottleneck - boolean, whether to use DenseNetBC with bottleneck structure
            filters - integers, no. of filters use in dense block
            theta - float, theta is referred to as the compression factor
            modeltype - string, type of cspnet variants i.e. CSPDenseNet, CSPNet (Fusion First), CSPNet (Fusion Last) 

        Output:
            model - a Keras Model() instance

    """

    if modeltype=='b':   # csp-densent
        transition_dense = True
        transition_fuse = True
    if modeltype=='c':    # fusion first
        transition_dense = False
        transition_fuse = True
    if modeltype=='d':    # fusion last
        transition_dense = True
        transition_fuse = False


    input_image = tf.keras.layers.Input(shape = (224, 224, 3))
    
    x = Conv_BN(input_image, 64, 7, strides=2, activation='relu')
    x = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2))(x)

    #CSPNet Block
    num_blocks = densenet_models[backbone]
    for i in range(len(num_blocks)):

        # partial
        in_filters = x.shape[-1]
        skip = tf.keras.layers.Lambda(lambda x: x[..., :in_filters//2])(x)
        x = tf.keras.layers.Lambda(lambda x: x[..., in_filters//2:])(x)

        # dense block
        for j in range(num_blocks[i]):
            # dense block
            if bottleneck:
                x = dense_block_b(x, filters)      # [N,h,w,nK]
            else:
                x = dense_block(x, filters)      # [N,h,w,nK]

        # transition
        if transition_dense:
            # transition block
            x = csp_transition_block(x, theta)

        # fusion
        x = tf.keras.layers.concatenate([skip, x], axis=-1)
        if transition_fuse and i != len(num_blocks)-1:         # last block but not last layer
            x = transition_block(x, theta)


    # for last dense block
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    # head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # model
    model = tf.keras.models.Model(input_image, x)

    return model

# densenet models
densenet_models = {'dense121': [6,12,24,16],
            'dense169': [6,12,32,32],
            'dense201': [6,12,48,32],
            'dense264': [6,12,64,48]}


input_shape = (32,32,3)
num_classes = 10
backbone='dense121'

cspdensenet121_model = CSPDenseNet(input_shape,num_classes,backbone)
print(cspdensenet121_model.summary())
