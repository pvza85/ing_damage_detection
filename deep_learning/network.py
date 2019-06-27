import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

def inception(input_layer, num_class):

    conv1_7_7 = tflearn.conv_2d(input_layer, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
    pool1_3_3 = tflearn.max_pool_2d(conv1_7_7, 3, strides=2)
    pool1_3_3 = tflearn.local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = tflearn.conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
    conv2_3_3 = tflearn.conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
    conv2_3_3 = tflearn.local_response_normalization(conv2_3_3)
    pool2_3_3 = tflearn.max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
    inception_3a_1_1 = tflearn.conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = tflearn.conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = tflearn.conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu', name='inception_3a_3_3')
    inception_3a_5_5_reduce = tflearn.conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
    inception_3a_5_5 = tflearn.conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
    inception_3a_pool = tflearn.max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = tflearn.conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                    name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = tflearn.merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                mode='concat', axis=3)

    inception_3b_1_1 = tflearn.conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
    inception_3b_3_3_reduce = tflearn.conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                      name='inception_3b_3_3_reduce')
    inception_3b_3_3 = tflearn.conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
    inception_3b_5_5_reduce = tflearn.conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                      name='inception_3b_5_5_reduce')
    inception_3b_5_5 = tflearn.conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
    inception_3b_pool = tflearn.max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
    inception_3b_pool_1_1 = tflearn.conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                    name='inception_3b_pool_1_1')

    # merge the inception_3b_*
    inception_3b_output = tflearn.merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                mode='concat', axis=3, name='inception_3b_output')

    pool3_3_3 = tflearn.max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = tflearn.conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = tflearn.conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = tflearn.conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = tflearn.conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = tflearn.conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu', name='inception_4a_5_5')
    inception_4a_pool = tflearn.max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
    inception_4a_pool_1_1 = tflearn.conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4a_pool_1_1')

    inception_4a_output = tflearn.merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=3, name='inception_4a_output')

    inception_4b_1_1 = tflearn.conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = tflearn.conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                      name='inception_4b_3_3_reduce')
    inception_4b_3_3 = tflearn.conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = tflearn.conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                      name='inception_4b_5_5_reduce')
    inception_4b_5_5 = tflearn.conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4b_5_5')

    inception_4b_pool = tflearn.max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
    inception_4b_pool_1_1 = tflearn.conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4b_pool_1_1')

    inception_4b_output = tflearn.merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                mode='concat', axis=3, name='inception_4b_output')

    inception_4c_1_1 = tflearn.conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
    inception_4c_3_3_reduce = tflearn.conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                      name='inception_4c_3_3_reduce')
    inception_4c_3_3 = tflearn.conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = tflearn.conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                      name='inception_4c_5_5_reduce')
    inception_4c_5_5 = tflearn.conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4c_5_5')

    inception_4c_pool = tflearn.max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = tflearn.conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4c_pool_1_1')

    inception_4c_output = tflearn.merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                mode='concat', axis=3, name='inception_4c_output')

    inception_4d_1_1 = tflearn.conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = tflearn.conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                      name='inception_4d_3_3_reduce')
    inception_4d_3_3 = tflearn.conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = tflearn.conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                      name='inception_4d_5_5_reduce')
    inception_4d_5_5 = tflearn.conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu', name='inception_4d_5_5')
    inception_4d_pool = tflearn.max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
    inception_4d_pool_1_1 = tflearn.conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                    name='inception_4d_pool_1_1')

    inception_4d_output = tflearn.merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                mode='concat', axis=3, name='inception_4d_output')

    inception_4e_1_1 = tflearn.conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = tflearn.conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                      name='inception_4e_3_3_reduce')
    inception_4e_3_3 = tflearn.conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = tflearn.conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                      name='inception_4e_5_5_reduce')
    inception_4e_5_5 = tflearn.conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = tflearn.max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
    inception_4e_pool_1_1 = tflearn.conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                    name='inception_4e_pool_1_1')

    inception_4e_output = tflearn.merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3,
                                mode='concat')
    #inception_4e_output = tflearn.dropout(inception_4e_output_, 0.8)
    pool4_3_3 = tflearn.max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')
    inception_5a_1_1 = tflearn.conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = tflearn.conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = tflearn.conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = tflearn.conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = tflearn.conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5a_5_5')
    inception_5a_pool = tflearn.max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
    inception_5a_pool_1_1 = tflearn.conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                    name='inception_5a_pool_1_1')

    inception_5a_output_ = tflearn.merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,
                                mode='concat')
    inception_5a_output = tflearn.dropout(inception_5a_output_, 0.5)
    inception_5b_1_1 = tflearn.conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = tflearn.conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                      name='inception_5b_3_3_reduce')
    inception_5b_3_3 = tflearn.conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = tflearn.conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                      name='inception_5b_5_5_reduce')
    inception_5b_5_5 = tflearn.conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
    inception_5b_pool = tflearn.max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
    inception_5b_pool_1_1 = tflearn.conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                    name='inception_5b_pool_1_1')
    inception_5b_output = tflearn.merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3,
                                mode='concat')

    pool5_7_7 = tflearn.avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = tflearn.dropout(pool5_7_7, 0.5)
    loss = tflearn.fully_connected(pool5_7_7, num_class, activation='softmax')

    return loss

def VGGNet(input_layer, num_class):


    x = tflearn.conv_2d(input_layer, 64, 3, activation='relu', name='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc6')
    x = tflearn.dropout(x, 0.4, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc7')
    x = tflearn.dropout(x, 0.4, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', name='fc8', restore=False)
    return x
def ResNet(input_layer, num_class):
    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5
    
    net = tflearn.conv_2d(input_layer, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    net = tflearn.fully_connected(net, num_class, activation='softmax')
    return net

def NiN(input_layer, num_class):
    network = conv_2d(input_layer, 192, 5, activation='relu')
    network = conv_2d(network, 160, 1, activation='relu')
    network = conv_2d(network, 96, 1, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)
    network = conv_2d(network, 192, 5, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = avg_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 10, 1, activation='relu')
    network = avg_pool_2d(network, 8)
    network = flatten(network)
    return network