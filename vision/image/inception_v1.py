import tensorflow as tf

import utils


def inception_conv(x,
                   num_filters,
                   kernel_size,
                   strides,
                   padding,
                   data_format,
                   base_name
                   ):
    y = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name='{}/conv1'.format(base_name)
    )(x)

    y = tf.keras.layers.BatchNormalization(
        epsilon=1E-3,
        axis=utils.get_bn_axis(data_format),
        name='{}/bn'.format(base_name)
    )(y)

    y = tf.keras.layers.ReLU(name='{}/relu'.format(base_name))(y)

    return y


def conv_seq_branch(x,
                    num_filters,
                    kernel_sizes,
                    strides,
                    paddings,
                    data_format,
                    base_name
                    ):
    for i, (filter_count, kernel_size, stride, padding) in enumerate(zip(
            num_filters, kernel_sizes, strides, paddings)):
        x = inception_conv(x,
                           num_filters=filter_count,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding=padding,
                           data_format=data_format,
                           base_name='{}_{}'.format(base_name, i + 1)
                           )(x)
    return x

def conv_3x3_branch(x,
                    num_filters,
                    kernel_sizes,
                    strides,
                    paddings,
                    data_format,
                    base_name
                    ):
    for i, (filter_count, kernel_size, stride, padding) in enumerate(zip(
            num_filters, kernel_sizes, strides, paddings)):
        x = inception_conv(x,
                           num_filters=filter_count,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding=padding,
                           data_format=data_format,
                           base_name='{}_{}'.format(base_name, i + 1)
                           ) (x)
        out_channels = filter_count

    x = inception_conv(x,
                       num_filters=out_channels,
                       kernel_size=(1,3),
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       base_name='{}'

                       )


def inception_init_block(x,
                         data_format
                         ):
    y = inception_conv(x,
                       num_filters=32,
                       kernel_size=3,
                       strides=2,
                       padding='valid',
                       data_format=data_format,
                       base_name='init_1'
                       )(x)

    y = inception_conv(x,
                       num_filters=32,
                       kernel_size=3,
                       strides=1,
                       padding='valid',
                       data_format=data_format,
                       base_name='init_2'
                       )(y)

    y = inception_conv(x,
                       num_filters=64,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       data_format=data_format,
                       base_name='init_3'
                       )(y)

    y = tf.keras.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='valid',
        data_format=data_format,
        name='pool1'
    )(y)

    y = inception_conv(x,
                       num_filters=80,
                       kernel_size=1,
                       strides=1,
                       padding='valid',
                       data_format=data_format,
                       base_name='init_4'
                       )(y)

    y = inception_conv(x,
                       num_filters=192,
                       kernel_size=3,
                       strides=1,
                       padding='valid',
                       data_format=data_format,
                       base_name='init_5'

                       )(y)

    y = tf.keras.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='valid',
        data_format=data_format,
        name='pool2'
    )(y)

    return y
