import tensorflow as tf


def get_channel_axis(data_format):
    if data_format == 'channels_last':
        return -1
    else:
        return 1


def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns:
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor,
                           divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def conv_block(x,
               num_filters,
               kernel_size,
               strides,
               padding,
               dilation=1,
               groups=1,
               use_bias=False,
               use_bn=True,
               bn_eps=1E-5,
               activation='relu',
               data_format='channels_first',
               base_name='ConvBlock'
               ):
    y = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation,
        groups=groups,
        use_bias=use_bias,
        data_format=data_format,
        name='{}/conv_preact'.format(base_name)
    )(x)

    if use_bn:
        y = tf.keras.layers.BatchNormalization(
            axis=get_channel_axis(data_format),
            epsilon=bn_eps,
            name='{}/conv_bn'.format(base_name)
        )(y)

    if activation is not None:
        y = tf.keras.layers.Activation(activation=activation,
                                       name='{}/conv/conv_postact'.format(
                                           base_name))(y)

    return y


def conv1x1_block(
        x,
        num_filters,
        strides=1,
        groups=1,
        use_bias=False,
        use_bn=True,
        bn_eps=1E-5,
        activation='relu',
        data_format='channels_first',
        base_name='Conv1x1Block'
):
    y = conv_block(
        x,
        num_filters=num_filters,
        strides=strides,
        kernel_size=1,
        padding='valid',
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        base_name=base_name
    )
    return y


def conv3x3_block(
        x,
        num_filters,
        strides=1,
        padding='same',
        dilation=1,
        groups=1,
        use_bias=False,
        use_bn=True,
        bn_eps=1E-5,
        activation='relu',
        data_format='channels_first',
        base_name='Conv3x3Block'
):
    y = conv_block(
        x,
        num_filters=num_filters,
        strides=strides,
        kernel_size=3,
        dilation=dilation,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        base_name=base_name
    )
    return y


def conv7x7_block(
        x,
        num_filters,
        strides=1,
        padding='same',
        use_bias=False,
        use_bn=True,
        bn_eps=1E-5,
        activation='relu',
        data_format='channels_first',
        base_name='Conv7x7Block'
):
    y = conv_block(
        x,
        num_filters=num_filters,
        strides=strides,
        kernel_size=7,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        data_format=data_format,
        base_name=base_name
    )
    return y
