import tensorflow as tf

import utils


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
        base_name=None
):
    y = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
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
            axis=utils.get_channel_axis(data_format=data_format),
            epsilon=bn_eps,
            name='{}/bn'.format(base_name)
        )(y)

    y = tf.keras.layers.Activation(activation=activation,
                                   name='{}/conv_postact'.format(base_name))(
        y)

    return y


def vgg_dense(
        x,
        num_units,
        name
):
    y = tf.keras.layers.Dense(
        units=num_units,
        name='{}_preact'.format(name)
    )(x)
    y = tf.keras.layers.ReLU(name='{}_postact'.format(name))(y)
    y = tf.keras.layers.Dropout(rate=0.5, name='{}_dropout'.format(name))(y)
    return y


def vgg_output(x,
               num_classes
               ):
    y = vgg_dense(x,
                  num_units=4096,
                  name='fc1')

    y = vgg_dense(y,
                  num_units=4096,
                  name='fc2')
    if num_classes > 0:
        y = tf.keras.layers.Dense(
            units=num_classes,
            name='logits'
        )(y)
    return y


def vgg(
        input_height,
        input_width,
        num_filters,
        use_bias=True,
        use_bn=False,
        num_classes=1000,
        data_format='channels_first',
        model_name='vgg'
):
    if data_format == 'channels_first':
        input_shape = (3, input_height, input_width)
    else:
        input_shape = (input_height, input_width, 3)

    y = tf.keras.Input(shape=input_shape, batch_size=None)
    image_input = y
    for i, filters_per_stage in enumerate(num_filters):
        for j, filters in enumerate(filters_per_stage):
            y = conv3x3_block(
                y,
                num_filters=filters,
                use_bias=use_bias,
                use_bn=use_bn,
                data_format=data_format,
                base_name='block{}/unit{}'.format(i + 1, j + 1)
            )

        y = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2,
            padding='valid',
            data_format=data_format,
            name='pool{}'.format(i + 1)
        )(y)

    y = tf.keras.layers.Flatten(data_format=data_format, name='flatten')(y)
    y = vgg_output(y, num_classes=num_classes)

    model = tf.keras.Model(
        inputs=image_input,
        outputs=y,
        name=model_name
    )
    return model


def get_vgg(
        model_name,
        input_height,
        input_width,
        blocks,
        use_bias=True,
        use_bn=False,
        num_classes=1000,
        data_format='channels_first',
):
    if blocks == 11:
        layers = [1, 1, 2, 2, 2]
    elif blocks == 13:
        layers = [2, 2, 2, 2, 2]
    elif blocks == 16:
        layers = [2, 2, 3, 3, 3]
    elif blocks == 19:
        layers = [2, 2, 4, 4, 4]
    else:
        raise ValueError(
            "Unsupported VGG with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    vgg_model = vgg(
        input_height=input_height,
        input_width=input_width,
        num_filters=channels,
        use_bias=use_bias,
        use_bn=use_bn,
        num_classes=num_classes,
        data_format=data_format,
        model_name=model_name
    )

    return vgg_model


def vgg_11(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='vgg_11',
        input_height=input_height,
        input_width=input_width,
        blocks=11,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def vgg_13(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='vgg_13',
        input_height=input_height,
        input_width=input_width,
        blocks=13,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def vgg_16(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='vgg_16',
        input_height=input_height,
        input_width=input_width,
        blocks=16,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def vgg_19(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='vgg_19',
        input_height=input_height,
        input_width=input_width,
        blocks=19,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_11(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_11',
        input_height=input_height,
        input_width=input_width,
        blocks=11,
        use_bn=True,
        use_bias=False,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_13(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_13',
        input_height=input_height,
        input_width=input_width,
        blocks=13,
        use_bn=True,
        use_bias=False,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_16(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_16',
        input_height=input_height,
        input_width=input_width,
        blocks=16,
        use_bn=True,
        use_bias=False,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_19(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_19',
        input_height=input_height,
        input_width=input_width,
        blocks=19,
        use_bn=True,
        use_bias=False,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_11b(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_11b',
        input_height=input_height,
        input_width=input_width,
        blocks=11,
        use_bn=True,
        use_bias=True,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_13b(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_13b',
        input_height=input_height,
        input_width=input_width,
        blocks=13,
        use_bn=True,
        use_bias=True,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_16b(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_16b',
        input_height=input_height,
        input_width=input_width,
        blocks=11,
        use_bn=True,
        use_bias=True,
        data_format=data_format,
        num_classes=num_classes
    )
    return model


def bn_vgg_19b(
        input_height,
        input_width,
        data_format,
        num_classes
):
    model = get_vgg(
        model_name='bn_vgg_19b',
        input_height=input_height,
        input_width=input_width,
        blocks=11,
        use_bn=True,
        use_bias=True,
        data_format=data_format,
        num_classes=num_classes
    )
    return model
