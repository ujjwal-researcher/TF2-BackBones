import tensorflow as tf

from utils import conv1x1_block, conv3x3_block, conv7x7_block


def resblock(
        x,
        num_filters,
        strides,
        use_bias=False,
        use_bn=True,
        data_format='channels_first',
        base_name='ResnetBlock'
):
    y = conv3x3_block(
        x,
        num_filters=num_filters,
        strides=strides,
        use_bias=use_bias,
        use_bn=use_bn,
        data_format=data_format,
        base_name='{}/unit1'.format(base_name)
    )

    y = conv3x3_block(
        y,
        num_filters=num_filters,
        use_bias=use_bias,
        use_bn=use_bn,
        activation=None,
        data_format=data_format,
        base_name='{}/unit2'.format(base_name)
    )
    return y


def resbottleneck(
        x,
        num_filters,
        strides,
        padding='same',
        dilation=1,
        conv1_stride=False,
        bottleneck_factor=4,
        data_format='channels_first',
        base_name='ResnetBottleneck'
):
    mid_filters = num_filters // bottleneck_factor
    y = conv1x1_block(x,
                      num_filters=mid_filters,
                      strides=(strides if conv1_stride else 1),
                      data_format=data_format,
                      base_name='{}/unit1'.format(base_name)
                      )

    y = conv3x3_block(
        y,
        num_filters=mid_filters,
        strides=(1 if conv1_stride else strides),
        padding=padding,
        dilation=dilation,
        data_format=data_format,
        base_name='{}/unit2'.format(base_name)
    )

    y = conv1x1_block(y,
                      num_filters=num_filters,
                      activation=None,
                      data_format=data_format,
                      base_name='{}/unit3'.format(base_name)
                      )
    return y


def resunit(x,
            input_channels,
            num_filters,
            strides,
            padding='same',
            dilation=1,
            use_bias=False,
            use_bn=True,
            bottleneck=True,
            conv1_stride=False,
            data_format='channels_first',
            base_name='ResUnit'
            ):
    resize_identity = (input_channels != num_filters) or (strides != 1)
    if resize_identity:
        y = conv1x1_block(
            x,
            num_filters=num_filters,
            strides=strides,
            use_bias=use_bias,
            use_bn=use_bn,
            activation=None,
            data_format=data_format,
            base_name='{}/identity_conv'.format(base_name)
        )
    else:
        y = x

    if bottleneck:
        z = resbottleneck(
            x,
            num_filters=num_filters,
            strides=strides,
            padding=padding,
            dilation=dilation,
            conv1_stride=conv1_stride,
            data_format=data_format,
            base_name='{}/body'.format(base_name)
        )
    else:
        z = resblock(
            x,
            num_filters=num_filters,
            strides=strides,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            base_name='{}/body'.format(base_name)
        )

    y = y + z
    y = tf.keras.layers.ReLU(name='{}/post_act'.format(base_name))(y)

    return y


def resnetinitblock(x,
                    num_filters,
                    data_format='channels_first',
                    base_name='ResnetInitBlock'
                    ):
    y = conv7x7_block(
        x,
        num_filters=num_filters,
        strides=2,
        data_format=data_format,
        base_name='{}/unit1'.format(base_name)
    )

    y = tf.keras.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same',
        data_format=data_format,
        name='{}/pool'.format(base_name)
    )(y)

    return y


def resnet(
        input_height,
        input_width,
        channels,
        init_block_channels,
        bottleneck,
        conv1_stride,
        num_classes=1000,
        data_format='channels_first',
        base_name='Resnet'
):
    if data_format == 'channels_first':
        input_shape = (3, input_height, input_width)
    else:
        input_shape = (input_height, input_width, 3)

    x = tf.keras.Input(shape=input_shape, batch_size=None)
    y = x
    y = resnetinitblock(
        y,
        num_filters=init_block_channels,
        data_format=data_format,
        base_name='{}/init_block'.format(base_name)
    )

    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            y = resunit(
                y,
                input_channels=in_channels,
                num_filters=out_channels,
                strides=strides,
                bottleneck=bottleneck,
                conv1_stride=conv1_stride,
                data_format=data_format,
                base_name='{}/block{}'.format(base_name, i)
            )
            in_channels = out_channels
    y = tf.keras.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        data_format=data_format,
        name='{}/final_pool'.format(base_name)
    )(y)

    if num_classes > 0:
        y = tf.keras.layers.Flatten(data_format=data_format, name='flatten')(y)
        y = tf.keras.layers.Dense(
            units=num_classes,
            name='logits'.format(base_name)
        )(y)

    model = tf.keras.Model(
        inputs=x,
        outputs=y,
        name=base_name
    )
    return model


def get_resnet(
        blocks,
        input_height,
        input_width,
        bottleneck=None,
        conv1_stride=True,
        width_scale=1.0,
        num_classes=1000,
        data_format='channels_first',
        model_name=None
):
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError(
            "Unsupported ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in
                               channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (
                j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in
                    enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    model = resnet(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        num_classes=num_classes,
        data_format=data_format,
        base_name=model_name
    )

    return model


def resnet10(input_height,
             input_width,
             data_format='channels_first',
             num_classes=1000):
    return get_resnet(input_height=input_height, input_width=input_width,
                      blocks=10,
                      num_classes=num_classes,
                      data_format=data_format,
                      model_name='resnet10')


def resnet12(input_height,
             input_width,
             data_format='channels_first',
             num_classes=1000):
    return get_resnet(input_height=input_height, input_width=input_width,
                      blocks=12,
                      num_classes=num_classes,
                      data_format=data_format,
                      model_name='resnet12')


def resnet14(input_height,
             input_width,
             data_format='channels_first',
             num_classes=1000):
    return get_resnet(input_height=input_height, input_width=input_width,
                      blocks=14,
                      num_classes=num_classes,
                      data_format=data_format,
                      model_name='resnet14')


def resnetbc14b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=14,
        bottleneck=True,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnetbc14b'
    )


def resnet16(input_height,
             input_width,
             data_format='channels_first',
             num_classes=1000):
    return get_resnet(input_height=input_height, input_width=input_width,
                      blocks=16,
                      num_classes=num_classes,
                      data_format=data_format,
                      model_name='resnet16')


def resnet18_wd4(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=18,
        data_format=data_format,
        num_classes=num_classes,
        width_scale=0.25,
        model_name='resnet18_wd4'
    )


def resnet18_wd2(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=18,
        data_format=data_format,
        num_classes=num_classes,
        width_scale=0.5,
        model_name='resnet18_wd2'
    )


def resnet18_w3d4(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=18,
        data_format=data_format,
        num_classes=num_classes,
        width_scale=0.75,
        model_name='resnet18_w3d4'
    )


def resnet18(input_height,
             input_width,
             data_format='channels_first',
             num_classes=1000):
    return get_resnet(input_height=input_height, input_width=input_width,
                      blocks=18,
                      num_classes=num_classes,
                      data_format=data_format,
                      model_name='resnet18')


def resnet26(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=26,
        bottleneck=False,
        num_classes=num_classes,
        data_format=data_format,
        model_name='resnet26'
    )


def resnetbc26b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=26,
        bottleneck=True,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnetbc26b'
    )


def resnet34(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=34,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet34'
    )


def resnetbc38b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=38,
        bottleneck=True,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnetbc38b'
    )


def resnet50(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=50,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet50'
    )


def resnet50b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=50,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet50b'
    )


def resnet101(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=101,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet101'
    )


def resnet101b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=101,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet101b'
    )


def resnet152(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=152,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet152'
    )


def resnet152b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=152,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet152b'
    )


def resnet200(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=200,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet200'
    )


def resnet200b(
        input_height,
        input_width,
        data_format='channels_first',
        num_classes=1000
):
    return get_resnet(
        input_height=input_height,
        input_width=input_width,
        blocks=200,
        conv1_stride=False,
        data_format=data_format,
        num_classes=num_classes,
        model_name='resnet200b'
    )
