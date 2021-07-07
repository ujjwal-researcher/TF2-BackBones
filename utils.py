def get_channel_axis(data_format):
    if data_format == 'channels_last':
        return -1
    else:
        return 1