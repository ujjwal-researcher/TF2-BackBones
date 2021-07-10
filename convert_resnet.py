from vision.image import resnet as myresnet
from glob import glob
from loguru import logger
import tensorflow as tf
import resnet_orig
import re
import os
import numpy as np
from time import time
from copy import deepcopy

tf.random.set_seed(time())

models = [
        'resnet10',
        'resnet12',
        'resnet14',
        'resnetbc14b',
        'resnet16',
        'resnet18_wd4',
        'resnet18_wd2',
        'resnet18_w3d4',
        'resnet18',
        'resnet26',
        'resnetbc26b',
        'resnet34',
        'resnetbc38b',
        'resnet50',
        'resnet50b',
        'resnet101',
        'resnet101b',
        'resnet152',
        'resnet152b',
        'resnet200',
        'resnet200b',
    ]


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def find_model_file(model_type):
    model_files = glob('*.h5')
    for m in model_files:
        if '{}-'.format(model_type) in m:
            return m
    return None



def remap_our_model_variables(our_variables, model_name):
    remapped = list()
    reg = re.compile(r'(stage\d+)')
    for var in our_variables:
        newvar = var.replace(model_name, 'features/features')
        stage_search = re.search(reg, newvar)
        if stage_search is not None:
            stage_search = stage_search[0]
            newvar = newvar.replace(stage_search, '{}/{}'.format(stage_search,
                                                         stage_search))
        newvar = newvar.replace('conv_preact', 'conv/conv')
        newvar = newvar.replace('conv_bn','bn')
        newvar = newvar.replace('logits','output1')
        remapped.append(newvar)

    remap_dict = dict([(x,y) for x,y in zip(our_variables, remapped)])
    logger.info(remap_dict)
    return remap_dict

def get_correct_variable(variable_name, trainable_variable_names):
    for i, var in enumerate(trainable_variable_names):
        if variable_name == var:
            return i
    logger.info('Uffff.....')
    return None


layer_regexp_compiled = re.compile(r'(.*)\/.*')
model_files = glob('*.h5')
a = np.ones(shape=(1,224,224,3), dtype=np.float32)
inp = tf.constant(a, dtype=tf.float32)
for model_type in models:
    logger.info('Model is {}.'.format(model_type))
    model = eval('myresnet.{}(input_height=224,input_width=224,'
                 'num_classes=1000,data_format="channels_last")'.format(
        model_type))
    model2 = eval('resnet_orig.{}(data_format="channels_last")'.format(
        model_type))
    model2.build(input_shape=(None,224, 224,3))
    model_name=find_model_file(model_type)
    logger.info('Model file is {}.'.format(model_name))
    original_weights = deepcopy(model2.weights)
    if model_name is not  None:
        e = model2.load_weights(model_name, by_name=True, skip_mismatch=False)
        print(e)
        loaded_weights = deepcopy(model2.weights)
    else:
        logger.info('Pretrained model is not available for {}.'.format(
            model_type))
        continue
    diff = [np.mean(x.numpy()-y.numpy()) for x,y in zip(original_weights,
                                               loaded_weights)]


    our_model_weights = model.weights
    their_model_weights = model2.weights
    assert (len(our_model_weights) == len(their_model_weights))
    our_variable_names = [x.name for x in model.weights]
    their_variable_names = [x.name for x in model2.weights]
    remap_dict = remap_our_model_variables(our_variable_names, model_type)
    new_weights = list()
    for i in range(len(our_model_weights)):
        our_name = model.weights[i].name
        remapped_name = remap_dict[our_name]
        source_index = get_correct_variable(remapped_name, their_variable_names)
        new_weights.append(
            deepcopy(model2.weights[source_index].numpy()))

        logger.debug('Copying from {} ({}) to {} ({}).'.format(
            model2.weights[
                                                        source_index].name,
            model2.weights[source_index].value().shape,
                                                    model.weights[
                                                        i].name,
        model.weights[i].value().shape))

    logger.info(len(new_weights))
    logger.info('Setting new weights')
    model.set_weights(new_weights)
    logger.info('Finished setting new weights.')
    their_output = model2(inp, training=False)
    our_output = model(inp, training=False)
    logger.info(np.max(their_output.numpy() - our_output.numpy()))
    logger.info(diff) # This must be 0.0
    # model.summary()
    # model2.summary()
    break
