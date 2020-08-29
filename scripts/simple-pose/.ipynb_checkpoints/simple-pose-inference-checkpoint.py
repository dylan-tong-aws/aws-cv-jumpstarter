import os
from os import walk
import sys, logging

import numpy as np
import mxnet as mx
from mxnet import gluon

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord


logger = logging.getLogger('')

    
def load_imperative_model(model_dir) :
    
    try:
        if os.environ['USE_EIA'] == "1":
            device = [mx.eia()] #use EIA for low cost GPU acceleration
        else :
            #use cpu with MK DNN acceleration
            device = [mx.cpu()]
    except:
        device = [mx.cpu()]
        
    kwargs = {'ctx': context, 
              'num_joints': 17,
              'pretrained': False,
              'pretrained_base': False,
              'pretrained_ctx': device}
    
    base, w = get_model_info(model_dir)
    net = model_zoo.get_model('simple_pose_resnet18_v1b', **kwargs)
    net.load_parameters(os.path.join(model_dir,w))
    
    return net
    

def load_sym_model(sym_f, param_f, model_dir) :

    try:
        if os.environ['USE_EIA'] == "1":
            device = [mx.eia()] #use EIA for low cost GPU acceleration
        else :
            #use cpu with MK DNN acceleration
            device = [mx.cpu()]
    except:
        device = [mx.cpu()]

    sym_file = os.path.join(model_dir, sym_f)
    param_file = os.path.join(model_dir, param_f)
    
    return gluon.nn.SymbolBlock.imports(sym_file, ['data'], param_file, ctx=device)
       
def model_fn(model_dir):

    pose_model = load_imperative_model(model_dir)
    #pose_model = load_sym_model(os.environ["SYM_FILE_NAME"], os.environ["PARAM_FILE_NAME"], model_dir)
    
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    detector.reset_class(["person"], reuse_weights=['person'])
    
    return (pose_model,detector)

def input_fn(request_body, request_content_type):

    input_object, img = data.transforms.presets.ssd.load_test(mx.img.imdecode(request_body), 512)
    return input_object

def predict_fn(input_object, model):

     try:
        if os.environ['USE_EIA'] == "1":
            #use EIA for low cost GPU acceleration
            input_object = input_object.copyto(mx.eia())
    except:
        logger.error("Failed to load data into EIA")     
    
    
    
    c= cid[0].asnumpy().reshape(cid[0].shape[0]*cid[0].shape[1])
    s=score[0].asnumpy().reshape(score[0].shape[0]*score[0].shape[1])
    bb= bbox[0].asnumpy().reshape(bbox[0].shape[0]*bbox[0].shape[1])
    
    return np.concatenate((c,s,bb))
    
def output_fn(prediction, content_type):
    
    response_body = prediction.tobytes()
    return response_body