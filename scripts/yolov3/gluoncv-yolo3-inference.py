import os
from os import walk
import sys, logging

import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv

logger = logging.getLogger('')

def load_sym_model(sym_f, param_f, model_dir) :

    try:
        if os.environ['USE_EIA'] == "1":
            device = [mx.eia()] #use EIA for low cost GPU acceleration
        elif os.environ['USE_GPU'] == "1":
            device = [mx.gpu()]
        else :
            #use cpu with MK DNN acceleration
            device = [mx.cpu()]
    except:
        device = [mx.cpu()]

    sym_file = os.path.join(model_dir, sym_f)
    param_file = os.path.join(model_dir, param_f)
    
    return gluon.nn.SymbolBlock.imports(sym_file, ['data'], param_file, ctx=device)
       
def model_fn(model_dir):

    model = load_sym_model(os.environ["SYM_FILE_NAME"], os.environ["PARAM_FILE_NAME"], model_dir)
    return model

def input_fn(request_body, request_content_type):

    input_object, image = gcv.data.transforms.presets.yolo.transform_test(mx.img.imdecode(request_body), int(os.environ['IMG_WIDTH']))
    return input_object

def predict_fn(input_object, model):

    try:
        if os.environ['USE_EIA'] == "1":
            #use EIA for low cost GPU acceleration
            input_object = input_object.copyto(mx.eia())
        elif os.environ['USE_GPU'] == "1":
            input_object = input_object.copyto(mx.gpu())
    except:
        logger.error("Failed to load data into EIA")   

    cid, score, bbox = model(input_object)  
    
    c= cid[0].asnumpy().reshape(cid[0].shape[0]*cid[0].shape[1])
    s=score[0].asnumpy().reshape(score[0].shape[0]*score[0].shape[1])
    bb= bbox[0].asnumpy().reshape(bbox[0].shape[0]*bbox[0].shape[1])
    
    return np.concatenate((c,s,bb))
    
def output_fn(prediction, content_type):
    
    response_body = prediction.tobytes()
    return response_body