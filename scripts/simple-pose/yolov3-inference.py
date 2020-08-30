import os
import json
import sys, logging

import numpy as np
import mxnet as mx
from mxnet import gluon

from gluoncv import model_zoo, data

logger = logging.getLogger('')
           
def model_fn(model_dir):
    
    try:
        if os.environ['USE_EIA'] == "1":
            device = mx.eia() #use EIA for low cost GPU acceleration
        elif os.environ['USE_GPU'] == "1":
            device = mx.gpu()
        else :
           #use cpu with MK DNN acceleration
            device = mx.cpu()
    except:
        logger.error("Failed to set desired device context.")
        device = mx.cpu()
    
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, ctx=device)
    detector.reset_class(["person"], reuse_weights=['person'])
    detector.hybridize()
    
    return detector

def input_fn(request_body, content_type):
    
    if content_type == "image/jpeg":
        
        x,img = data.transforms.presets.yolo.transform_test(mx.img.imdecode(request_body),
                                                            int(os.environ['IMG_WIDTH']))
        return (x,img)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(content_type))

    return request_body

def predict_fn(input_object, model):
    
    inp, img = input_object
    
    try:
        if os.environ['USE_EIA'] == "1":
            #use EIA for low cost GPU acceleration
            device = mx.eia()
            x = inp.copyto(device)

        elif os.environ['USE_GPU'] == "1":
            device = mx.gpu()
            x = inp.copyto(device)
        else :
            device = mx.cpu()
            x = inp
    except:
        logger.error("Failed to load data into desired context") 
        device = mx.cpu()
        x = inp
        
    cid, scores, bbox = model(x)

    return {"cid":cid.asnumpy().tolist(), 
            "scores":scores.asnumpy().tolist(), 
            "bbox":bbox.asnumpy().tolist(),
            "img":img.tolist()}
    
def output_fn(prediction, accept):
    
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
        