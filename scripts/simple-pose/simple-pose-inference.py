import os
import json
import sys, logging

import numpy as np
import mxnet as mx
from mxnet import gluon

from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

logger = logging.getLogger('')
    
def load_imperative_model(model_dir, device) :
        
    kwargs = {'ctx': device, 
              'num_joints': 17,
              'pretrained': False,
              'pretrained_base': False,
              'pretrained_ctx': device}
    
    base, w = get_model_info(model_dir)
    net = model_zoo.get_model('simple_pose_resnet18_v1b', **kwargs)
    net.load_parameters(os.path.join(model_dir,w))
    
    return net

def load_sym_model(sym_f, param_f, model_dir, device) :

    sym_file = os.path.join(model_dir, sym_f)
    param_file = os.path.join(model_dir, param_f)
    
    return gluon.nn.SymbolBlock.imports(sym_file, ['data'], param_file, ctx=device)
       
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
    
    try:
        if os.environ['MX_MODE'] == "imperative" :
            pose_model = load_imperative_model(model_dir, device)
        elif os.environ['MX_MODE'] == "symbolic" :
            pose_model = load_sym_model(os.environ["SYM_FILE_NAME"], 
                                        os.environ["PARAM_FILE_NAME"], 
                                        model_dir, device)  
        else :
            pose_model = load_sym_model(os.environ["SYM_FILE_NAME"], 
                                        os.environ["PARAM_FILE_NAME"], 
                                        model_dir, device)

#        pose_model = get_model('simple_pose_resnet18_v1b', 
#                               num_joints=17, pretrained=True,
#                               ctx=device, pretrained_ctx=device)

        return pose_model
    except:
        RuntimeException("Failed load: {} {}. Mode: {} Ctx: {}."
                         .format(os.environ["SYM_FILE_NAME"],
                                 os.environ["PARAM_FILE_NAME"],
                                 os.environ['MX_MODE'], device))

def input_fn(request_body, content_type):

    try:
        if content_type == "application/json":

            json_payload = json.loads(request_body)
            img = mx.nd.array(json_payload["img"])
            cid = mx.nd.array(json_payload["cid"])
            scores = mx.nd.array(json_payload["scores"])
            bbox = mx.nd.array(json_payload["bbox"])

            return (img, cid, scores, bbox)
        else:
            raise RuntimeException("{} content type is not supported.".format(content_type))
    except:
        raise RuntimeException("Failed to parse request.")
            
def copy_to_device(inp, device):

    img, cid, scores, bbox = inp                
               
    img = img.copyto(device)
    cid = cid.copyto(device)
    scores = scores.copyto(device)
    bbox = bbox.copyto(device)
                          
    return (img, cid, scores, bbox)                      
                         
def predict_fn(input_object, model):

    try:
        if os.environ['USE_EIA'] == "1":
            device = mx.eia() 
            img, cid, scores, bbox = copy_to_device(input_object, device)             
        elif os.environ['USE_GPU'] == "1":
            device = mx.gpu()
            img, cid, scores, bbox = copy_to_device(input_object, device)              
        else :
            device = mx.cpu()
            img, cid, scores, bbox = input_object               
    except:
        device = mx.cpu()
        img, cid, scores, bbox = input_object                                 
        logger.error("Failed to load data into desired context")                   
                          
    pose_input, upscale_bbox = detector_to_simple_pose(img, cid, scores, bbox)
    predicted_heatmap = model(pose_input.as_in_context(device))
    predicted_heatmap = model(pose_input)
    keypoints, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    
    c = cid[0].asnumpy().reshape(cid[0].shape[0]*cid[0].shape[1])
    s = scores[0].asnumpy().reshape(scores[0].shape[0]*scores[0].shape[1])
    bb = bbox[0].asnumpy().reshape(bbox[0].shape[0]*bbox[0].shape[1])
    
    kp = keypoints.asnumpy().reshape(keypoints.shape[0]*keypoints.shape[1]*keypoints.shape[2])
    cfd = confidence.asnumpy().reshape(confidence.shape[0]*confidence.shape[1]*confidence.shape[2])

    return np.concatenate((c,s,bb,kp,cfd))
    
def output_fn(prediction, accept):
    
    if accept == "application/x-npy":
        return prediction.tobytes()
    else:
        raise RuntimeException("{} accept type is not supported.".format(accept))