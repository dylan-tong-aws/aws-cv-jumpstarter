import os
from os import walk
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv

def load_sym_model(sym_f, param_f, model_dir) :

#    try:
#        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
#        device = [mx.gpu(0)]
#        print('GPU device is available')
#    except:
    if os.environ['USE_EIA'] == "1":
        device = [mx.eia()] #use EIA for low cost GPU acceleration
    else #use cpu with MK DNN acceleration
        device = [mx.cpu()]
#        print('Using CPU on local machine. GPU device was not detected')

    sym_file = os.path.join(model_dir, sym_f)
    param_file = os.path.join(model_dir, param_f)
    return gluon.nn.SymbolBlock.imports(sym_file, ['data'], param_file, ctx=device)
       
def model_fn(model_dir):
    
    model = load_sym_model(os.environ["SYM_FILE_NAME"], os.environ["PARAM_FILE_NAME"], model_dir)
    return model

def input_fn(request_body, request_content_type):
    
    input_object, image = gcv.data.transforms.presets.yolo.transform_test(mx.img.imdecode(request_body), 512)
    return input_object

def predict_fn(input_object, model):

    if os.environ['USE_EIA']  == "1":
        input_object = input_object.copyto(mx.eia())
        
    cid, score, bbox = model(input_object)  
    
    c= cid[0].asnumpy().reshape(cid[0].shape[0]*cid[0].shape[1])
    s=score[0].asnumpy().reshape(score[0].shape[0]*score[0].shape[1])
    bb= bbox[0].asnumpy().reshape(bbox[0].shape[0]*bbox[0].shape[1])
    
    return np.concatenate((c,s,bb))
    
def output_fn(prediction, content_type):
    
    response_body = prediction.tobytes()
    
    return response_body