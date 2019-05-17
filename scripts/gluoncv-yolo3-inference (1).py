import os
from os import walk
import json
import time
import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv


def get_model_info(model_dir):

    # Temp hack. I would like to have this as part of configurations.
    CLASSES =  ['Cardinal','Northern_Flicker','American_Goldfinch',
          'Ruby_throated_Hummingbird','Blue_Jay']

    for (dirpath, dirnames, filenames) in walk(model_dir):
        for f in filenames :
            idx = f.find('_best.params')
            if idx > 0 :
                TRAINED_WEIGHTS = f
                BASE_MODEL = f[0:idx]

      #classes_df = pd.read_csv(os.path.join(model_dir, 'classes.csv'), header=None)
  #CLASSES = classes_df[1].tolist()
                                 
  #  with open(os.path.join(model_dir,'model_info.json')) as model_info :
  #      meta = json.load(model_info)
  #      BASE_MODEL = meta['base']
  #      TRAINED_WEIGHTS = meta['weights']

    
    return BASE_MODEL, TRAINED_WEIGHTS, CLASSES
   
def model_fn(model_dir):
    
    ##todo: should support GPU as well
    ctx = mx.cpu()
    
    base, w, cls = get_model_info(model_dir)
    net = gcv.model_zoo.get_model(base, classes=cls, pretrained_base=False)
    net.load_parameters(os.path.join(model_dir,w))
    
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    
    ##todo: should support GPU as well
    ctx = mx.cpu()
  
    x, image = gcv.data.transforms.presets.yolo.transform_test(mx.img.imdecode(data), 512)
    cid, score, bbox = net(x)  
    
    c= cid[0].asnumpy().reshape(cid[0].shape[0]*cid[0].shape[1])
    s=score[0].asnumpy().reshape(score[0].shape[0]*score[0].shape[1])
    bb= bbox[0].asnumpy().reshape(bbox[0].shape[0]*bbox[0].shape[1])
    
    stack = np.concatenate((c,s,bb))
   
    response_body = stack.tobytes()
    
    return response_body, output_content_type