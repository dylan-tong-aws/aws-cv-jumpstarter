from train_srgan import SRGenerator
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from matplotlib import pyplot as plt
import cv2
from mxnet import image
import argparse
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test with srgan gan networks.')
    parser.add_argument('--images', type=str, required=True,
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu id: e.g. 0. use -1 for CPU')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    # context list
    if opt.gpu_id == '-1':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(int(opt.gpu_id.strip()))

    netG = SRGenerator()
    netG.load_parameters(opt.pretrained)
    netG.collect_params().reset_ctx(ctx)
    image_list = [x.strip() for x in opt.images.split(',') if x.strip()]
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ax = None
    for image_path in image_list:

        print("Original image:")
        #img = image.imread(image_path)
        orig_image = Image.open(image_path)
        plt.imshow(orig_image)
        plt.show()

        w,h = orig_image.size
        cropped_w_min = int(w*0.1)
        cropped_h_min = int(h*0.1)
        cropped_w = int(w*0.65)
        cropped_h = int(h*0.65)
        
        print("Original image: {}, {} Cropped Image: {}, {}, {}, {}".format(w,h,cropped_w_min,cropped_h_min,cropped_w,cropped_h))
        cropped_im = orig_image.crop((cropped_w_min,cropped_h_min,cropped_w,cropped_h))
        plt.imshow(cropped_im)  
        plt.show()
        
        img = mx.nd.array(np.array(cropped_im))
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        
        output = netG(img)
        
        b,c,w,h = output.shape
        predict = mx.nd.squeeze(output)
        predict = ((predict.transpose([1,2,0]).asnumpy() * 0.5 + 0.5) * 255).astype('uint8')
        plt.imshow(predict)
        plt.show()