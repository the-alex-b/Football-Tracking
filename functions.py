import sys
import os
sys.path.append('./ExtractPitchLines')

# os.chdir('./ExtractPitchLines')

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
# from util import html

def CreatePix2PixModel():
# --which_direction AtoB --model two_pix2pix --name soccer_seg_detection_pix2pix --output_nc 1 --dataset_mode aligned --which_model_netG unet_256 --norm batch --how_many 186 --loadSize 256

    opt = TestOptions().parse()
    print(opt)
    print("0--")

    # Custom stuff that is normally passed on command line
    opt.dataroot = './ExtractPitchLines/datasets/soccer_seg_detection'
    opt.which_direction = 'AtoB'
    opt.model = 'two_pix2pix'
    opt.name =  'Linedetection'
    opt.output_nc = 1
    opt.dataset_mode = 'aligned'
    opt.which_model_netG= 'unet_256'
    opt.norm = 'batch'
    opt.how_many = 186
    opt.loadSize = 256
    
    # determine if you use GPU
    # opt.gpu_ids = -1

    # Default stuff
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.continue_train = False

    print(opt.dataroot)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # visualizer = Visualizer(opt)
    # # create website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch)

        # return model
    return model