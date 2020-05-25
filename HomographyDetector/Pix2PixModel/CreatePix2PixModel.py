# This file will create the pix2pixmodel
from .options import TestOptions
from .data_loader import CreateDataLoader
from .models import create_model



def CreatePix2PixModel(gpu):
    # --which_direction AtoB --model two_pix2pix --name soccer_seg_detection_pix2pix --output_nc 1 --dataset_mode aligned --which_model_netG unet_256 --norm batch --how_many 186 --loadSize 256

    opt = TestOptions()  # .parse()
    # print("0--")

    # Custom stuff that is normally passed on command line
    opt.dataroot = './ExtractPitchLines/datasets/soccer_seg_detection'
    opt.which_direction = 'AtoB'
    opt.model = 'two_pix2pix'
    opt.name = 'Linedetection'
    opt.output_nc = 1
    opt.dataset_mode = 'aligned'
    opt.which_model_netG = 'unet_256'
    opt.norm = 'batch'
    opt.how_many = 186
    opt.loadSize = 256

    # determine if you use GPU
    # Use GPU
    # opt.gpu_ids = [0]
    # Use CPU
    # opt.gpu_ids = []

    if gpu == True:
        opt.gpu_ids = [0]
    else:
        opt.gpu_ids = []

    # Default stuff
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.continue_train = False

    # Newly added
    opt.phase = 'test'
    opt.resize_or_crop = 'resize_and_crop'
    opt.isTrain = False
    opt.checkpoints_dir = './checkpoints'
    opt.input_nc = 3
    opt.ndf = 64
    opt.ngf = 64
    opt.no_dropout = False
    opt.init_type = 'normal'
    opt.which_epoch = 'latest'
    opt.which_model_netD = 'basic'

    # print(opt.dataroot)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    return model
