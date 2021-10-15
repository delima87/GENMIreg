
import os
from PIL.Image import LANCZOS
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch
import ntpath

def save_visuals(visuals):
    images = []
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        if label == "real_A" or label == "fake_B":
            images.append(im)
    return images

def predict_completion(dataroot,modelName, checkpointns_dir):
    opt = TestOptions()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot = dataroot
    opt.name = modelName
    opt.checkpoints_dir = checkpointns_dir
    opt.dataset_mode = "unaligned"
    opt.phase = "test"
    opt.max_dataset_size = float("inf")
    opt.direction = 'AtoB'
    opt.input_nc = 3
    opt.output_nc = 3
    opt.preprocess = "resize_and_crop"
    opt.load_size = 256
    opt.crop_size = 256
    opt.model = "cycle_gan"
    opt.no_flip = True
    opt.isTrain = False
    opt.display_winsize = 256
    opt.epoch = "latest"
    opt.verbose = True
    opt.suffix = ""
    opt.ngf = 64
    opt.ndf = 64
    opt.netG = "resnet_9blocks"
    opt.netD = "basic"
    opt.norm = "instance"
    opt.no_dropout = True
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.gpu_ids = '0'
    opt.load_iter = 0
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    opt.eval = True
    opt.num_test = 50
    opt.aspect_ratio = 1.0
    opt.n_layers_D = 3
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model gi
    model.setup(opt)  
    final_images = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # web_dir = os.path.join("results", opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        c_img = save_visuals(visuals)
        final_images.extend(c_img)
    print("Process correctly done", len(final_images))
if __name__ == "__main__":
    checkpoints = "./checkpoints"
    name_model = "feather_remote_sensing" 
    data = "./temp"
    predict_completion(data,name_model,checkpoints)
    