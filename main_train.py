import os
import random
from easydict import EasyDict as edict
import json
import logging
import sys
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
from datetime import date

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")


# dataset
from lib.dataloader import PluckerData3D_precompute
# configuration
from config import get_config
# trainer
from trainer_plucker import PluckerTrainer


# main function

def main(configs):

    train_loader = DataLoader(PluckerData3D_precompute( phase='train', config = configs), batch_size=configs.train_batch_size, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(PluckerData3D_precompute(phase='valid', config = configs), batch_size=1, shuffle=False, drop_last=False, num_workers=1)

    trainer = PluckerTrainer(configs, train_loader, val_loader)
    # train
    trainer.train()


if __name__ == '__main__':

    configs = get_config()
    # -------------------------------------------------------------
    """You can change the configurations here or in the file config.py"""

    # select dataset
    configs.dataset = "structured3D"
    # configs.dataset = "semantic3D"

    # dataset path
    configs.data_dir = "./dataset"

    # select which GPU to be used
    configs.gpu_inds = 0

    # This is a model number, set it to whatever you want
    configs.model_nb = str(date.today())

    # training batch size
    configs.train_batch_size = 12

    # learning rate
    configs.train_lr = 1e-3

    dconfig = vars(configs)

    # if your training is terminated unexpectly, uncomment the following line and set the resume_dir to continue
    # configs.resume_dir = "./output"

    if configs.resume_dir:
        resume_config = json.load(open(configs.resume_dir + "/" + configs.dataset + "/" + configs.model_nb + '/config.json', 'r'))
        for k in dconfig:
            if k in resume_config:
                dconfig[k] = resume_config[k]
        # most recent checkpoint
        dconfig['resume'] = os.path.join(resume_config['out_dir'], resume_config['dataset'], configs.model_nb) + '/checkpoint.pth'
        # the best checkpoint
        # dconfig['resume'] = os.path.join(resume_config['out_dir'], resume_config['dataset'], configs.model_nb) + '/best_val_checkpoint.pth'

    else:
        dconfig['resume'] = None

    logging.info('===> Configurations')
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))


    # Convert to dict
    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True


    main(configs)






