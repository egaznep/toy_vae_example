import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np

import torch
from torch.utils.data import dataset
import torch.utils.data.dataloader

import src.models.autoencoder
import src.data.load_dataset
from src.config.load_config import load_config
from src.visualization.visualize import visualize_signal_pairs
from src.common import get_constructor, magma_init


class Prediction:
    def __init__(self, model_config, model_save_path, *args, **kwargs):
        self.cfg = model_config
        self.model_save_path = model_save_path
        self.setup_cuda()
        self.setup_model(**self.cfg)

    def setup_cuda(self, cuda_device_id=0):
        torch.backends.cuda.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda')
        magma_init()

    def setup_model(self, architecture, *args, **kwargs):        
        constructor = get_constructor('src.models', architecture['type'])
        self.model = constructor(**architecture).to(self.device)
        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                x_hat = self.model(x)
                return_list.append(x_hat.cpu().numpy())
        return return_list
        
    def to_latent_space(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                z = self.model.to_latent_space(x)
                return_list.append(z.cpu().numpy())
        return return_list

    def to_waveform(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, z in enumerate(test_loader):
                z = z.to(self.device)
                x_hat = self.model.to_waveform(z)
                return_list.append(x_hat.cpu().numpy())
        return return_list

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('experiment_cfg_path', type=click.Path(exists=True))
def main(data_path, experiment_cfg_path):
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    data_path = Path(data_path)

    # config loader
    cfg = load_config(experiment_cfg_path) 
    N = cfg['prediction']['num_samples']
    L = cfg['model']['architecture']['num_input']
    t = np.arange(L)/16000

    # data loader
    dataset_name_prefix = cfg['dataset']['name']
    test_dataset = src.data.load_dataset.Waveform_dataset(data_path, '{}_test.hdf5'.format(dataset_name_prefix), size=N)
    test_loader = torch.utils.data.dataloader.DataLoader(dataset=test_dataset, **cfg['test_loader'])

    # model
    model_save_path = Path(cfg['model']['path']) / cfg['model']['name'] 

    predictor = Prediction(cfg['model'], model_save_path, **cfg['training'])

    inputs = [x.numpy() for x in test_loader]
    results = predictor.predict(test_loader)

    inputs, results = np.asarray(inputs), np.asarray(results)
    inputs = np.reshape(inputs, (N, -1))
    results = np.reshape(results, (N, -1))

    #automatically visualize if called as the main script
    if __name__ == '__main__':
        visualize_signal_pairs(t, inputs, results)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    results = main()