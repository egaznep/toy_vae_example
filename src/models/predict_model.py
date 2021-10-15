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

    def setup_model(self, architecture, *args, **kwargs):
        self.model = src.models.autoencoder.AE(**architecture).to(self.device)
        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                x_hat = self.model(x)
                return_list.append((x.cpu().numpy().ravel(), x_hat.cpu().numpy().ravel()))
        return return_list
        
    def to_latent_space(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                z = self.model.to_latent_space(x)
                x_hat = self.model.to_waveform(z)
                return_list.append((x.cpu().numpy().ravel(), z.cpu().numpy().ravel()))
        return return_list

    def to_waveform(self, test_loader):
        self.model.eval()

        return_list = []
        with torch.no_grad():
            for i, z in enumerate(test_loader):
                z = z.to(self.device)
                x_hat = self.model.to_waveform(z)
                return_list.append((z.cpu().numpy().ravel(), x_hat.cpu().numpy().ravel()))
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
    # TODO: load this too
    t = np.arange(100)/16000

    # data loader
    dataset_name_prefix = cfg['dataset']['name']
    test_dataset = src.data.load_dataset.Waveform_dataset(data_path, '{}_test.hdf5'.format(dataset_name_prefix), size=N)
    test_loader = torch.utils.data.dataloader.DataLoader(dataset=test_dataset, **cfg['test_loader'])

    # model
    model_save_name = "{}_{}".format(cfg['experiment']['name'], cfg['model']['name'])
    model_save_path = project_dir / cfg['model']['path'] / model_save_name 

    predictor = Prediction(cfg['model'], model_save_path, **cfg['training'])

    results = predictor.predict(test_loader)

    #automatically visualize if called as the main script
    if __name__ == '__main__':
        visualize_signal_pairs(t, results)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    results = main()