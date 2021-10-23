import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import importlib

import torch
import torch.autograd
from torch.utils.data import dataset
import torch.utils.data.dataloader

import tensorboardX

import src.models
import src.data.load_dataset
from src.config.load_config import load_config
from src.common import get_constructor

class Training:
    def __init__(self, config, *args, **kwargs):
        # parse config
        
        self.seed = config['random']['seed']
        self.num_epoch = config['training']['num_epoch']
        self.dim_input = (1,config['model']['architecture']['num_input'])
        self.cfg = config['model']

        self.setup_cuda()
        self.dummy_input = torch.autograd.Variable(torch.zeros(self.dim_input).to(self.device))
        self.setup_tensorboard(config['experiment']['name'],**config['tensorboard'])
        self.setup_model(**self.cfg)

    def setup_cuda(self, cuda_device_id=0):
        torch.backends.cuda.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)

    def setup_model(self, architecture, loss, optim, **kwargs):
        constructor = get_constructor('src.models', architecture['type'])
        self.model = constructor(**architecture).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=optim['lr'])
        self.add_tensorboard_graph(self.model)

    def train(self, train_loader, test_loader, model_save_path):
        logging.info('Starting training')
        for i in range(self.num_epoch):
            self.epoch = i
            self.train_epoch(train_loader)
            self.test_epoch(test_loader)
        logging.info('Saved model to:{}'.format(model_save_path))
        torch.save(self.model.state_dict(), model_save_path)

    def train_epoch(self, train_loader):
        self.model.train()
        loss_list = []
        for i, x in enumerate(train_loader):
            x = x.to(self.device)

            self.optim.zero_grad()
            x_hat = self.model(x)
            loss = self.model.loss(x, x_hat)
            loss.backward()
            self.optim.step()
            loss_list.append(loss)
        
        self.calculate_loss_stats(loss_list)

    def test_epoch(self, test_loader):
        self.model.eval()

        with torch.no_grad():
            loss_list = []
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                x_hat = self.model(x)
                loss = self.model.loss(x, x_hat)
                loss_list.append(loss)
        
        self.calculate_loss_stats(loss_list, False)
        

    def calculate_loss_stats(self, loss_list, is_train=True):
        loss_tensor = torch.stack(loss_list).detach()

        avg = torch.mean(loss_tensor).cpu().numpy()[()]
        max = torch.max(loss_tensor).cpu().numpy()[()]
        min = torch.min(loss_tensor).cpu().numpy()[()]

        mode = 'Train' if is_train else 'Test'
        logging.info("Epoch: {}\t {}_loss: (Min, Avg, Max) = {}".format(self.epoch+1, mode, (min, avg, max)))
        self.writer.add_scalar(mode + 'Loss', avg, self.epoch)

    def setup_tensorboard(self, folder_name, save_path, **kwargs):
        path = Path(save_path) / folder_name
        self.writer = tensorboardX.SummaryWriter(path)
    
    def add_tensorboard_graph(self, model):
        self.writer.add_graph(self.model, self.dummy_input)
        pass

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

    # data loader
    dataset_name_prefix = cfg['dataset']['name']
    train_dataset = src.data.load_dataset.Waveform_dataset(data_path, '{}_train.hdf5'.format(dataset_name_prefix))
    test_dataset = src.data.load_dataset.Waveform_dataset(data_path, '{}_test.hdf5'.format(dataset_name_prefix))
    train_loader = torch.utils.data.dataloader.DataLoader(dataset=train_dataset, **cfg['train_loader'])
    test_loader = torch.utils.data.dataloader.DataLoader(dataset=test_dataset, **cfg['test_loader'])

    # model
    trainer = Training(cfg)
    model_save_path = Path(cfg['model']['path']) / cfg['model']['name'] 

    trainer.train(train_loader, test_loader, model_save_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()