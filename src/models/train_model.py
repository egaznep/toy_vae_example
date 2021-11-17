import re
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import importlib

import torch
import torch.autograd
from torch.utils.data import dataset
import torch.utils.data.dataloader

import ignite.utils
import ignite.handlers.early_stopping
import ignite.engine
import ignite.metrics
import ignite.contrib.handlers
import ignite.contrib.handlers.tensorboard_logger
import ignite.handlers.param_scheduler

from copy import deepcopy

import tensorboardX

import src.models
import src.data.load_dataset
from src.config.load_config import load_config
from src.common import get_constructor, magma_init
import src.torch_extensions


class Training:
    def __init__(self, config, *args, **kwargs):
        # parse config

        self.seed = config['random']['seed']
        self.num_epoch = config['training']['num_epoch']
        self.dim_input = (1, config['model']['architecture']['num_input'])
        self.cfg = config['model']
        self.early_stopping = config['training']['early_stopping']
        self.reduce_lr_plateau = config['training']['reduce_lr_plateau']

        self.setup_cuda()
        self.dummy_input = torch.autograd.Variable(
            torch.zeros(self.dim_input).to(self.device))
        self.setup_tensorboard(
            config['experiment']['name'] + config['model']['name'], **config['tensorboard'])
        self.setup_model(**self.cfg)
        self.setup_ignite()

    def setup_cuda(self, cuda_device_id=0):
        torch.backends.cuda.fasval = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        magma_init()

    def setup_model(self, architecture, loss, optim, **kwargs):
        constructor = get_constructor('src.models', architecture['type'])
        self.model = constructor(**architecture).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=optim['lr'])

    def setup_ignite(self):
        ignite.utils.manual_seed(self.seed)

        val_metrics = {key: ignite.metrics.Loss(self.model.loggable_losses[key])
            for key in self.model.loggable_losses}

        def prepare_batch(batch, device=None, non_blocking=False, *args, **kwargs):
            converted = ignite.utils.convert_tensor(
                batch, device, non_blocking)
            return converted, converted

        def output_transform(x, y, y_pred, loss=None):
            return {'y': y, 'y_pred': y_pred, 'criterion_kwargs': {}, 'loss': loss}

        self.trainer = ignite.engine.create_supervised_trainer(
            self.model, self.optim, self.model.loss, device=self.device, prepare_batch=prepare_batch, output_transform=output_transform)
        self.evaluator = ignite.engine.create_supervised_evaluator(
            self.model, val_metrics, device=self.device, prepare_batch=prepare_batch, output_transform=output_transform)
        for mtrc in val_metrics:
            val_metrics[mtrc].attach(self.trainer, mtrc)
        
        # prevent messages from cluttering the log
        self.trainer.logger.setLevel(logging.WARN)
        self.evaluator.logger.setLevel(logging.WARN)

        # save graph to tensorboard
        self.tb_logger.writer.add_graph(self.model, self.dummy_input)

        # attach events - tensorboard loggers
        losses = [loss for loss in self.model.loggable_losses]
        self.tb_logger.attach_output_handler(
            self.trainer, ignite.engine.Events.EPOCH_COMPLETED, tag='training', 
            metric_names=losses,
            global_step_transform=ignite.contrib.handlers.tensorboard_logger.global_step_from_engine(self.trainer))
        self.tb_logger.attach_output_handler(
            self.evaluator, ignite.engine.Events.EPOCH_COMPLETED, tag='validation', 
            metric_names=losses,
            global_step_transform=ignite.contrib.handlers.tensorboard_logger.global_step_from_engine(self.trainer))

        # attach events - early stopping
        def score_function(engine):
            return -engine.state.metrics['loss']
        self.es = ignite.handlers.early_stopping.EarlyStopping(**self.early_stopping, score_function=score_function, trainer=self.trainer)
        self.evaluator.add_event_handler(ignite.engine.Events.COMPLETED, self.es)

        # attach events - learning rate scheduling
        self.ps = src.torch_extensions.ReduceLROnPlateauScheduler(self.optim, metric_name='loss', **self.reduce_lr_plateau)
        self.evaluator.add_event_handler(ignite.engine.Events.COMPLETED, self.ps)

        @self.trainer.on(ignite.engine.Events.STARTED)
        def on_start(engine):
            logging.info('Starting training')

        @self.trainer.on(ignite.engine.Events.COMPLETED)
        def on_complete(engine):
            torch.save(self.model.state_dict(), self.model_save_path)
            logging.info('Training complete. Saved model to:{}'.format(
                self.model_save_path))

        @self.evaluator.on(ignite.engine.Events.COMPLETED)
        def on_complete(engine):
            # print loss etc.
            logging.info(
                f'Avg validation loss: {engine.state.metrics["loss"]}')

    def train(self, train_loader, val_loader, model_save_path):
        self.model_save_path = model_save_path
        self.loss_list = []

        @self.trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
        def on_epoch_complete(engine):
            logging.info(
                f'Training epoch {engine.state.epoch} complete. Avg training loss: {engine.state.metrics["loss"]}')
            self.evaluator.run(val_loader)
        self.trainer.run(train_loader, self.num_epoch)

    def setup_tensorboard(self, folder_name, save_path, **kwargs):
        path = Path(save_path) / folder_name
        self.tb_logger = ignite.contrib.handlers.TensorboardLogger(
            log_dir=path)


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
    train_dataset = src.data.load_dataset.Waveform_dataset(
        data_path, '{}_train.hdf5'.format(dataset_name_prefix))
    val_dataset = src.data.load_dataset.Waveform_dataset(
        data_path, '{}_val.hdf5'.format(dataset_name_prefix))
    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset, **cfg['train_loader'])
    val_loader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset, **cfg['val_loader'])

    # model
    trainer = Training(cfg)
    model_save_path = Path(cfg['model']['path']) / cfg['model']['name']

    trainer.train(train_loader, val_loader, model_save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
