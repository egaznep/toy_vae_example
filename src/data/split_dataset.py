# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import h5py
import sklearn.model_selection

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs scripts to split processed data (from ../processed) 
        into train and test datasets, again in (../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting train and test...')

    # TODO: make it readable from a config file
    train_percent = 0.8

    # features file
    with h5py.File(Path(input_filepath) / 'X.hdf5', 'r') as X_file:
        sinusoids = X_file['sinusoids'][:]
        N = len(sinusoids)
    
    # labels file
    with h5py.File(Path(input_filepath) / 'y.hdf5', 'r') as y_file:
        frequencies = y_file['frequencies'][:]
        amplitudes = y_file['amplitudes'][:]
        phases = y_file['phases'][:]

    s_tr, s_te, f_tr, f_te, a_tr, a_te, p_tr, p_te = \
        sklearn.model_selection.train_test_split(sinusoids, frequencies,
        amplitudes, phases, train_size=train_percent, random_state=0)

    # features train
    with h5py.File(Path(output_filepath) / 'X_train.hdf5', 'w') as X_file:
        X_file.create_dataset("sinusoids", data=s_tr)
    # features test
    with h5py.File(Path(output_filepath) / 'X_test.hdf5', 'w') as X_file:
        X_file.create_dataset("sinusoids", data=s_te)
    
    # labels file
    with h5py.File(Path(output_filepath) / 'y_train.hdf5', 'w') as y_file:
        y_file.create_dataset("frequencies", data=f_tr)
        y_file.create_dataset("amplitudes", data=a_tr)
        y_file.create_dataset("phases", data=p_tr)
    # labels file
    with h5py.File(Path(output_filepath) / 'y_test.hdf5', 'w') as y_file:
        y_file.create_dataset("frequencies", data=f_te)
        y_file.create_dataset("amplitudes", data=a_te)
        y_file.create_dataset("phases", data=p_te)

    
    logger.info('completed splitting dataset into (train, test) = ({},{})'.format(len(f_tr), len(f_te)))
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
