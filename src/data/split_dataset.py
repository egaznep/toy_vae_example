# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import h5py
from sklearn.model_selection import train_test_split

from src.config.load_config import load_config
from copy import deepcopy

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(config_path, input_filepath, output_filepath):
    """ Runs scripts to split processed data (from ../processed) 
        into train and test datasets, again in (../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting train and test...')

    # TODO: make it readable from a config file
    cfg = load_config(config_path)
    percent = deepcopy(cfg['percent'])
    suffixes = cfg['suffix']
    arrays = {}
    # identifiers for label arrays
    label_arrays = []

    # ensure that we have blank files for splits
    for d in ['X', 'y']:
        for suffix in suffixes:
            with h5py.File(Path(output_filepath) / f'{d}_{suffix}.hdf5', 'w') as X_file:
                pass

    # load features
    with h5py.File(Path(input_filepath) / 'X.hdf5', 'r') as X_file:
        N = None
        for entry in X_file:
            arrays[entry] = X_file[entry][:]
            if N is None:
                N = len(arrays[entry])
    
    # load labels
    with h5py.File(Path(input_filepath) / 'y.hdf5', 'r') as y_file:
        for entry in y_file:
            assert entry not in arrays # throw an exception if data and labels have entry with same name
            arrays[entry] = y_file[entry][:]
            label_arrays.append(entry)

    # perform split
    assert sum(percent) == 1., f"percent must add to 1, but it adds to sum{percent} = {sum(percent)}"
    remainder = [arrays[x] for x in arrays]
    for i, (suffix, per) in enumerate(zip(suffixes,percent)):
        # split
        print('Aye!', len(suffixes), len(arrays), percent, i, per)
        if i == len(suffixes)-1:
            current_split = remainder
        else:
            splits = train_test_split(*remainder, test_size=1-per, random_state=0)
            current_split = splits[::2]
        # save current
        for split,entry in zip(current_split,arrays):
            print(len(split), suffix, entry)
            if entry not in label_arrays:
                with h5py.File(Path(output_filepath) / f'X_{suffix}.hdf5', 'a') as X_file:
                    X_file.create_dataset(entry, data=split)
            else:
                with h5py.File(Path(output_filepath) / f'y_{suffix}.hdf5', 'a') as y_file:
                    y_file.create_dataset(entry, data=split)

        # update remainder and percentages
        remainder = splits[1::2]
        percent[i+1:] = [min(value / (1-percent[i]),1) for value in percent[i+1:]]

    logger.info(f'completed splitting dataset into {suffixes} = ({[int(N*p) for p in cfg["percent"]]})')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
