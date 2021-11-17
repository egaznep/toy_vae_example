# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import h5py
from src.config.load_config import load_config

def generate_sinusoids(frequencies, amplitudes, phases, duration, fs, *args, **kwargs):
    t = (np.arange(duration)/fs)[None,:]
    sinusoids = amplitudes*np.sin(2*np.pi*frequencies*t + phases, dtype=np.float32)
    sinusoids = sinusoids.astype(np.float32)
    
    return t, sinusoids

# we don't have a "raw" dataset pulled from somewhere, 
# rather we generate our own data.
def generate_random_sinusoids(seed=0, N=10000, fs=16000, duration=100, frequency_range=[320, 8000], amplitude_range=[0,10], phase_range=[-np.pi, np.pi], *args, **kwargs):
    """ Returns sinusoids.

    Generates sinusoids according to the configuration (with fixed sampling 
    rate, length, random but choosable frequency, amplitude, phase)

    Args:
        seed (int, optional): Random generator seed. Defaults to 0.
        N (int, optional): [description]. Defaults to 10000.
        fs (int, optional): [description]. Defaults to 16000.
        duration (int, optional): Length of the sinusoid in samples. Defaults to 100.
        amplitude_range (list, optional): [description]. Defaults to [-10,10].
        phase_range (list, optional): [description]. Defaults to [-np.pi, np.pi].
    """
    assert len(frequency_range) == 2
    assert len(amplitude_range) == 2
    assert len(phase_range) == 2
    assert frequency_range[1] >= frequency_range[0]
    assert amplitude_range[1] >= amplitude_range[0]
    assert phase_range[1] >= phase_range[0]

    def rescale_unif(arr, minimum, maximum, *args, **kwargs):
        mean = (minimum + maximum)/2
        scale = maximum - minimum
        return (arr - 1/2) * scale + mean

    rng = np.random.default_rng(seed)
    frequencies = rescale_unif(rng.random(size=N, dtype=np.float32), *frequency_range)[:,None]
    amplitudes = rescale_unif(rng.random(size=N, dtype=np.float32), *amplitude_range)[:,None]
    phases = rescale_unif(rng.random(size=N, dtype=np.float32), *phase_range)[:,None]
    
    _, sinusoids = generate_sinusoids(frequencies, amplitudes, phases, duration, fs)
    return sinusoids, frequencies, amplitudes, phases


@click.command()
@click.argument('generator_config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(generator_config_path, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read generator config
    config = load_config(generator_config_path)

    sinusoids, frequencies, amplitudes, phases = generate_random_sinusoids(**config)

    # features file
    with h5py.File(Path(output_filepath) / 'X.hdf5', 'w') as X_file:
        X_file.create_dataset("sinusoids", data=sinusoids)
    
    # labels file
    with h5py.File(Path(output_filepath) / 'y.hdf5', 'w') as y_file:
        y_file.create_dataset("frequencies", data=frequencies)
        y_file.create_dataset("amplitudes", data=amplitudes)
        y_file.create_dataset("phases", data=phases)
    
    logger.info('completed generating N={} sinusoids, wrote to: {}'.format(len(sinusoids), output_filepath))
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
