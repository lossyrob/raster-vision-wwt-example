"""Script for running Raster Vision Jobs"""
import sys
import os
from subprocess import Popen
from rastervision.utils.misc import (terminate_at_exit)
from random import shuffle

import click
import logging

import rastervision as rv
from rastervision.experiment import (ExperimentLoader, LoaderError)
from rastervision.runner import (ExperimentRunner)
from rastervision.rv_config import RVConfig

log = logging.getLogger(__name__)

BASE_COMMAND = ['rastervision', '-p', 'wwt', 'run', 'local',
                '-e', 'wwt.chip_classification', '-a', 'root_uri', '/opt/data/rv_root']


def print_error(msg):
    click.echo(click.style(msg, fg='red'), err=True)

def add_exp_arg(cmd, k, v):
    cmd.extend(['-a', k, v])
    return cmd

def delete_some_negative_chips(chip_id, max_imbalance_factor=3):
    """This deletes out some negative chips, since there are many background examples
    and few postive examples, which can skew the model to predicting negative results.

    max_imbalance_factor is the multiple of negative chips that we will allow.
    """
    chip_dir = '/opt/data/rv_root/chip/{}/'.format(chip_id)
    if not os.path.exists(chip_dir):
        return

    no_plants_dir = os.path.join(chip_dir, 'training', 'no_plant')

    num_plants = len(os.listdir(os.path.join(chip_dir, 'training', 'plant')))
    no_plants =  os.listdir(no_plants_dir)
    num_no_plants = len(no_plants)
    if num_no_plants > (num_plants * max_imbalance_factor):
        shuffle(no_plants)
        for np in no_plants[num_plants * max_imbalance_factor:]:
            os.remove(os.path.join(no_plants_dir, np))
    print('Deleted {} negative chips to balance training data. Now {} positive chips, {} negative chips.'.format(num_no_plants - num_plants * max_imbalance_factor, num_plants, len(os.listdir(no_plants_dir))))

def add_options(cmd, rerun=False, test=False, dry_run=False, verbose=False,
                data_dir=None, bands=None, stats_id=None, chip_id=None, model_id=None,
                predict_output_dir=None):
    cmd = cmd.copy()

    if not data_dir:
        print_error("You must specify a data directory")
        sys.exit(1)

    cmd = add_exp_arg(cmd, 'data_dir', data_dir)

    if rerun:
        cmd.append('--rerun')

    if test:
        cmd = add_exp_arg(cmd, 'test', 'yes')

    if dry_run:
        cmd.append('--dry-run')

    if verbose:
        cmd = cmd[0:1] + ['-v'] + cmd[1:]

    if bands:
        cmd = add_exp_arg(cmd, 'bands', bands)

    if stats_id:
        cmd = add_exp_arg(cmd, 'stats_id', stats_id)
    else:
        cmd = add_exp_arg(cmd, 'stats_id', 'default')

    if chip_id:
        cmd = add_exp_arg(cmd, 'chip_id', chip_id)
    else:
        cmd = add_exp_arg(cmd, 'chip_id', 'default')

    if model_id:
        cmd = add_exp_arg(cmd, 'model_id', model_id)
    else:
        cmd = add_exp_arg(cmd, 'model_id', 'default')

    if predict_output_dir:
        cmd = add_exp_arg(cmd, 'predict_output_dir', predict_output_dir)

    return cmd

def run_command(cmd):
    print('Running command: {}'.format(' '.join(cmd)))

    process = Popen(cmd)
    terminate_at_exit(process)
    exitcode = process.wait()
    if exitcode != 0:
        sys.exit(exitcode)
    else:
        return 0

@click.group()
def main():
    pass

@main.command(
    'stats', short_help='Run Raster Vision stats computation (slow).',
    help=('Generate the stats.json that is required for resampling a dataset from '
          'uint16 to uint8 (byte) images. This is required until Raster Vision handles '
          'uint16 data natively\n'
          '\n'
          'DATA_DIR = The directory that holds the training data. Cannot contain spaces.\n'
          '\n'
          'STATS_ID = The ID that you will use in other commands to use these statistics.'))
@click.argument('data_dir')
@click.argument('stats_id')
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
@click.option('--test', '-t', is_flag=True, default=False,
              help='Run a test run with a small part of the dataset.')
@click.option('--dry-run', '-n', help='Do a dry run to inspect what commands  will be run.')
@click.option('--verbose', '-v', help='Verbose output.', is_flag=True, default=False)
def stats(data_dir, stats_id, rerun, test, dry_run, verbose):
    cmd = add_options(BASE_COMMAND, rerun=rerun, data_dir=data_dir, test=test,
                      dry_run=dry_run, verbose=verbose, stats_id=stats_id)
    cmd.append('analyze')
    run_command(cmd)

@main.command(
    'chip', short_help='Run Raster Vision chipping.',
    help=('Generate the training chips from a dataset.\n'
          '\n'
          'DATA_DIR = The directory that holds the training data. Cannot contain spaces.\n'
          '\n'
          'CHIP_ID = The ID that will be used in other commands for this set of training chips.'))
@click.argument('data_dir')
@click.argument('chip_id')
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
@click.option('--test', '-t', is_flag=True, default=False,
              help='Run a test run with a small part of the dataset.')
@click.option('--dry-run', '-n', help='Do a dry run to inspect what commands  will be run.')
@click.option('--verbose', '-v', help='Verbose output.', is_flag=True, default=False)
@click.option('--bands', '-b', help='Bands to use. USE 0 INDEXING! So the first band is at index 0. Default is 4,2,1. ', default='4,2,1')
@click.option('--stats-id', '-s', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the statistics in rv_root/analyze/default/stats.json')
@click.option('--max-imbalance-factor', '-m', default=3, type=int, help='Maximum imbalance factor for negative and positive chips. Deletes negative chips such that len(negative_chips) <= len(positive_chips) * max_imabalance_factor')
def chip(data_dir, chip_id, rerun, test, dry_run, verbose, bands, stats_id, max_imbalance_factor):
    cmd = add_options(BASE_COMMAND, rerun=rerun, data_dir=data_dir, test=test,
                      dry_run=dry_run, verbose=verbose, stats_id=stats_id, chip_id=chip_id)
    cmd.append('chip')
    run_command(cmd)

    # Post processing to balance out chips, since there are scarce positive examples.
    delete_some_negative_chips(chip_id, max_imbalance_factor)

@main.command(
    'train', short_help='Run Raster Vision training.',
    help=('Run the train, predict, eval, and bundle steps of the Raster Vision process. '
          'Data from these commands will be found in rv_root, under the respective commands, '
          'in a folder with the same name as the MODEL_ID.\n'
          '\n'
          'DATA_DIR = The directory that holds the training data. Cannot contain spaces.\n'
          '\n'
          'MODEL_ID = The ID that you will use in other commands to use these statistics.'))
@click.argument('data_dir')
@click.argument('model_id')
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
@click.option('--test', '-t', is_flag=True, default=False,
              help='Run a test run with a small part of the dataset.')
@click.option('--dry-run', '-n', help='Do a dry run to inspect what commands  will be run.')
@click.option('--verbose', '-v', help='Verbose output.', is_flag=True, default=False)
@click.option('--bands', '-b', help='Bands to use. USE 0 INDEXING! So the first band is at index 0. Default is 4,2,1. ', default='4,2,1')
@click.option('--stats-id', '-s', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the statistics in rv_root/analyze/default/stats.json')
@click.option('--chip-id', '-c', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the chips in rv_root/chip/default/')
def train(data_dir, rerun, model_id, test, dry_run, verbose, bands, stats_id, chip_id):
    cmd = add_options(BASE_COMMAND, rerun=rerun, data_dir=data_dir, test=test,
                      dry_run=dry_run, verbose=verbose, stats_id=stats_id,
                      chip_id=chip_id, model_id=model_id)
    cmd.extend(['train', 'predict', 'eval', 'bundle'])
    run_command(cmd)

@main.command(
    'predict', short_help='Predict against a set of unlabeled images using a trained model.',
    help=('Generate the stats.json that is required for resampling a dataset from '
          'uint16 to uint8 (byte) images. This is required until Raster Vision handles '
          'uint16 data natively\n'
          '\n'
          'INPUT_DIR = The directory that holds the data to predict on. Cannot contain spaces.\n'
          '\n'
          'OUTPUT_DIR = The directory to save the prediction results to. Cannot contain spaces.'))
@click.argument('input_dir')
@click.argument('output_dir')
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
@click.option('--test', '-t', is_flag=True, default=False,
              help='Run a test run with a small part of the dataset.')
@click.option('--dry-run', '-n', help='Do a dry run to inspect what commands  will be run.')
@click.option('--verbose', '-v', help='Verbose output.', is_flag=True, default=False)
@click.option('--bands', '-b', help='Bands to use. USE 0 INDEXING! So the first band is at index 0. Default is 4,2,1. ', default='4,2,1')
@click.option('--stats-id', '-s', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the statistics in rv_root/analyze/default/stats.json')
@click.option('--chip-id', '-c', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the chips in rv_root/chip/default/')
@click.option('--model-id', '-m', default='default', help='Stats ID to run against a specific statistics analysis. Default will use the chips in rv_root/chip/default/')
def predict(input_dir, output_dir, rerun, test, dry_run, verbose, bands, stats_id, chip_id, model_id):
    cmd = add_options(BASE_COMMAND, rerun=rerun, data_dir=input_dir, test=test,
                      dry_run=dry_run, verbose=verbose, stats_id=stats_id,
                      chip_id=chip_id, model_id=model_id, predict_output_dir=output_dir)
    cmd.append('predict')
    run_command(cmd)


if __name__ == '__main__':
    main()
