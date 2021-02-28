

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys

import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
from lib.modelarts_tools import prepare_data_on_modelarts, push_data_back_on_modelarts
import os



def add_prefix(pref,tar,key):
    tar[key]=os.path.join(pref,tar[key])
    return


def main(args):
    print('main started with args: {}'.format(args))
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        add_prefix(args.train_local,supervisor_config,'base_dir')
        add_prefix(args.data_local,supervisor_config['data'],'dataset_dir')
        add_prefix(args.data_local,supervisor_config['data'],'graph_pkl_filename')

        print('using supervisor_config: {}'.format(supervisor_config))
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')


        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(os.path.join(args.data_local,graph_pkl_filename))

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    # sys.path.append(os.getcwd())
    add_argument = ''
    sys.argv.extend(add_argument.split(' '))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', default='../data', type=str, help='the training and validation data path')
    parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
    parser.add_argument('--train_url', default='../output',  type=str, help='the path to save training outputs')
    parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
    parser.add_argument('--local_data_root', default='/cache', type=str,
                        help='a directory used for transfer data between local path and OBS path')
    parser.add_argument('--config_filename', default='model/dcrnn_la.yaml', type=str,
                        help='Configuration filename for restoring the model. start from data_dir')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args, unknown = parser.parse_known_args()

    print('unknown args:{}'.format(unknown))

    args= prepare_data_on_modelarts(args)

    args.config_filename=os.path.join(args.data_local,args.config_filename)
    try:
        main(args)
    except KeyboardInterrupt as e:
        print('main is interrupted by {} '.format(e))
    # Exception 捕获除了 SystemExit 、 KeyboardInterrupt 和 GeneratorExit 之外的所有异常。
    # 还想捕获这三个异常，将 Exception 改成 BaseException 即可。
    except BaseException as e:
        print('main encountered an exception : {} '.format(e))
    else:
        print('main ended without exception')

    print('try to move data to obs')
    push_data_back_on_modelarts()
    print('moving data finished.')
