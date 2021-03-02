import argparse
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

def main(args):
    print('main started with args: {}'.format(args))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    data = np.load(args.prediction_result_path)
    pred, truth = (data[key] for key in data.files)
    for i in range(0,3,1):
        # idx=random.randrange(pred.shape[0])
        idx=i
        p,t=pred[idx,11,:],truth[idx,11,:]
        plt.figure(figsize=(10, 3.5))
        plt.plot(p,label='prediction')
        plt.plot(t,label='truth')
        plt.title('Test sample {}'.format(idx+1))
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(args.save, 'Figure_{}.png'.format(idx + 1)))
        # plt.show()


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    add_argument = ''
    sys.argv.extend(add_argument.split(' '))

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_result_path', default='../output/dcrnn_predictions_la.npz', type=str, help='the prediction results')
    parser.add_argument('--save', default='../figures/METR-LA',
                        type=str, help='The path to save the fig')

    args, unknown = parser.parse_known_args()

    print('unknown args:{}'.format(unknown))

    main(args)