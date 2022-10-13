import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch_geometric.data import DataLoader
import Data
import GraphNet
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main():
    # some experiment settings
    model_path = './trained/GCN'
    data_path = './brain_network'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    logging.info('Prepare data...')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataLoader = DataLoader(Data.Brain_network(data_path), batch_size=1, shuffle=False, **kwargs)

    # network construction
    logging.info('Initialize network...')
    net = GraphNet.GraphNet(input_dim=128)
    logging.info('  + Number of Model params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    # move the network to GPU/CPU
    net = net.to(device)

    # get trained model
    save_model = torch.load(os.path.join(model_path, 'model_GraphTransformer.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    logging.info("Model restored from file: {}".format(model_path))

    # testing for samples
    net.eval()
    pred_age = list()
    true_age = list()

    for batch_idx, data in enumerate(dataLoader):
        if batch_idx == 4:
            continue
        # get the output from network
        data = data.to(device)

        with torch.no_grad():
            output = net(data)

        pred_age.append(output.item())
        true_age.append(data.y.item())

    pred_age = np.array(pred_age)
    true_age = np.array(true_age)

    # print the pred_age and true_age
    print('estimated age is:')
    print(pred_age)
    print('true age is:')
    print(true_age)

    # index calculation
    MAE = np.mean(np.abs(true_age - pred_age))
    RMSE = np.sqrt(np.mean(np.square(true_age - pred_age)))

    # print the test results
    logging.info('MAE: {:.8f} \t RMSE: {:.8f}'.format(MAE, RMSE))

if __name__ == '__main__':
    main()


