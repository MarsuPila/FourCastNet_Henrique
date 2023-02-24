import sys
import os.path
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.data_loader_multifiles import GetDataset, GetCosmoDataset

class Params():
    dt = 1
    n_history = 4
    # in_channels  = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    in_channels = ['U_10M', 'V_10M', 'PMSL', 'PS', 'T_2M', 'FI', 'T', 'U', 'u_100', 'V', 'v_100', 'RELHUM']
    out_channels = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    crop_size_x = None
    crop_size_y = None
    roll = False
    two_step_training = True
    orography = False
    add_noise = False
    normalize = False
    global_means_path = None #'/scratch/f1000/dealmeih/ds/FCN/FCN_ERA5_data_v0/global_means.npy'
    global_stds_path  = None #'/scratch/f1000/dealmeih/ds/FCN/FCN_ERA5_data_v0/global_stds.npy'
    add_grid = False
    normalization = 'zscore'

    # invars_2d = ['U_10M', 'V_10M', 'PMSL', 'PS', 'T_2M']
    # invars_3d = ['FI', 'T', 'U', 'u_100', 'V', 'v_100', 'RELHUM']

if __name__ == "__main__":
    params = Params()
    if sys.argv[1] == 'cosmo':
        params.era5 = False
        if len(sys.argv)>3:
            params.noramlize = True
            params.global_means_path = os.path.join(str(sys.argv[3]), 'global_means.npy')
            params.global_stds_path  = os.path.join(str(sys.argv[3]), 'global_stds.npy')
        dataset = GetCosmoDataset(params, str(sys.argv[2]), train=False)
    else: # era5
        params.era5 = True
        params.in_channels = params.out_channels
        dataset = GetDataset(params, str(sys.argv[2]), train=False)

    print(len(dataset))
    print(dataset[0][0].shape, dataset[-1][1].shape)

