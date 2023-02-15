from utils.data_loader_multifiles import GetDataset, GetCosmoDataset

class Params():
    dt = 1
    n_history = 0
    in_channels  = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    out_channels = [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    crop_size_x = None
    crop_size_y = None
    roll = False
    two_step_training = False
    orography = False
    add_noise = False
    normalize = False
    global_means_path = '/scratch/f1000/dealmeih/ds/FCN/FCN_ERA5_data_v0/global_means.npy'
    global_stds_path  = '/scratch/f1000/dealmeih/ds/FCN/FCN_ERA5_data_v0/global_stds.npy'
    add_grid = False

if __name__ == "__main__":
    params = Params()
    # dataset = GetDataset(params, '/scratch/f1000/dealmeih/ds/FCN/FCN_ERA5_data_v0/test', train=False)
    dataset = GetCosmoDataset(params, '/scratch/f1000/dealmeih/m01/prod_extr/2015', train=False)

    print(len(dataset), len(dataset[0]))
    print(dataset[0][0].shape, dataset[0][1].shape)

