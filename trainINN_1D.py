import numpy as np
import torch
import time
from datetime import datetime
from torch.utils.data import TensorDataset
import torch.utils.data as Data
import h5py
import utilities as utl
import network as net
from utilities import preprocess_train

if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda:0")
    print(device)
    train_data_folder = "./training_data/"
    train_data_h5name = "2_5_layers_100000.h5"

    epoch = 200 

    with h5py.File(train_data_folder + train_data_h5name, 'r') as h5f:
        rho_raw = np.log10(np.array(h5f['rho']))  # shape = (n_trace, n_pixel)
        curve_raw = np.array(h5f['curve'])  # shape = (n_trace, n_exp)

    n_trace, n_exp = curve_raw.shape  

    curve_raw,scaler = preprocess_train(curve_raw)

    n_trace, x_dim, y_dim, e_dim, z_dim, tot_dim = utl.get_dims(rho_raw, curve_raw)

    zero_padding = np.zeros((n_trace, y_dim))
    z = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), n_trace)

    n_train = int(n_trace * .8)
    n_test= n_trace - n_train

    rho_train,rho_test = utl.split_data(rho_raw,n_train,n_test)
    curve_train,curve_test = utl.split_data(curve_raw,n_train,n_test)
    zeros_train,zeros_test = utl.split_data(zero_padding,n_train,n_test)
    z_train, z_test = utl.split_data(z, n_train, n_test)

    # INN construction, input [rho, padding], output [z, curve]
    # Input and output data for training

    in_train = np.concatenate([rho_train, zeros_train], axis=-1).astype('float32') # numpy array
    out_train = np.concatenate([z_train, curve_train], axis=-1).astype('float32') # numpy array

    in_train_ts = torch.from_numpy(in_train)
    out_train_ts = torch.from_numpy(out_train)

    torch_dataset = TensorDataset(in_train_ts, out_train_ts)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = 128,
        shuffle = True,
        num_workers = 2)

    # Input and output data for test

    in_test = np.concatenate([rho_test, zeros_test], axis=-1).astype('float32')
    out_test = np.concatenate([z_test, curve_test], axis=-1).astype('float32')

    in_test_ts = torch.from_numpy(in_test)
    out_test_ts = torch.from_numpy(out_test)

    inn = net.define_inn(tot_dim, n_layer=5)
    inn.to(device)

    dim_dict = {"model": x_dim,
                "data": y_dim,
                "latent": z_dim,
                "pad": e_dim,
                }

    # weight_dict = {"forward": 0.0066325,
    #                "inverse": 0.0066634,
    #                "latent":  0.0012180, 
    #                "pad":     0.0010331,
    #                "boundary":0.0049588,
    #               }
    weight_dict = {"forward": 4.683579495772815,
                   "inverse": 2.346344381183016,
                   "latent":  1.6935092760189019, 
                   "pad":     1.2761187405742211,
                   "boundary":0.5635023718959832,
                  }
    loss_curve = net.inn_forward(inn, device, loader, epoch,
                    in_train_ts, out_train_ts, in_test_ts, out_test_ts,
                    dim_dict, weight_dict,
                    lr=0.00140748034806598, weight_decay=1.1777147444576608e-05, gamma=0.9587121634680469,
                    save_folder="./saved_network/")

    timestamp = datetime.today().strftime('%m-%d-%Y_%H-%M-%S')
    with h5py.File(f"./saved_network/loss_curve_{timestamp}.h5", 'w') as h5f:
        h5f.create_dataset("train_L_pred", data=loss_curve[0][:, 0])
        h5f.create_dataset("train_L_inv", data=loss_curve[0][:, 1])
        h5f.create_dataset("train_L_latent", data=loss_curve[0][:, 2])
        h5f.create_dataset("train_L_pad", data=loss_curve[0][:, 3])
        h5f.create_dataset("train_L_boundary", data=loss_curve[0][:, 4])
        h5f.create_dataset("train_total_loss", data=loss_curve[0][:, 5])

        h5f.create_dataset("test_L_pred", data=loss_curve[1][:, 0])
        h5f.create_dataset("test_L_inv", data=loss_curve[1][:, 1])
        h5f.create_dataset("test_L_latent", data=loss_curve[1][:, 2])
        h5f.create_dataset("test_L_pad", data=loss_curve[1][:, 3])
        h5f.create_dataset("test_L_boundary", data=loss_curve[1][:, 4])
        h5f.create_dataset("test_total_loss", data=loss_curve[1][:, 5])

    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))

