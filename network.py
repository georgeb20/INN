import numpy as np
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from datetime import datetime
import time
from torch.optim.lr_scheduler import ExponentialLR
import utilities as utl

def subnet_fc(c_in, c_out):
    subnet = nn.Sequential(
        nn.Linear(c_in, 512), 
        nn.ReLU(0.1),
        nn.Linear(512, c_out)
    )
    for layer in subnet.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    return subnet

def define_inn(io_dim: int, n_layer: int=4):
    inn = Ff.SequenceINN(io_dim)
    for k in range(n_layer):
        inn.append(Fm.GLOWCouplingBlock, subnet_constructor=subnet_fc)
    total_params = sum(p.numel() for p in inn.parameters())
    print(f"Number of parameters: {total_params/1e6:.2f}M")

    return inn

def inn_forward(inn_network, device, dataloader, epoch: int,
                in_train_ts, out_train_ts, in_test_ts, out_test_ts,
                dim_dict, weight_dict,
                lr=0.001, weight_decay=0.00001, gamma=0.995,
                save_folder="./saved_network/"):
    
    x_dim = dim_dict["model"]
    y_dim = dim_dict["data"]
    z_dim = dim_dict["latent"]
    e_dim = dim_dict["pad"]

    w_pred = weight_dict["forward"]
    w_inv = weight_dict["inverse"]
    w_latent = weight_dict["latent"]
    w_pad = weight_dict["pad"]
    w_boundary = weight_dict["boundary"]

    optimizer = torch.optim.Adam(inn_network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    train_losses = np.zeros((epoch, 6))
    test_losses = np.zeros((epoch, 6))

    start = time.perf_counter()
    ts = datetime.today().strftime('%m-%d-%Y_%H-%M-%S')
    best_loss = float('inf')

    for i in range(epoch):
        tic_epo = time.time()
        inn_network.to(device)
        inn_network.train()
        for step, (x_data, y_data) in enumerate(dataloader):
            optimizer.zero_grad()
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            # Input side
            x = x_data[:, :x_dim]
            e = x_data[:, -e_dim:]
            # Output side
            z = y_data[:, :z_dim]
            y = y_data[:, -y_dim:]

            # Forward and backward propagations
            y_pred, log_jac_det = inn_network(x_data)
            x_inv, _ = inn_network(y_data, rev=True)

            L_pred = w_pred * nn.MSELoss()(y_pred[:, -y_dim:], y)
            L_inv  = w_inv  * utl.mmd(x_inv[:, :x_dim], x)
            L_latent = w_latent * utl.mmd(y_pred[:, :z_dim], z)
            L_pad  = w_pad  * nn.L1Loss()(x_inv[:, -e_dim:], e)
            L_boundary = w_boundary* utl.boundary_loss(x_inv[:, :x_dim], x)

            loss = L_pred + L_inv + L_latent + L_pad + L_boundary

            loss.backward()
            torch.nn.utils.clip_grad_norm_(inn_network.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        if i % 1 ==0:
            ### Training Calculation
            inn_network.to('cpu')
            inn_network.eval()
            y_pred,_ = inn_network(in_train_ts)
            x_inv, _=inn_network(out_train_ts, rev = True)

            L_pred = w_pred * nn.MSELoss()(y_pred[:,-y_dim:], out_train_ts[:,-y_dim:])
            L_inv = w_inv * utl.mmd(x_inv[1:100,:x_dim],in_train_ts[1:100,:x_dim])
            L_latent = w_latent * utl.mmd(y_pred[1:100,:z_dim],out_train_ts[1:100,:z_dim])
            L_pad = w_pad * nn.L1Loss()(x_inv[:,-e_dim:],in_train_ts[:,-e_dim:])
            L_boundary = w_boundary * utl.boundary_loss(x_inv[:,:x_dim], in_train_ts[:,:x_dim])
            loss = L_pred+L_inv+L_latent+L_pad+L_boundary

            ### Test Calculation
            y_pred_t,jac_pred = inn_network(in_test_ts)
            x_inv_t, _= inn_network(out_test_ts, rev = True)

            L_pred_t = w_pred * nn.MSELoss()(y_pred_t[:,-y_dim:], out_test_ts[:,-y_dim:])
            L_inv_t = w_inv * utl.mmd(x_inv_t[1:100,:x_dim],in_test_ts[1:100,:x_dim])
            L_latent_t = w_latent * utl.mmd(y_pred_t[1:100,:z_dim],out_test_ts[1:100,:z_dim])
            L_pad_t = w_pad * nn.L1Loss()(x_inv_t[:,-e_dim:],in_test_ts[:,-e_dim:])
            L_boundary_t = w_boundary * utl.boundary_loss(x_inv_t[:,:x_dim], in_test_ts[:,:x_dim])

            loss_t = L_pred_t+L_inv_t+L_latent_t+L_pad_t+L_boundary_t

        print("Epoch: " + str(i) + 
            " loss: " + str(loss.detach().numpy()) + 
            " Lr: " + str(optimizer.param_groups[0]['lr']) + 
            " L_pred: " + str(L_pred.detach().numpy()) + 
            " L_inv: " + str(L_inv.detach().numpy()) + 
            " L_latent: " + str(L_latent.detach().numpy()) + 
            " L_pad: " + str(L_pad.detach().numpy())  +
            " L_Boundary: " + str(L_boundary.detach().numpy()) + 
            "\n"+
            " loss_t: " + str(loss_t.detach().numpy()) + 
            " L_pred_t: " + str(L_pred_t.detach().numpy()) + 
            " L_inv_t: " + str(L_inv_t.detach().numpy()) +
            " L_latent_t: " + str(L_latent_t.detach().numpy()) + 
            " L_pad_t: " + str(L_pad_t.detach().numpy()) +
            " L_Boundary_t: " + str(L_boundary_t.detach().numpy())
        )

        train_losses[i] = [L_pred.item(), L_inv.item(), L_latent.item(), L_pad.item(), L_boundary.item(), loss.item()]
        test_losses[i] = [L_pred_t.item(), L_inv_t.item(), L_latent_t.item(), L_pad_t.item(), L_boundary_t.item(), loss_t.item()]

        print("time for epoch {0}: {1}".format(i, time.time()-tic_epo))
        print("-------------------------------------------------------------------------------------")

        if loss_t < best_loss:
            best_loss = loss_t
            torch.save(inn_network.state_dict(), save_folder + 'inn_best.pth')
            
        if (i+1) % 25 == 0:
            torch.save(inn_network.state_dict(),save_folder + 'inn_{0}epoch_{1}.pth'.format(i, ts))

    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start)) 

    torch.save(inn_network.state_dict(), save_folder + 'INN_{}.pth'.format(ts))
    
    return train_losses, test_losses

