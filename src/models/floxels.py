
"""
# Created: 2024-07-27 11:40
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * TODO
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Floxels to our codebase implementation.
# one more package need install: pip install FastGeodis==1.0.4 --no-build-isolation
"""

import dztimer, torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
from .basic.opt_module import VoxelGrid, EarlyStopping, DT
from .basic import wrap_batch_pcs

def cluster(pc, min_samples=4, eps=0.5):
    pc_np = pc.detach().to("cpu").numpy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(pc_np)

    clusters = dbscan.labels_
    clusters = torch.Tensor(clusters).int().to(pc.device)
    return clusters


def cluster_loss(flow, clusters):
    clusters_unique, indices = torch.unique(clusters, return_inverse=True)
    cluster_sums = torch.zeros((clusters_unique.shape[0], 3), device=flow.device)
    cluster_counts = torch.zeros((clusters_unique.shape[0], 1), device=flow.device)
    
    cluster_sums.scatter_add_(0, indices[:, None].expand(-1, 3), flow)
    cluster_counts.scatter_add_(0, indices[:, None], torch.ones_like(flow[:, :1]))

    cluster_means = cluster_sums / cluster_counts

    mean_expanded = cluster_means[indices]
    l2_loss_per_point = (flow - mean_expanded).norm(dim=1)
    return l2_loss_per_point

class Floxels(nn.Module):
    # default parameters from Sec 3.1
    def __init__(self, grid_factor=10., itr_num=500, lr=0.05, min_delta=0.01, early_patience=250,
                 verbose=False, point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3], voxel_size=0.5, num_frames=3, flow_num=1,
                 cluster_weight=0.6):
        super().__init__()

        self.grid_factor = grid_factor # grid cell size=1/grid_factor.
        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.lr = lr
        self.early_patience = early_patience
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        # FIXME later, as I messed up frames num here I think....
        self.num_frames = num_frames
        self.flow_num = flow_num

        self.cluster_weight = cluster_weight

        self.timer = dztimer.Timing()
        self.timer.start("Floxels Model Inference")
        print(f"\n---LOG [model]: Floxels setup total frames: {num_frames+flow_num-1}, itr_num: {itr_num}, lr: {lr}, early_patience: {early_patience} with min_delta: {min_delta}.")
            
    def optimize(self, dict2loss):
        device = dict2loss['pc0'].device

        self.timer[1].start("Network Initialization")
        net = VoxelGrid(point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size)
        net = net.to(device)
        net.train()
        self.timer[1].stop()

        self.timer[2].start("DT Map Building")
        pc0 = dict2loss['pc0']
        pc0_min = torch.min(pc0.squeeze(0), 0)[0]
        pc0_max = torch.max(pc0.squeeze(0), 0)[0]        
        dict2loss.pop('pc0', None)

        dt_dict = {}
        for key_ in dict2loss.keys():
            pc_aim = dict2loss[key_]
            pc2_min = torch.min(pc_aim.squeeze(0), 0)[0]
            pc2_max = torch.max(pc_aim.squeeze(0), 0)[0]
            
            xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc0_min<pc2_min, pc0_min, pc2_min) * self.grid_factor-1) / self.grid_factor
            xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc0_max>pc2_max, pc0_max, pc2_max)* self.grid_factor+1) / self.grid_factor
            # print('xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}'.format(xmin_int, xmax_int, ymin_int, ymax_int, zmin_int, zmax_int))
            
            # NOTE: build DT map
            dt_dict[key_] = DT(pc_aim.clone().squeeze(0).to(device), (xmin_int, ymin_int, zmin_int), (xmax_int, ymax_int, zmax_int), self.grid_factor, device)
        self.timer[2].stop()

        self.timer[3].start("Clustering")
        clusters = cluster(pc0)
        self.timer[3].stop()
        params = net.parameters()

        best_forward = {'loss': torch.inf}

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
        early_stopping = EarlyStopping(patience=self.early_patience, min_delta=self.min_delta)
        # Define lambda_gamma linear scaling parameters

        # FIXME: shall we move this to paramters config.
        epochs = np.array([0, 100])
        weight_factor = np.array([0.1, 0.01])  # Changed from 0.01 to 0.01 as per author's spec

        frame_keys = sorted([key for key in dict2loss.keys() if key.startswith('pch')], reverse=False)
        frame_future_keys = sorted([key for key in dict2loss.keys() if (key.startswith('pc') and not key.startswith('pch'))], reverse=False)

        self.timer[4].start("Optimization")
        for itr_ in range(self.iteration_num):
            optimizer.zero_grad()
            self.timer[4][1].start("Network Time")
            forward_flow = net(pc0)
            self.timer[4][1].stop()

            self.timer[4][2].start("loss")
            loss = 0

            # loss = dt_dict['pc1'].torch_bilinear_distance((pc0 + forward_flow).squeeze(0), truncate_dist=5.0).mean()
            for time_index, frame_key in enumerate(frame_future_keys):
                dt = dt_dict[frame_key]
                pesudo_pc = pc0 + (time_index + 1) * forward_flow
                # since time_index starts from 0
                loss += dt.torch_bilinear_distance(pesudo_pc.squeeze(0), truncate_dist=5.0).mean() * pow(1/(time_index+1), 2)

            # (0, 'pch1s'), (1, 'pch2s'), ...
            for time_index, frame_key in enumerate(reversed(frame_keys)):
                dt = dt_dict[frame_key]
                pesudo_pc = pc0 - (time_index + 1) * forward_flow
                # since time_index starts from 0
                loss += dt.torch_bilinear_distance(pesudo_pc.squeeze(0), truncate_dist=5.0).mean() * pow(1/(time_index+1), 2)
            self.timer[4][2].stop()

            # FIXME: lambda_d weight didn't specify before. 
            # loss *= lambda_d # ? ask Hanqiu later
            total_num_frames = self.num_frames + self.flow_num - 1
            cl_loss = cluster_loss(
                forward_flow[clusters >= 0],
                clusters[clusters >= 0]
            )
            loss += cl_loss.mean() * self.cluster_weight * (total_num_frames - 1)

            loss_flownorm = torch.norm(forward_flow, p=2) * (total_num_frames - 1)
            lambda_gamma = np.interp(itr_, epochs, weight_factor)

            # FIXME: check with author about this weight. since it's not work.... 
            loss += lambda_gamma * loss_flownorm

            if loss <= best_forward['loss']:
                best_forward['loss'] = loss.item()
                best_forward['flow'] = forward_flow

            if early_stopping.step(loss) and 'flow' in best_forward: # at least one step
                break

            self.timer[4][3].start("Loss Backward")
            loss.backward()
            self.timer[4][3].stop()

            self.timer[4][4].start("Optimizer Step")
            optimizer.step()
            scheduler.step()
            self.timer[4][4].stop()
        self.timer[4].stop()

        if self.verbose:
            self.timer.print(random_colors=True, bold=True)

        return best_forward
    
    def range_limit_(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        return pc[mask], mask
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])
        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs(batch, num_frames=self.num_frames)
        self.timer[0].stop()

        batch_final_flow = []
        for batch_id in range(batch_sizes):
            pc0 = pcs_dict["pc0s"][batch_id,...]
            pc1 = pcs_dict["pc1s"][batch_id,...]
            selected_pc0, rm0 = self.range_limit_(pc0)
            selected_pc1 = self.range_limit_(pc1)[0]
            pchs, pcs = {}, {}
            for i in range(1, self.num_frames - 1):
                pchs[f'pch{i}'] = self.range_limit_(pcs_dict[f'pch{i}s'][batch_id,...])[0]

            for i in range(2, self.flow_num + 1):
                pcs[f'pc{i}'] = self.range_limit_(pcs_dict[f'pc{i}s'][batch_id,...])[0]

            # since pl in val and test mode will disable_grad.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    dict2loss = {
                        'pc0': selected_pc0.clone().detach(), #.requires_grad_(True),
                        'pc1': selected_pc1.clone().detach(), #.requires_grad_(True)
                    }
                    dict2loss.update(pchs)
                    dict2loss.update(pcs)
                    model_res = self.optimize(dict2loss)
            
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = model_res["flow"].clone().detach().requires_grad_(False)
            batch_final_flow.append(final_flow)

        res_dict = {"flow": batch_final_flow,
                    "loss": [model_res["loss"]],
                    "pose_flow": pcs_dict["pose_flows"]  # using identity here
                    }
        
        return res_dict
