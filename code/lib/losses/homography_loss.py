import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from ast import Compare
import numpy as np
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss


def generate_corners3d(points, dims, ry, cls_ids):
    cls_mean_size =torch.Tensor([[1.76255119    ,0.66068622   , 0.84422524   ],
                        [1.52563191462 ,1.62856739989, 3.88311640418],
                        [1.73698127    ,0.59706367   , 1.76282397   ]]).cuda()
    dim_w_mean=torch.zeros((dims.shape[0],3)).cuda()
    for i in range(dims.shape[0]):
        dim_w_mean[i]=cls_mean_size[cls_ids[i]]
        #print(dims)
    dims_=dims+dim_w_mean
    points[:,1:2] += dims_[:,0:1]/2
    corners3ds_ls=[]
    for i in range(ry.shape[0]):
        xyz_corner = torch.Tensor([[0, 0, 0, 0, -1, -1, -1, -1],
                                   [1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2],
                                   [1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2]]).cuda()
        xyz_corners=(xyz_corner*dims_[i].unsqueeze(-1))[[2,0,1]]

        ry_=ry[i]
        R = torch.Tensor([[torch.cos(ry_), 0, torch.sin(ry_)],
                        [0, 1, 0],
                        [-torch.sin(ry_), 0, torch.cos(ry_)]]).cuda()
        corners3d = torch.matmul(R, xyz_corners ).transpose(1,0)
        corners3ds_ls.append(torch.cat((torch.zeros(1,3).cuda(),corners3d[:4,:]),0))

    corners3ds=torch.cat(corners3ds_ls,0).view(-1,5,3).cuda()
    pts = corners3ds + points.unsqueeze(1)
    return pts


def compute_applyHomography(xy_src, xy_dst,batch_idxs,logger):
    gts = torch.unique(batch_idxs, sorted=True).tolist()
    A=[]
    H_ls=[]
    for i in gts:
        dst=xy_dst[batch_idxs==i].view(-1,2)
        src=xy_src[batch_idxs==i].view(-1,2)
        H=homography_matrix(src, dst,logger)
        H_ls.append(H)
        A.append(applyHomography(src, H).view(-1,5,2))

    A = torch.cat(A,0).view(-1,2).cuda()
    #print("H_ls",H_ls)
    return A

def homography_matrix(X, Y,logger):
    N = list(X.size())[0]
    A = torch.zeros(2*N, 9, dtype=torch.float32).cuda()
    A[0::2, 0:2] = X
    A[0::2, 2:3] = torch.ones(N, 1).cuda()
    A[1::2, 3:5] = X
    A[1::2, 5:6] = torch.ones(N, 1).cuda()
    A[0::2, 6:8] = X
    A[1::2, 6:8] = X
    A[:, 8:9] = torch.ones(2*N, 1).cuda()
    Y_vec = torch.reshape(Y, (2*N, 1))
    A[:, 6:7] = -A[:, 6:7] * Y_vec
    A[:, 7:8] = -A[:, 7:8] * Y_vec
    A[:, 8:9] = -A[:, 8:9] * Y_vec
    
    _, _, V = torch.linalg.svd(A)
    H_torch= torch.reshape(V[-1], (3, 3))
    if torch.any(H_torch.isnan()):
        log_str="H is NaN NaN NaN !!!!!"
        logger.info(log_str)
    #H_torch = H_torch / H_torch[2, 2]
    return H_torch
    
def homography_matrix1(xy_src, xy_dst,logger):#Nx2
    src = torch.cat((xy_src,torch.ones(xy_src.shape[0],1).cuda()),1).float()
    dst = torch.cat((xy_dst,torch.ones(xy_dst.shape[0],1).cuda()),1).float()
    n_points = src.shape[0]
    Tr=torch.Tensor([[[0,0,0],
                      [0,0,-1],
                      [0,1,0]],
                     [[0,0,1],
                      [0,0,0],
                      [-1,0,0]],
                     [[0,-1,0],
                      [1,0,0],
                      [0,0,0]]]).cuda()           
    A = []
    for i in range(n_points):#x1,x2    y1,y2
        A.append((Tr@dst[i].view(-1,1)@src[i].view(1,3)).view(3,9))
    A = torch.cat(A,0).float().cuda()
    _, _, V_transpose = torch.linalg.svd(A)
    H_torch= torch.reshape(V_transpose[-1], (3, 3))

    return H_torch 

def affine_transform(pt, t):
    pts = torch.cat((pt,torch.ones(pt.shape[0],1).cuda()),1).view(-1,3,1)
    new_pt=(t.type(torch.float32)@pts).view(-1,2)
    return new_pt
    

def applyHomography(xy, H):#BxNx2
    xyz = torch.cat((xy,torch.ones(xy.shape[0],1).cuda()),1)
    new_xyz = xyz@(H.transpose(1,0))
    _xyz = new_xyz /(new_xyz[:,-1:]+1e-10)#.repeat(1,3)
    return _xyz[:,[0,1]]
class Homography_Loss(nn.Module):
    def __init__(self,epoch,calibs,info,logger):
        super().__init__()
        self.epoch = epoch
        self.calibs=calibs
        self.info=info
        self.logger=logger
        
    def forward(self,outputs,targets):
        homography_Loss = self.homographyLoss(outputs,targets)
        return homography_Loss
    
    
    def project_image_to_rect(self,uv,depth,P2):
        c_u = P2[:,0, 2].view(-1,1)
        c_v = P2[:,1, 2].view(-1,1)
        f_u = P2[:,0, 0].view(-1,1)
        f_v = P2[:,1, 1].view(-1,1)
        t_x = P2[:,0, 3].view(-1,1) / (-f_u)
        t_y = P2[:,1, 3].view(-1,1) / (-f_v)
        c=torch.cat((c_u,c_v),1).cuda()
        f=torch.cat((f_u,f_v),1).cuda()
        t=torch.cat((t_x,t_y),1).cuda()
        Tr=uv-c
        pts3d = (Tr*depth)/f + t
        pts_3d_rect = torch.cat((pts3d,depth),1).cuda()
        return pts_3d_rect
        

    def homographyLoss(self,outputs,targets):
        #decode batch
        reg_mask_gt = targets["mask_2d"]
        targets_bbox_points_center = targets["center_3dto2d"]
#         flip_status = self.info["flip_status"].repeat(1,50).cuda()
        flip_mask = self.info["flip_status"].view(-1)
        crop_mask = self.info["crop_status"].view(-1)
        
        batch=targets_bbox_points_center.shape[0]
#         flatten_reg_mask_gt = (reg_mask_gt.view(-1).bool()) & (~(flip_status.reshape(-1).bool()))
        flatten_reg_mask_gt = reg_mask_gt.view(-1).bool()
        batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1).to(reg_mask_gt.device)
        batch_idxs = batch_idxs[flatten_reg_mask_gt] 
        
        #get coordinate center object in img coordinate
        valid_targets_bbox_points_center=targets_bbox_points_center.view(-1, 2)[flatten_reg_mask_gt]
        
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        calib =torch.zeros(valid_targets_bbox_points_center.shape[0],3,4).type(torch.float).cuda()
        ratio =torch.zeros(valid_targets_bbox_points_center.shape[0],2).type(torch.float).cuda()
        inv=torch.zeros(valid_targets_bbox_points_center.shape[0],2,3,dtype=torch.float64).cuda()
        flip_mask_points=torch.zeros_like(batch_idxs).cuda()
        for gt in gts:
            inv[batch_idxs == gt]  =self.info['trans_inv'][gt].cuda()
            calib[batch_idxs == gt]=self.calibs[gt].type(torch.float)
            if flip_mask[gt] == True:
                flip_mask_points[batch_idxs == gt] = 1
                
        flip_mask_points=flip_mask_points.bool().view(-1)
        
#         print("calib",calib)
#         print("flip_mask",flip_mask)
#         print("crop_mask",crop_mask)
        if torch.all(flip_mask_points):
            print("No img flip !!!")
            return torch.tensor(0).float().cuda()
        
        ##################################################
        cls_ids =targets['cls_ids'].view(-1, 1)[flatten_reg_mask_gt]  
        #print(cls_ids)
        ry=targets['ry'].view(-1, 1)[flatten_reg_mask_gt]
        size_3d=targets["size_3d"].view(-1, 3)[flatten_reg_mask_gt]
        Pgt=targets['Pgt'].view(-1, 5,2)[flatten_reg_mask_gt]#get points bottom in img coordinate
        Qgt=targets['Qgt'].view(-1, 5,3)[flatten_reg_mask_gt]
        Qgt_= Qgt[:,:,0::2].view(-1,2)
        
        #print("batch ids",batch_idxs)
        Replicated_Losses={'centerGT_depth_PRED':(True,False),'centerPRED_depth_GT':(False,True),'centerPRED_depth_PRED':(False,False)}#(use_offset3d_target,use_depth_target)
        #Replicated_Losses={'centerGT_depth_PRED':(False,True)}#(use_offset3d_target,use_depth_target)
        count_mask=[]
        loss_term={}
        for name_losss,replicated in Replicated_Losses.items():
            use_offset3d_target,use_depth_target=replicated
            if use_offset3d_target:
                offset3d=targets['offset_3d'].view(-1, 2)[flatten_reg_mask_gt]
            else:
                offset3d=outputs['offset_3d']
            
            if use_depth_target:
                depths=targets['depth'].view(-1, 1)[flatten_reg_mask_gt]

            else:
                depths=outputs['depth'][:,0:1]

            proj_center3d=affine_transform((valid_targets_bbox_points_center + offset3d)*4,inv)
            points=self.project_image_to_rect(proj_center3d ,depths,calib)# center 3d-grad
            Qpred = generate_corners3d(points, size_3d,ry,cls_ids)#Nx5x3
            Reproject_Pgt =compute_applyHomography(Pgt,Qpred[:,:,0::2],batch_idxs,self.logger)
            x_mask=((Reproject_Pgt[:,0:1]>-40)&(Reproject_Pgt[:,0:1]<40))
            z_mask=((Reproject_Pgt[:,1:]>0)&(Reproject_Pgt[:,1:]<80))
            mask=(x_mask & z_mask).view(-1)
            homography_loss_mask=F.smooth_l1_loss(Reproject_Pgt[mask], Qgt_[mask] ,reduction="mean")
            reproject_loss=F.smooth_l1_loss(Qpred[:,:,0::2].view(-1,2)[~mask],Qgt_[~mask],reduction="mean")
            loss_term[name_losss]=homography_loss_mask + 0.5*reproject_loss           
 

        total_lossHomo=torch.tensor(0).float().cuda()
        count_loss=0
        for _ ,vLoss in loss_term.items():
            total_lossHomo+=vLoss
            count_loss+=1
        total_lossHomo_mask = total_lossHomo / count_loss if count_loss > 0 else total_lossHomo
        return total_lossHomo_mask