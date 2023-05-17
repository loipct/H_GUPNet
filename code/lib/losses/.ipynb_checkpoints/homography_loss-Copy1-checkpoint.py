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

num_heading_bin = 12
def _gather_feat(feat, ind, mask=None):

    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat
def extract_target_from_tensor(target, mask):
    return target[mask]

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C

def get_heading_angle(heading):
    heading_bin, heading_res = heading[:,0:12], heading[:,12:24]

    cls = torch.argmax(heading_bin,-1).view(-1,1)
    res = heading_res.gather(1, cls)
    #print("res : ",res.shape)
    return class2angle(cls, res, to_label_format=True)

def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * math.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format :
        angle[angle > math.pi] -=  2 * math.pi
    return angle


def alpha2ry(alpha,u,P2):
    u=u.view(-1,1)
    cu,fu=P2[:,0,2].view(-1,1),P2[:,0,0].view(-1,1)
    ry = alpha + torch.atan2(u - cu, fu)
    return ry
  
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

def uncertainty_generate_corners3d(points,points_log_std, dims, ry, cls_ids):
    cls_mean_size =torch.Tensor([[1.76255119    ,0.66068622   , 0.84422524   ],
                        [1.52563191462 ,1.62856739989, 3.88311640418],
                        [1.73698127    ,0.59706367   , 1.76282397   ]]).cuda()
    dim_w_mean=torch.zeros((dims.shape[0],3)).cuda()
    for i in range(dims.shape[0]):
        dim_w_mean[i]=cls_mean_size[cls_ids[i]]
    dims_=dims+dim_w_mean# size 3D 
    points[:,1:2] += dims_[:,0:1]/2 # h,w,l - bottom_center   
    
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
    pts_log_std = torch.zeros_like(corners3ds).cuda()
    pts_log_std += points_log_std.unsqueeze(1)
    return pts, pts_log_std

# def compute_applyHomography(xy_src, xy_dst,batch_idxs):
#     gts = torch.unique(batch_idxs, sorted=True).tolist()
#     A=[]
#     for i in gts:
#         dst=xy_dst[batch_idxs==i].view(-1,2)
#         src=xy_src[batch_idxs==i].view(-1,2)
#         H=homography_matrix(src, dst)
#         A.append(applyHomography(src, H).view(-1,5,2))
#     A = torch.cat(A,0).view(-1,2).cuda()
#     return A
def compute_applyHomography(xy_src, xy_dst,batch_idxs,logger):
    gts = torch.unique(batch_idxs, sorted=True).tolist()
    A=[]
    #H_ls=[]
    for i in gts:
        dst=xy_dst[batch_idxs==i].view(-1,2)
        src=xy_src[batch_idxs==i].view(-1,2)
        H=homography_matrix(src, dst,logger)
        #H_ls.append(H)
        A.append(applyHomography(src, H).view(-1,5,2))

    A = torch.cat(A,0).view(-1,2).cuda()
    #print(H_ls)
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
        
    def uncertainty_project_image_to_rect(self,uv,depth,P2):
        depth_input, depth_log_variance = depth[:, 0:1], depth[:, 1:2]
        c_u = P2[:,0, 2].view(-1,1)
        c_v = P2[:,1, 2].view(-1,1)
        f_u = P2[:,0, 0].view(-1,1)
        f_v = P2[:,1, 1].view(-1,1)
        t_x = P2[:,0, 3].view(-1,1) / (-f_u)
        t_y = P2[:,1, 3].view(-1,1) / (-f_v)
        c=torch.cat((c_u,c_v),1)
        f=torch.cat((f_u,f_v),1)
        t=torch.cat((t_x,t_y),1)
        Tr=uv-c
        pts3d = (Tr*depth_input)/f + t
        pts3d_log_std=(torch.abs(Tr/f).log() + depth_log_variance)
        pts_3d_rect = torch.cat((pts3d,depth_input),1) 
        pts3d_log_std=torch.cat((pts3d_log_std,depth_log_variance),1) 
        return pts_3d_rect,pts3d_log_std

    def homographyLoss(self,outputs,targets):
        #decode batch
        reg_mask_gt = targets["mask_2d"]
        targets_bbox_points_center = targets["center_3dto2d"]  
        batch=targets_bbox_points_center.shape[0]
        flatten_reg_mask_gt = reg_mask_gt.view(-1).bool()
        batch_idxs = torch.arange(batch).view(-1, 1).expand_as(reg_mask_gt).reshape(-1).to(reg_mask_gt.device)
        batch_idxs = batch_idxs[flatten_reg_mask_gt] 
        #get coordinate center object in img coordinate
        valid_targets_bbox_points_center=targets_bbox_points_center.view(-1, 2)[flatten_reg_mask_gt]
        
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        calib =torch.zeros(valid_targets_bbox_points_center.shape[0],3,4).type(torch.float).cuda()
        ratio =torch.zeros(valid_targets_bbox_points_center.shape[0],2).type(torch.float).cuda()
        inv=torch.zeros(valid_targets_bbox_points_center.shape[0],2,3,dtype=torch.float64).cuda()
        
        for gt in gts:
            inv[batch_idxs == gt]  =self.info['trans_inv'][gt].cuda()
            calib[batch_idxs == gt]=self.calibs[gt].type(torch.float)
#             ratio[batch_idxs == gt]=self.info['bbox_downsample_ratio'][gt].type(torch.float).cuda()
        ##################################################
        cls_ids =targets['cls_ids'].view(-1, 1)[flatten_reg_mask_gt]  
        ry=targets['ry'].view(-1, 1)[flatten_reg_mask_gt]
        size_3d=targets["size_3d"].view(-1, 3)[flatten_reg_mask_gt]
        Pgt=targets['Pgt'].view(-1, 5,2)[flatten_reg_mask_gt]#get points bottom in img coordinate
        Qgt=targets['Qgt'].view(-1, 5,3)[flatten_reg_mask_gt]
        Qgt_= Qgt[:,:,0::2].view(-1,2)
        Replicated_Losses={'centerGT_depth_PRED':(True,False),'centerPRED_depth_GT':(False,True),'centerPRED_depth_PRED':(False,False)}#(use_offset3d_target,use_depth_target)
        #Replicated_Losses={'centerGT_depth_PRED':(True,False),'centerPRED_depth_GT':(False,True)}#(use_offset3d_target,use_depth_target)
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
            Reproject_Pgt =compute_applyHomography(Pgt,Qpred[:,:,0::2] ,batch_idxs,self.logger)
            Qpred_=Qpred[:,:,0::2].view(-1,2)
            x_mask=((Reproject_Pgt[:,0:1]>-45)&(Reproject_Pgt[:,0:1]<45))
            z_mask=((Reproject_Pgt[:,1:]>0)&(Reproject_Pgt[:,1:]<80))
            mask=(x_mask & z_mask).reshape(-1)
            Reproject_Pgt_mask=Reproject_Pgt[mask]
            Qgt_mask         =Qgt_[mask]
            count_mask.append(Reproject_Pgt_mask.shape[0])
            homography_loss_mask=F.smooth_l1_loss(Reproject_Pgt_mask, Qgt_mask ,reduction="mean")#+F.smooth_l1_loss(Qpred_[~mask], Qgt_[~mask] ,reduction="mean")
            #homography_loss_mask=F.smooth_l1_loss(Reproject_Pgt, Qgt_ ,reduction="mean")
#             homography_loss=F.smooth_l1_loss(Reproject_Pgt, Qgt_ ,reduction="mean")*1/5
        
#             loss_term[name_losss]=(homography_loss_mask,homography_loss)
            loss_term[name_losss]=homography_loss_mask
        #print("valid points %d/%d"%(int(Qgt_mask.shape[0]),int(Reproject_Pgt.shape[0])))
        #print(loss_term)
        #preprocess Loss 
        count_mask.append(Qgt_.shape[0])
        #print("count filter points : %d/%d/%d/ %d:"%(int(count_mask[0]),int(count_mask[1]),int(count_mask[2]),int(count_mask[3])))
        total_lossHomo_mask=torch.tensor(0).float().cuda()
        #total_lossHomo=torch.zeros(1).cuda()
        count_loss=0
        for _ ,vLoss in loss_term.items():
            total_lossHomo_mask+=vLoss
            #total_lossHomo+=vLoss[1]
            count_loss+=1
#         total_lossHomo = total_lossHomo / count_loss if count_loss >0 else total_lossHomo
        total_lossHomo_mask = total_lossHomo_mask / count_loss if count_loss >0 else total_lossHomo_mask
        #print("loss_mask : %f"%(float(total_lossHomo_mask)))
        #print("homo_value :%f / %f , count_component : %d/3"%(total_lossHomo_mask,total_lossHomo,count_loss))
        return total_lossHomo_mask