import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import GupnetLoss,Hierarchical_Task_Learning
# from lib.losses.homography_loss import Homography_Loss 
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.losses.homography_loss import Homography_Loss 
import torch
torch.autograd.set_detect_anomaly(True)


import math 
# def weightHomo_schedule(t,maxT):
#     return 0.2/((math.e**((1-t/maxT)**2))*math.sqrt(math.e)/math.sqrt(2))

# def weightHomo_schedule(t,maxT):
#     return 0.2/((math.e**(((1-t/maxT)**2)*math.sqrt(math.e))))
def weightHomo_schedule0(t,maxT):
    return 0.1/(math.e**((1-t/maxT)**2))

def weightHomo_schedule(t,maxT):
    return 0.2/(math.e**((1-t/maxT)**2))
class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.loss_weights_init={}
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.past_losses=[]
        self.init_diff = None
        
        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch , self.past_losses, self.init_diff= load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1
            print(self.init_diff)
        self.model = torch.nn.DataParallel(model).to(self.device)
        
    def train(self):
        start_epoch = self.epoch
#         if start_epoch > 0 and len(self.past_losses) > 0:
#             # old past losses
#             past_losses = self.past_losses[-1].view(-1)
#             ei_loss = {}
#             loss_term = ['seg_loss', 'offset2d_loss', 'size2d_loss', 'depth_loss', 'offset3d_loss', 'size3d_loss', 'heading_loss']
#             assert len(loss_term) == len(past_losses)
#             for i in range(len(loss_term)):
#                 ei_loss[loss_term[i]] = past_losses[i].detach()

#             print("ei_loss:",ei_loss)

#         else:
#             ei_loss = self.compute_e0_loss()
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss,5,self.past_losses, self.init_diff)
        print("past_losses:",self.past_losses)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights,past_losses = loss_weightor.compute_weight(ei_loss,self.epoch)
            print("past_losses1:",past_losses)
            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)                     
            ei_loss = self.train_one_epoch(loss_weights)
            self.past_losses=past_losses
            self.epoch += 1
            
            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:  
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
            

            # save trained model
            if (self.epoch % self.cfg_train['save_frequency']) == 0:
                if (self.init_diff == None):
                    self.init_diff = loss_weightor.get_init_diff()
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch,self.past_losses, self.init_diff), ckpt_name, self.logger)


            if (self.epoch % self.cfg_train['eval_frequency']) == 0:
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.eval_one_epoch()

            
        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        flag_test_homo = False
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                criterion = GupnetLoss(self.epoch)
                
                outputs = self.model(inputs,coord_ranges,calibs,targets)
                if not flag_test_homo:
                    homog_loss = Homography_Loss(self.epoch,calibs,info,self.logger)
                    homog_loss_value=homog_loss(outputs,targets)
                    print(homog_loss_value)
                    flag_test_homo = True
                    
                _, loss_terms = criterion(outputs, targets)
                # loss_terms['homography_loss']=homog_loss_value
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict        
    def train_one_epoch(self,loss_weights=None):
        self.model.train()
        disp_dict = {}
        stat_dict = {}
        count=0
        total_homo=torch.tensor(0).float().cuda()
        weight_homo=0
        #compute weight for homo loss
#         if self.epoch >=80 and self.epoch <=120:
#             weight_homo=weightHomo_schedule0(self.epoch,120)
#         else:
#             weight_homo=0.1
        weight_homo=0.2
        log_weight_component = 'weight_homo_component : %.4f'%(weight_homo)    
        self.logger.info(log_weight_component)
        count_homo=0
        for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = GupnetLoss(self.epoch)
            # homog_loss = Homography_Loss(self.epoch,calibs)
            outputs = self.model(inputs,coord_ranges,calibs,targets)
            # homog_loss_value=homog_loss(outputs,targets)
            total_loss, loss_terms = criterion(outputs, targets)
            homo=False
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    
                    if key =='homography_loss' :
                        if self.epoch >=70 and self.epoch <= 140:
                            homog_loss = Homography_Loss(self.epoch,calibs,info,self.logger)
                            homog_loss_value=homog_loss(outputs,targets)
                            total_loss += weight_homo *  loss_weights[key].detach() * homog_loss_value
                            total_homo += homog_loss_value
                            homo=True
                    else:
                        total_loss += loss_weights[key].detach()*loss_terms[key]
                            
            total_loss.backward()
            self.optimizer.step()
            
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key] 
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]   
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                    
                if homo:
                    
                    log_str += ' %s:%.4f,' %("homo_loss", float(homog_loss_value) )
                self.logger.info(log_str)
        #print("use homo : %d/464"%(count))
                
        print("count_filter loss %d"%(count_homo))        
        log_str_avg = 'Avg_Loss_Epoch[%04d/%04d]' % (self.epoch+1, self.cfg_train['max_epoch'])       
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
            log_str_avg += ' %s:%.4f,' %(key,stat_dict[key] )
        #total_homo   = total_homo/count_homo if count_loss > 0 else torch.tensor(0).float()
        log_str_avg += ' %s:%.4f,' %("homo_loss",total_homo/trained_batch)    
        self.logger.info(log_str_avg)
        
        return stat_dict    
    def eval_one_epoch(self):
        self.model.eval()
        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device) 
                coord_ranges = coord_ranges.to(self.device)
    
                # the outputs of centernet
                outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val')

                dets = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()
                
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold = self.cfg_test['threshold'])                 
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        self.save_results(results)
           
                
    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()