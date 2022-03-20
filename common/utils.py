import os, torch, glob, subprocess, shutil
from PIL.Image import merge
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from common.metrics import run_evaluation


def get_transform(is_train):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    return transform



class PostProcessor():
    def __init__(self, args):
        self.exp_path = args.exp_path
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.groundtruth = []
        self.prediction = []
        self.groundtruthfile = f'{self.save_path}/gt.csv.rank.{args.rank}'
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'
        self.current_seg = None
        self.segoutput = []
    
    def update(self, outputs, targets):
        #postprocess outputs of one minibatch
        segid = targets[0][0] + ':' + str(targets[-1].item())
        if self.current_seg is None:
            self.current_seg = segid
            self.segoutput.append((outputs, targets))
        elif segid == self.current_seg:
            self.segoutput.append((outputs, targets))
        else:
            self._merge_output()
            # update segments
            self.current_seg = segid
            self.segoutput = [(outputs, targets)]
    
    def _merge_output(self):
        # merge and save segments
        pred = torch.cat([p[0] for p in self.segoutput], dim=0)
        pred = F.softmax(pred.mean(0), dim=-1)
        start = min([p[1][-3].item() for p in self.segoutput])
        end = max([p[1][-2].item() for p in self.segoutput])
        uid, idx = self.current_seg.split(':')
        label = self.segoutput[0][1][2].item()
        self.groundtruth.append([uid, idx, start, end, label])
        self.prediction.append([uid, idx, start, end, 1, pred[1].item()])

    def save(self):
        if len(self.segoutput) != 0:
            self._merge_output()
        if os.path.exists(self.groundtruthfile):
            os.remove(self.groundtruthfile)
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        gt_df = pd.DataFrame(self.groundtruth)
        gt_df.to_csv(self.groundtruthfile, index=False, header=None)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
    
    def get_mAP(self):
        #merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)

        gt_file = f'{merge_path}/gt.csv'
        if os.path.exists(gt_file):
            os.remove(gt_file)
        gts = glob.glob(f'{self.save_path}/gt.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(gts), gt_file)
        subprocess.call(cmd, shell=True)
        pred_file = f'{merge_path}/pred.csv'
        if os.path.exists(pred_file):
            os.remove(pred_file)
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)
        return run_evaluation(gt_file, pred_file)
    
    
class test_PostProcessor():
    def __init__(self, args):
        self.exp_path = args.exp_path
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
#         self.groundtruth = []
        self.prediction = []
#         self.groundtruthfile = f'{self.save_path}/gt.csv.rank.{args.rank}'
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'
        self.current_seg = None
        self.current_fid_list = []
        self.segoutput = []
    
    def update(self, outputs, sid, fid_list):
        #postprocess outputs of one minibatch
        segid = sid[0]
        if self.current_seg is None:
            self.current_seg = segid
            self.current_fid_list = fid_list
            self.segoutput.append(outputs)
        elif segid == self.current_seg:
            self.segoutput.append(outputs)
        else:
            self._merge_output()
            # update segments
            self.current_seg = segid
            self.current_fid_list = fid_list
            self.segoutput = [outputs]
    
    def _merge_output(self):
        # merge and save segments
        pred = torch.cat([p for p in self.segoutput], dim=0)
        pred = F.softmax(pred.mean(0), dim=-1)
        for fid in self.current_fid_list:
            self.prediction.append([self.current_seg, fid.item(), 1, pred[1].item()])

    def save(self):
        if len(self.segoutput) != 0:
            self._merge_output()
#         if os.path.exists(self.groundtruthfile):
#             os.remove(self.groundtruthfile)
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
#         gt_df = pd.DataFrame(self.groundtruth)
#         gt_df.to_csv(self.groundtruthfile, index=False, header=None)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
    
    def mkfile(self):
        #merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)

#         gt_file = f'{merge_path}/gt.csv'
#         if os.path.exists(gt_file):
#             os.remove(gt_file)
#         gts = glob.glob(f'{self.save_path}/gt.csv.rank.*')
#         cmd = 'cat {} > {}'.format(' '.join(gts), gt_file)
#         subprocess.call(cmd, shell=True)
        pred_file = f'{merge_path}/pred.csv'
        if os.path.exists(pred_file):
            os.remove(pred_file)
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)
#         return run_evaluation(gt_file, pred_file)
    
    


def save_checkpoint(state, save_path, is_best=False, is_dist=False):

    save_path = f'{save_path}/checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    epoch = state['epoch']
    filename= f'{save_path}/epoch_{epoch}.pth'
    torch.save(state, filename)
    
    if is_best:
        if os.path.exists(f'{save_path}/best.pth'):
            os.remove(f'{save_path}/best.pth')
        torch.save(state, f'{save_path}/best.pth')


def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


def collate_fn(batch):
    min_frames = min([b[0].shape[0] for b in batch])
    min_duration = min([b[1].shape[0] for b in batch])
    video = torch.cat([b[0][:min_frames, ...].unsqueeze(0) for b in batch], dim=0)
    audio = torch.cat([b[1][:min_duration].unsqueeze(0) for b in batch], dim=0)
    target = torch.cat([b[2] for b in batch])
    return video, audio, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count