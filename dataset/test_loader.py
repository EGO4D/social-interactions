import os, cv2, json, glob, logging, soundfile
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import hashlib
from tqdm import tqdm


logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def get_md5(s):
    md5hash = hashlib.md5(s.encode(encoding='UTF-8'))
    md5 = md5hash.hexdigest()
    return md5



def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples



# def make_dataset(data_path):

#     logger.info('load: ' + data_path)
#     vid_path_list = []
#     aud_path_list = []
#     sid_list = []
#     sid2fid_list = {}
    
#     for vid_id in tqdm(os.listdir(data_path)):
#         for seg_id in os.listdir(os.path.join(data_path, vid_id)):
#             sid = vid_id + ':' + seg_id
#             sid_list.append(sid)
#             sid2fid_list[sid] = []
#             aud_path_list.append(os.path.join(data_path, vid_id, seg_id, 'audio', 'aud.wav'))
#             if os.path.exists(os.path.join(data_path, vid_id, seg_id, 'face')):
#                 vid_path_list.append(os.path.join(data_path, vid_id, seg_id, 'face'))
#                 for img_path in os.listdir(os.path.join(data_path, vid_id, seg_id, 'face')):
#                     fid = img_path.split('_')[1].split('.')[0]
#                     sid2fid_list[sid].append(fid)
#             else:
#                 vid_path_list.append('None')
#     return vid_path_list, aud_path_list, sid_list, sid2fid_list


def make_dataset(data_path, seg_info, min_frames=15, max_frames=150):

    logger.info('load: ' + data_path)
    segments = []
    
    for sid in tqdm(os.listdir(data_path)):
        aud_path = os.path.join(data_path, sid, 'audio', 'aud.wav')
        if os.path.exists(os.path.join(data_path, sid, 'face')):
            vid_path = os.path.join(data_path, sid, 'face')
        else:
            vid_path = 'None'
        seg_length = seg_info[sid]['frame_num']
        start_frame = 0
        end_frame = seg_length-1
        if seg_length > max_frames:
            it = int(seg_length / max_frames)
            for i in range(it):
                sub_start = start_frame + i*max_frames
                sub_end = min(end_frame, sub_start + max_frames)
                sub_length = sub_end - sub_start + 1
                if sub_length < min_frames:
                    continue
                segments.append([sid, aud_path, vid_path, sub_start, sub_end])
        else:
            segments.append([sid, aud_path, vid_path, start_frame, end_frame])
    return segments



class test_ImagerLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, seg_info, transform=None):
        
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'image path not exist'
        
        self.seg_info = json.load(open(seg_info))
        
        self.segments = make_dataset(data_path, self.seg_info)
        self.transform = transform

    def __getitem__(self, indices):
        source_audio = self._get_audio(indices)
        source_video = self._get_video(indices)
        sid = self.segments[indices][0]
        return source_video, source_audio, sid, self.seg_info[sid]['frame_list']

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        sid, _, vid_path, start_frame, end_frame = self.segments[index]
        video = []
        
        if os.path.exists(vid_path):
            fid2path = {}
            for img_path in os.listdir(vid_path):
                fid = int(img_path.split('.')[0])
                fid2path[fid] = os.path.join(vid_path, img_path)

            for fid in range(start_frame, end_frame + 1):
                if fid in fid2path.keys():
                    img = cv2.imread(fid2path[fid])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    
                if debug:
                    import matplotlib.pyplot as plt
                    plt.imshow(img)
                    plt.show()  
                video.append(np.expand_dims(img, axis=0))
                
        else:
            for fid in range(self.frame_num[index]):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                
        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        return video
            
            
    def _get_audio(self, index):
        sid, aud_path, _, start_frame, end_frame = self.segments[index]
        audio, sample_rate = soundfile.read(aud_path)
        onset = int(start_frame / 30 * sample_rate)
        offset = int(end_frame / 30 * sample_rate)
        crop_audio = normalize(audio[onset: offset])
        # if self.mode == 'eval':
            # l = offset - onset
            # crop_audio = np.zeros(l)
        #     index = random.randint(0, len(self.segments)-1)
        #     uid, _, _, _, _, _ = self.segments[index]
        #     audio, sample_rate = soundfile.read(f'{self.audio_path}/{uid}.wav')
        #     crop_audio = normalize(audio[onset: offset])
        # else:
        #     crop_audio = normalize(audio[onset: offset])
        return torch.tensor(crop_audio, dtype=torch.float32)

