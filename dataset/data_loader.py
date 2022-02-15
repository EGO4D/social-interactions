import os, cv2, json, glob, logging, soundfile
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict


logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or 
            frame['frameNumber']==0 or
            len(frame['Person ID'])==0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)
    
    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1]+1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0,4):
            interpfn  = interp1d(framenum, bboxes[:,ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i  = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    #assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def get_bbox(uid, json_path):
    bboxes = {}
    vid_json_dir = os.path.join(json_path, uid)
    tracklets = glob.glob(f'{vid_json_dir}/*.json')
    for idx, t in enumerate(tracklets):
        with open(t, 'r') as j:
            frames = json.load(j)

        # check the bbox, interpolate when necessary
        frames = check(frames)

        for frame in frames:
            frameid = frame['frameNumber']
            bbox = (frame['x'],
                    frame['y'],
                    frame['x'] + frame['width'],
                    frame['y'] + frame['height'])
            identifier = str(frameid) + ':' + frame['Person ID']
            bboxes[identifier] = bbox

    return bboxes


def make_dataset(file_list, img_anno, audio_anno, stride=1, min_frames=15, max_frames=150):

    logger.info('load: ' + file_list)
    face_crop = {}
    segments = []

    with open(file_list, 'r') as f:
        videos = f.readlines()

    for uid in videos:
        uid = uid.strip()
        face_crop[uid] = get_bbox(uid, img_anno)

        with open(os.path.join(audio_anno, uid + '.json'), ) as js:
            gts = json.load(js)

        for idx, gt in enumerate(gts):
            if 'tags' not in gt:
                personid = gt['label']
                label = 0
                start_frame = int(gt['start_frame'])
                end_frame = int(gt['end_frame'])
                seg_length = end_frame - start_frame + 1

            else:
                personid = gt['label']
                label = 1
                start_frame = int(gt['start_frame'])
                end_frame = int(gt['end_frame'])
                seg_length = end_frame - start_frame + 1

            if ('train' in file_list and seg_length < min_frames) or (seg_length <= 1):
                continue
            elif seg_length > max_frames:
                it = int(seg_length / max_frames)
                for i in range(it):
                    sub_start = start_frame + i*max_frames
                    sub_end = min(end_frame, sub_start + max_frames)
                    sub_length = sub_end - sub_start + 1
                    if sub_length < min_frames:
                        continue
                    segments.append([uid, personid, label, sub_start, sub_end, idx])
            else:
                segments.append([uid, personid, label, start_frame, end_frame, idx])
    return segments, face_crop


class ImagerLoader(torch.utils.data.Dataset):
    def __init__(self, img_path, audio_path, file_list, img_json, audio_json,
                 stride=1, mode='train', transform=None):

        self.img_path = img_path
        assert os.path.exists(self.img_path), 'image path not exist'
        self.audio_path = audio_path
        assert os.path.exists(self.audio_path), 'audio path not exist'
        self.file_list = file_list
        assert os.path.exists(self.file_list), f'{mode} list not exist'
        self.img_json = img_json
        assert os.path.exists(self.img_json), 'json path not exist'
        self.audio_json = audio_json
        assert os.path.exists(self.audio_json), 'talking to me path not exist'

        segments, face_crop = make_dataset(file_list, img_json, audio_json, stride=stride)
        self.segments = segments
        self.face_crop = face_crop
        self.transform = transform
        self.mode = mode

    def __getitem__(self, indices):
        source_video = self._get_video(indices)
        source_audio = self._get_audio(indices)
        target = self._get_target(indices)
        return source_video, source_audio, target

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        uid, personid, _, start_frame, end_frame, _ = self.segments[index]
        video = []
        for i in range(start_frame, end_frame + 1):
            key = str(i) + ':' + str(personid)
            if key in self.face_crop[uid]:
                bbox = self.face_crop[uid][key]
                img = f'{self.img_path}/{uid}/img_{i:05d}.jpg'

                if not os.path.exists(img):
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    continue
                
                assert os.path.exists(img), f'img: {img} not found'
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                face = img[y1: y2, x1: x2, :]
                try:
                    face = cv2.resize(face, (224, 224))
                except:
                    # bad bbox
                    face = np.zeros((224, 224, 3), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt
                    plt.imshow(face)
                    plt.show()
                    
                video.append(np.expand_dims(face, axis=0))
            else:
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                continue
        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        return video

    def _get_audio(self, index):
        uid, _, _, start_frame, end_frame, _ = self.segments[index]
        audio, sample_rate = soundfile.read(f'{self.audio_path}/{uid}.wav')
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

    def _get_target(self, index):
        if self.mode == 'train':
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]
