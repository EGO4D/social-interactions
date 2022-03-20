import os
import json
import soundfile as sf
from tqdm import tqdm

data_dir = './final_test_data'
seginfo = {}

for seg_id in tqdm(os.listdir(data_dir)):
    seg_path = os.path.join(data_dir, seg_id)
    seginfo[seg_id] = {}
    frame_list = []
    for f in os.listdir(os.path.join(seg_path, 'face')):
        fid = int(f.split('.')[0])
        frame_list.append(fid)
    frame_list.sort()
    seginfo[seg_id]['frame_list'] = frame_list
    
    aud, sr = sf.read(os.path.join(seg_path, 'audio', 'aud.wav'))
    frame_num = int(aud.shape[0]/sr*30+1)
    seginfo[seg_id]['frame_num'] = max(frame_num, max(frame_list)+1)
    
with open('./seg_info.json','w') as f:
    json.dump(seginfo, f, indent=4)
        
