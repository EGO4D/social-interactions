import glob, os, shutil, json, csv, sys
from tqdm import tqdm
# vids = glob.glob('/Checkpoint/xuzhongcong/ego4d/video_imgs/*')
# for vid in vids:
#     imgs = glob.glob(f'{vid}/*.jpg')
#     if len(imgs) == 0:
#         print(os.path.basename(vid))
#         continue
#     maxframe = max([int(f.split('/')[-1][4:-4]) for f in imgs])
#     if maxframe < 9000:
#         print(os.path.basename(vid), maxframe)
        # shutil.rmtree(vid)
# videos = [os.path.basename(v)+'\n' for v in vids]

# with open('list/train.txt', 'w+') as f:
#     f.writelines(videos[:int(0.9*len(videos))])

# with open('list/test.txt', 'w+') as f:
#     f.writelines(videos[int(0.9*len(videos)):])
def consistent(a, b):
    return abs(a/30-b) < 0.039


# csv.field_size_limit(sys.maxsize)
# anno_uid = []
# videos = glob.glob('/media/zcxu/Data/ego4d_av_miniset/av_miniset_clips/*.mp4')
# video_uid = sorted([vid.split('/')[-1][:-4] for vid in videos])
# lines = []
# with open('/media/zcxu/Data/ego4d_av_miniset/Social_Step2_Miniset_Export.csv', 'r') as f:
# # with open('/media/zcxu/Data/ego4d_av_miniset/AV_Step123_Miniset_Export.csv', 'r') as f:
#     csv_reader = csv.reader(f)
#     for i, row in enumerate(csv_reader):
#         # if row[0] != 'ff8499e3-5b05-408a-91d2-cfbcc7a9841c':
#         #     continue
#         if i==0:
#             continue
#         field = json.loads(row[1])
#         for idx, k in enumerate(field[0]['payload']):
#             if 'tags' not in k:
#                 continue
#             if not consistent(k['start_frame'], k['start_time']):
#                 print(str(k), row[0])
#             if not consistent(k['end_frame'], k['end_time']):
#                 print(str(k), row[0])
#             if 'label' not in k:
#                 print(str(k), row[0])

# with open('/media/zcxu/Data/ego4d_av_miniset/wrong.csv', 'w+') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(['anno_uid','video_uid'])
#     for line in sorted(lines):
#         csv_writer.writerow(line)
csv.field_size_limit(sys.maxsize)
count0 = 0
count1 = 0
with open('/media/zcxu/Data/ego4d_av_miniset/Social_Step1_Miniset_Export.csv', 'r') as f:
    box_reader = csv.reader(f)
    res = glob.glob('data/result_TTM/*.json')

    for re in tqdm(res):
        with open(re, 'r') as f:
            field = json.loads(f.read())
        for gt in field:
            if 'tags' not in gt:
                continue
            if gt['tags'] == 0:
                count0 += 1
            if gt['tags'] == 1:
                count1 += 1
print(count0, count1)
