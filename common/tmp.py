import glob, os, shutil, json
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


# import csv, sys, glob, json
# csv.field_size_limit(sys.maxsize)
# anno_uid = []
# videos = glob.glob('/media/zcxu/Data/ego4d_av_miniset/av_miniset_clips/*.mp4')
# video_uid = sorted([vid.split('/')[-1][:-4] for vid in videos])
# lines = []
# with open('/media/zcxu/Data/ego4d_av_miniset/Social_Step1_Miniset_Export.csv', 'r') as f:
# # with open('/media/zcxu/Data/ego4d_av_miniset/AV_Step123_Miniset_Export.csv', 'r') as f:
#     csv_reader = csv.reader(f)
#     for row in csv_reader:
#         # if row[0] != 'ff8499e3-5b05-408a-91d2-cfbcc7a9841c':
#         #     continue
#         if row[0] == 'clip_uid':
#             continue
#         field = json.loads(row[2])
#         for idx, k in enumerate(field[0]['payload']):
#             zero_field = []
#             if k['width'] < 0:
#                 zero_field.append('width')
#             if k['height'] < 0:
#                 zero_field.append('height')
#             if k['x'] < 0:
#                 zero_field.append('x')
#             if k['y'] < 0:
#                 zero_field.append('y')
#             if len(zero_field) > 0:
#                 lines.append([row[0], idx, zero_field])

# with open('/media/zcxu/Data/ego4d_av_miniset/wrong.csv', 'w+') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(['anno_uid','video_uid'])
#     for line in sorted(lines):
#         csv_writer.writerow(line)

res = glob.glob('data/result/*.json')
count = 0
for re in res:
    with open(re, 'r') as f:
        field = json.loads(f.read())
    for record in field:
        count += (record['end_frame'] - record['start_frame'] + 1)
a = 1
