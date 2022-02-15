import os
import json


def get_json(json_path, result_path):
    with open(json_path) as f:
        reader = json.load(f)
        for video in reader['videos']:
            for clip in video['clips']:
                uid = clip['clip_uid']
                print(uid)
                person_data = clip['persons']
                for i in range(1, len(person_data)):
                    tracks = person_data[i]['tracking_paths']
                    for track in tracks:
                        track_list = []
                        track_id = track['track_id']
                        track_data = track['track']
                        for td in track_data:
                            frame = td['frame']
                            person = person_data[i]['person_id']
                            x = td['x']
                            y = td['y']
                            h = td['height']
                            w = td['width']
                            data = {'clip_uid': uid, 'frameNumber': frame, 'Person ID': person, 'x': x, 'y': y,
                                    'height': h, 'width': w}
                            track_list.append(data)

                        folder_path = os.path.join(result_path, uid)
                        json_name = track_id + '.json'
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        jsonfile_path = os.path.join(folder_path, json_name)
                        with open(jsonfile_path, 'a') as json_file:
                            json.dump(track_list, json_file)
