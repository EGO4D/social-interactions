import os
import json


def get_lam_result(json_path, result_path):
    with open(json_path) as f:
        reader = json.load(f)
        for video in reader['videos']:
            for clip in video['clips']:
                lam_list = []
                uid = clip['clip_uid']
                print(uid)
                lam_data_list = clip['social_segments_looking']
                if not lam_data_list:
                    continue
                for lam_data in lam_data_list:
                    if lam_data['person'] is None:
                        continue
                    person = lam_data['person']
                    start_frame = lam_data['start_frame']
                    end_frame = lam_data['end_frame']
                    temp_dict = {'label': person, 'start_frame': start_frame, 'end_frame': end_frame}
                    lam_list.append(temp_dict)

                json_name = uid + '.json'
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                jsonfile_path = os.path.join(result_path, json_name)
                with open(jsonfile_path, 'a') as json_file:
                    json.dump(lam_list, json_file)
