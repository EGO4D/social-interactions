import os
import json


def get_ttm_result(json_path, result_path):
    with open(json_path) as f:
        reader = json.load(f)
        for video in reader['videos']:
            for clip in video['clips']:
                ttm_list = []
                uid = clip['clip_uid']
                print(uid)
                ttm_data_list = clip['social_segments_talking']
                if not ttm_data_list:
                    continue
                for ttm_data in ttm_data_list:
                    start_frame = ttm_data['start_frame']
                    end_frame = ttm_data['end_frame']
                    if ttm_data['person'] is None:
                        continue
                    person = int(ttm_data['person'].replace("'", ""))
                    if ttm_data['target'] is None:
                        temp_dict = {'label': person, 'start_frame': start_frame, 'end_frame': end_frame}
                        ttm_list.append(temp_dict)
                    else:
                        tag = int(ttm_data['target'].replace("'", ""))
                        temp_dict = {'label': person, 'tags': tag, 'start_frame': start_frame,
                                     'end_frame': end_frame}
                        ttm_list.append(temp_dict)

                json_name = uid + '.json'
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                jsonfile_path = os.path.join(result_path, json_name)
                with open(jsonfile_path, 'a') as json_file:
                    json.dump(ttm_list, json_file)
