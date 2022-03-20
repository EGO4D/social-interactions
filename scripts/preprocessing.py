import os
from get_json import get_json
from get_lam_result import get_lam_result
from get_ttm_result import get_ttm_result
from merge import merge_json


json_path = 'data/json/av.json'                # the path of annotation json file
original_path = 'data/json_original'           # the saving path of annotation of the face bbox
result_LAM_path = 'data/result_LAM'            # the saving path of LAM annotation
result_TTM_path = 'data/result_TTM'            # the saving path of TTM annotation

print("merge the json file...")
merge_json('data/json', json_path)

print("processing tracklets...")
get_json(json_path, original_path)

print("processing looking_at_me annotation...")
get_lam_result(json_path, result_LAM_path)

print("processing talking_to_me annotation...")
get_ttm_result(json_path, result_TTM_path)
