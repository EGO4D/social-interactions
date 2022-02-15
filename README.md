# Ego4d Social Benchmark
This repository contains the codebase for Ego4d social benchmark -- *Talking-to-me* baseline models. 

Switch to [*Looking-at-me*](https://github.com/EGO4D/social-interactions/tree/lam).

***

### Dependencies

Start from building the environment
```
sudo apt-get install ffmpeg
pip install -r requirement.txt
```

***
### Data preparation

Skip the following steps if you already get the data following the instructions in *Looking-at-me*.

Download data manifest (`manifest.csv`) and annotations (`av_{train/val/test_unannotated}.json`) for audio-visual diarization benchmark following the Ego4D download [instructions](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).

Note: the default folder to save videos and annotations is ```./data```, please create symbolic links in ```./data``` if you save them in another directory. The structure should be like this:

data/
* csv/
  * manifest.csv
* json/
  * av_train.json
  * av_val.json
  * av_test_unannotated.json
* split/
  * test.list
  * train.list
  * val.list
  * full.list
* videos/
  * 00407bd8-37b4-421b-9c41-58bb8f141716.mp4
  * 007beb60-cbab-4c9e-ace4-f5f1ba73fccf.mp4
  * ...
  
Run the following script to download videos and generate clips:
```
python scripts/download_clips.py
```

Run the following scripts to preprocess the videos and annotations:

```
bash scripts/extract_frame.sh
bash scripts/extract_wave.sh
python scripts/preprocessing.py
```

### 2. Train
Run the following script to start training:
```
python run.py
```
Specify the arguments listed in [common/config.py](./common/config.py) if you want to customize the training.

Note: this codebase does **not** support multi-GPU mode.

### 3. Inference
Download the [checkpoints](https://drive.google.com/drive/folders/1MGrhm3J1dKoWPSL3RvC3qb3QeiIqe9vi?usp=sharing).

Run the following script for inference:
```
python run.py --eval --checkpoint ${checkpoint_path} --exp_path ${eval_output_dir}
```

Our model trained from scratch on Ego4d yields `mAP:75.92% ACC:77.66%` on validation set. 

### Citation

Please cite the following paper if our code is helpful to your research.
```
@article{grauman2021ego4d,
  title={Ego4d: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and others},
  journal={arXiv preprint arXiv:2110.07058},
  year={2021}
}
```
