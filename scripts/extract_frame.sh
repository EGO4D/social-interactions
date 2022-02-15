#!/bin/bash
if [ ! -d "data/video_imgs"  ];then
	mkdir "data/video_imgs"
fi
for file in `ls data/videos/*`
do
	name=$(basename $file .mp4)
	echo "$name"
	PTHH=data/video_imgs/$name
	if [ ! -d "$PTHH"  ];then
		mkdir "$PTHH"
	fi
	ffmpeg -i "$file" -f image2 -vf fps=30 -qscale:v 2 "$PTHH/img_%05d.jpg"
done
