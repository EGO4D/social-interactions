if [ ! -d "data/wave"  ];then
	mkdir "data/wave"
fi
for f in `ls data/videos/`
do
    echo ${f%.*}
    ffmpeg -y -i data/videos/${f} -qscale:a 0 -ac 1 -vn -threads 6 -ar 16000 data/wave/${f%.*}.wav -loglevel panic
done

