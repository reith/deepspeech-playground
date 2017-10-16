# Convert all .flac files within this folder to .wav files

find . -iname "*.flac" | wc

for flacfile in `find . -iname "*.flac"`
do
    [ -e "${flacfile%.*}.wav" ] || ffmpeg -i $flacfile -ab 64k -ac 1 -ar 16000 -loglevel 16 "${flacfile%.*}.wav"
done
