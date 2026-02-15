#!/bin/bash
source .venv/bin/activate

# 在這裡列出要處理的比賽
GAMES="G3 G4 G5 G7 G8 G9 G10"

for game in $GAMES; do
    for video in ~/videos/cuts/*_${game}_*.mp4; do
        if [ -f "$video" ]; then
            python detect_volleyball.py --video "$video"
        fi
    done
done
