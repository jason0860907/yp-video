#!/bin/bash
source .venv/bin/activate

# 在這裡列出要處理的比賽
GAMES="G11 G12 G13 G14 G15"

for game in $GAMES; do
    for video in ~/videos/cuts/*_${game}_*.mp4; do
        if [ -f "$video" ]; then
            python vlm_segment.py --video "$video"
        fi
    done
done
