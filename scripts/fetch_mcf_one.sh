#!/usr/bin/env bash
# Download one MCF scenario zip (e.g. 07) with resume and stall detection.
f=data/raw/mcf/zips/chute$1.zip
[ -f "$f" ] && exit 0
until curl -sfL -C - --speed-time 60 --speed-limit 5000 -o "$f.part" \
  "https://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute$1.zip"; do
  [ $? -eq 33 ] && break; sleep 5
done
mv "$f.part" "$f" && echo "$(date +%H:%M:%S) mcf chute$1 done"
