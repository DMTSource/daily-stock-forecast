#!/bin/bash
#curl -L -o DailyStockForecast.zip https://www.dropbox.com/sh/y5ph0xywsw65z78/AABy_zde8Pu4hW5lnBLGhsBUa?dl=1
#unzip DailyStockForecast.zip
#python DailyForecast.py
apt-get -y install imagemagick
IMAGE_URL=$(curl http://metadata/computeMetadata/v1/instance/attributes/url -H "X-Google-Metadata-Request: True")
TEXT=$(curl http://metadata/computeMetadata/v1/instance/attributes/text -H "X-Google-Metadata-Request: True")
CS_BUCKET=$(curl http://metadata/computeMetadata/v1/instance/attributes/cs-bucket -H "X-Google-Metadata-Request: True")
mkdir image-output
cd image-output
wget $IMAGE_URL
convert * -pointsize 50 -fill white -annotate +20+60 "$TEXT" output.png
gsutil cp -a public-read output.png gs://$CS_BUCKET/output.png
curl -sS -L -o DailyStockForecast.zip https://www.dropbox.com/sh/y5ph0xywsw65z78/AABy_zde8Pu4hW5lnBLGhsBUa?dl=1
unzip DailyStockForecast.zip
python DailyForecast.py
