#!/bin/bash
set -e

cd /home/ht-quizdemo
git pull
systemctl restart quizdemo
echo "部署完成"
