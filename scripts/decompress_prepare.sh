#!/bin/bash

COMPRESSED_FNAME="lfw-funneled.tgz.zip"
DECOMPRESSED_FOLDER="lfw_funneled"

echo "[INFO] removing old dir..."
rm -rf "${DECOMPRESSED_FOLDER}"
echo "[INFO] decompressing package..."
unzip -qq -c "${COMPRESSED_FNAME}" | tar x
venv/bin/python prepare_dataset
