#!/bin/bash

sudo apt-get install python3-tk 
sudo apt-get install curl

DEST_DIR="./networks"
mkdir -p "$DEST_DIR"

# Grab .py files and their download URLs directly
curl -s https://api.github.com/repos/lukasiktar/TransUNet_custom/contents/networks |
  grep -E '"(name|download_url)":' |
  paste - - |
  grep '\.py"' |
  while IFS= read -r line; do
    FILE=$(echo "$line" | cut -d '"' -f4)
    URL=$(echo "$line" | cut -d '"' -f8)

    curl -sSL "$URL" -o "$DEST_DIR/$FILE"
    echo "Downloaded: $FILE"
  done
  