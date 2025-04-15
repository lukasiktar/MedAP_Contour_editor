#!/bin/bash

DEST_DIR="./networks"
mkdir -p "$DEST_DIR"

# Dynamically fetch all .py files in the `networks` folder of the repo
FILES=$(curl -s https://api.github.com/repos/lukasiktar/TransUNet_custom/contents/networks |
         grep '"name":' | grep '\.py"' | cut -d '"' -f4)

BASE_URL="https://raw.githubusercontent.com/lukasiktar/TransUNet_custom/main/networks"

echo "Downloading Python files into $DEST_DIR ..."

for FILE in $FILES; do
    curl -sSL "$BASE_URL/$FILE" -o "$DEST_DIR/$FILE"
    echo "Downloaded: $FILE"
done

echo "All files from $BASE_URL downloaded into $DEST_DIR."
