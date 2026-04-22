#!/bin/bash
BEEKEEPER="http://lab.local:5000/api/v1/projects/breakout-world-models/files/checkpoints"
DEST="$(dirname "$0")/checkpoints"

mkdir -p "$DEST"

echo "Downloading world_model.pt..."
curl -f "$BEEKEEPER/world_model.pt" -o "$DEST/world_model.pt" || { echo "ERROR: world_model.pt failed"; exit 1; }

echo "Downloading q_model.pt..."
curl -f "$BEEKEEPER/q_model.pt" -o "$DEST/q_model.pt" || { echo "ERROR: q_model.pt failed"; exit 1; }

echo "Done."
ls -lh "$DEST"
