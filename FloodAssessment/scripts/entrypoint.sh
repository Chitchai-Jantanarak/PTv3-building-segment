#!/bin/bash
set -e

# Check if pointops is compiled
if [ -d "/workspace/PointTransformerV3/libs/pointops" ]; then
    echo "Checking pointops compilation..."
    cd /workspace/PointTransformerV3/libs/pointops
    
    # Simple check if there are any build artifacts (so files)
    if [ -z "$(find . -name '*.so' -print -quit)" ]; then
        echo "Compiling pointops..."
        python setup.py install
    else
        echo "pointops seems to be compiled."
    fi
    
    cd /workspace
else
    echo "Warning: PointTransformerV3/libs/pointops not found. Skipping compilation."
fi

exec "$@"
