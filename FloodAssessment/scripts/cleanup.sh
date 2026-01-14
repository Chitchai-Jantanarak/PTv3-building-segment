#!/bin/bash
# =============================================================================
# Aggressive Deep Cleanup Script
# =============================================================================
# This script attempts to clear disk space in every standard location.
# WARNING: This deletes caches. It will increase time for future installs.
#
# Usage:
#   bash scripts/cleanup.sh
# =============================================================================

echo "==========================================================="
echo "INITIAL DISK USAGE"
echo "==========================================================="
df -h /

echo ""
echo "Finding largest directories in ~ (Home):"
du -sh ~/* 2>/dev/null | sort -rh | head -n 10
echo ""

# 1. Conda Cleanup (Very Aggressive)
echo "1. Cleaning ALL Conda caches and unused packages..."
conda clean --all --yes
# Also check for pkgs directory manually if conda clean misses it
rm -rf ~/miniconda/pkgs/* 2>/dev/null
rm -rf ~/anaconda3/pkgs/* 2>/dev/null
rm -rf /opt/conda/pkgs/* 2>/dev/null

# 2. Pip Cleanup
echo "2. Cleaning Pip cache..."
pip cache purge
rm -rf ~/.cache/pip
rm -rf /root/.cache/pip

# 3. System User Caches (Generic)
echo "3. Removing generic user caches (~/.cache)..."
# This often contains pip, uv, yarn, and other tool caches
rm -rf ~/.cache/*

# 4. NPM/Yarn/Bun Cache (if present)
echo "4. Cleaning node caches..."
rm -rf ~/.npm
rm -rf ~/.yarn
rm -rf ~/.bun/install/cache

# 5. Apt Cache (System level)
if [ "$EUID" -eq 0 ]; then 
  echo "5. Cleaning Apt cache..."
  apt-get clean
  apt-get autoremove -y
  rm -rf /var/lib/apt/lists/*
fi

# 6. Trash
echo "6. Emptying Trash..."
rm -rf ~/.local/share/Trash/*

# 7. Temporary Files (older than 1 day to be safe-ish, or just all)
echo "7. Cleaning /tmp..."
# Only deleting files that look like temp build artifacts
rm -rf /tmp/pip-*
rm -rf /tmp/cu*  # CUDA temp files
rm -rf /tmp/tmp*
rm -rf /tmp/.mbr*

echo "==========================================================="
echo "FINAL DISK USAGE"
echo "==========================================================="
df -h /

echo ""
echo "Remaining large directories in ~ (Home):"
du -sh ~/* 2>/dev/null | sort -rh | head -n 10
