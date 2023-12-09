#!/bin/sh
# Exit script on error
set -e

# Set the working directory correctly
cd "$(dirname "$0")"

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************
GV_ROOT=`(cd ../.. && pwd)`

# Generates standard UNIX makefiles
mkdir -p $GV_ROOT/Generated_Linux/Tools

# Generates standard UNIX makefiles
cd $GV_ROOT/Generated_Linux/Tools
cmake -G "Unix Makefiles" $GV_ROOT/Development/Tools
