#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
mojo build --emit shared-lib src/openctm.mojo -o dist/libmojo-openctm.so -Xlinker -lm
