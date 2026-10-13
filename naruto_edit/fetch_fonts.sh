#!/usr/bin/env bash
# Downloads the two OFL fonts used by the edit (Anton + Noto Sans JP).
set -e
cd "$(dirname "$0")/fonts"
curl -sSL -o Anton.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf"
curl -sSL -o JP.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
