#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
inkscape --version
/usr/bin/chromium --version
inkscape figure-original.svg --export-type=png --export-filename=figure-render.png
cat > figure-browser-wrapper.html <<'HTML'
<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>14.9 independent figure rendering</title><style>body{margin:16px;background:white}img{display:block;width:640px;max-width:100%;height:auto}</style><img src="figure-original.svg" alt="快慢兩組指針在位置插值後，相鄰卡片的旋轉差都減半"></html>
HTML
/usr/bin/chromium --headless --no-sandbox --disable-gpu --hide-scrollbars --window-size=1280,800 --screenshot="$PWD/figure-desktop.png" "file://$PWD/figure-browser-wrapper.html"
/usr/bin/chromium --headless --no-sandbox --disable-gpu --hide-scrollbars --window-size=390,844 --screenshot="$PWD/figure-mobile.png" "file://$PWD/figure-browser-wrapper.html"
