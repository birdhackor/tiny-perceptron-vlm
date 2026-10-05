#!/usr/bin/env bash
set -eu
out=/tmp/phase4-14_9-primary
mkdir -p "$out"
curl --fail --location --proto '=https' --tlsv1.2 --output "$out/roformer-2104.09864v5.pdf" https://arxiv.org/pdf/2104.09864v5
curl --fail --location --proto '=https' --tlsv1.2 --output "$out/yarn-2309.00071v3.pdf" https://arxiv.org/pdf/2309.00071v3
sha256sum "$out/roformer-2104.09864v5.pdf" "$out/yarn-2309.00071v3.pdf"
pdftotext -layout "$out/roformer-2104.09864v5.pdf" "$out/roformer.txt"
pdftotext -layout "$out/yarn-2309.00071v3.pdf" "$out/yarn.txt"
