#!/usr/bin/env bash
# Prompt 80, section 2: prompt 79's three 80x30 placement runs again under the step probe.
# Run from the repository root in Git Bash. The three run at once (the prompt's limit).
# Each reuses its flow states from prompt 79's run file, as prompt 79 reused prompt 75's.
set -u
out=results/builder80/placement
mkdir -p "$out"
for name in door breach dust; do
    .venv/Scripts/python.exe docs/reports/probe80/probe80.py run \
        "results/builder79/configs/placement/scenario_${name}.yaml" \
        --out "$out/${name}.h5" \
        --flows-from "results/builder79/placement/${name}.h5" \
        --concurrent 2 \
        > "$out/${name}.txt" 2> "$out/${name}.log" &
done
wait
echo "exit"
