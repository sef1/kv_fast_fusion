#!/bin/bash
# Capture py-spy stacks of every vLLM process belonging to THIS repo (worker + EngineCore),
# for hang diagnosis. Run while the system is wedged; output lands in stacks_<HHMMSS>/.
# Workers get an extra --native dump (NCCL/CUDA frames live below Python).
# Scoped by /proc/<pid>/cwd|exe under REPO_ROOT — never touches other users' vLLM servers.
set -u
REPO_ROOT=${REPO_ROOT:-/data/users/sefi/from_git/vllm_013/vllm_ff}
PYSPY="$REPO_ROOT/.venv/bin/py-spy"
OUT="stacks_$(date +%H%M%S)"
mkdir -p "$OUT"

for pid in $(pgrep -u "$USER" -f "VLLM::"); do
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null)
    [[ "$cwd" == "$REPO_ROOT"/* || "$exe" == "$REPO_ROOT"/* ]] || continue
    name=$(cat "/proc/$pid/comm" 2>/dev/null | tr -c 'A-Za-z0-9_' '_')
    echo "dumping $pid ($name)..."
    "$PYSPY" dump --pid "$pid" > "$OUT/${name}_${pid}.txt" 2>&1
    if [[ "$name" == *Worker* ]]; then
        "$PYSPY" dump --native --pid "$pid" > "$OUT/${name}_${pid}_native.txt" 2>&1
    fi
done

echo "Stacks in $OUT/:"
ls -la "$OUT"
