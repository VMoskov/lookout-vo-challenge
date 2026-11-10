#!/usr/bin/env bash
# Source this to select bundled assets and native environments; never launches a job.
FERMARBLE_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

fermarble_python() {
  local bundled="$FERMARBLE_REPO_ROOT/portable/environments/$1/bin/python" legacy=$2
  if [ -x "$bundled" ]; then printf '%s\n' "$bundled"
  elif [ -x "$legacy" ]; then printf '%s\n' "$legacy"
  else printf '%s\n' "$bundled"; fi
}

export FERMARBLE_DPVO_PYTHON=${FERMARBLE_DPVO_PYTHON:-$(fermarble_python dpvo "$HOME/micromamba/envs/dpvo/bin/python")}
export FERMARBLE_PANSR_PYTHON=${FERMARBLE_PANSR_PYTHON:-$(fermarble_python pansr "$HOME/micromamba/envs/pansr/bin/python")}
export FERMARBLE_PI3_PYTHON=${FERMARBLE_PI3_PYTHON:-$(fermarble_python pi3 "$HOME/venvs/pi3/bin/python")}
export FERMARBLE_VGGT_PYTHON=${FERMARBLE_VGGT_PYTHON:-$(fermarble_python vggt "$HOME/venvs/vggt/bin/python")}

fermarble_checkout() {
  local bundled="$FERMARBLE_REPO_ROOT/portable/checkouts/$1"
  if [ -d "$bundled" ]; then printf '%s\n' "$bundled"
  else printf '%s\n' "$FERMARBLE_REPO_ROOT/../$1"; fi
}

export DPVO_DIR=${DPVO_DIR:-$(fermarble_checkout DPVO-859bbbf)}
export PI3_DIR=${PI3_DIR:-$(fermarble_checkout Pi3)}
export VGGT_OMEGA_DIR=${VGGT_OMEGA_DIR:-$(fermarble_checkout vggt-omega)}
export OMEGA_CKPT=${OMEGA_CKPT:-$FERMARBLE_REPO_ROOT/portable/checkpoints/omega/vggt_omega_1b_512.pt}
export PYTHONPATH="$(fermarble_checkout vggt):$(fermarble_checkout PanSR)${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$FERMARBLE_REPO_ROOT/portable/huggingface/hub" ]; then
  export HF_HOME=${HF_HOME:-$FERMARBLE_REPO_ROOT/portable/huggingface}
  export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
fi

fermarble_require_python() {
  if [ ! -x "$1" ]; then
    printf 'Missing %s Python: %s. See %s/PORTABILITY.md for native environment setup.\n' "$2" "$1" "$FERMARBLE_REPO_ROOT" >&2
    return 1
  fi
}
