#!/usr/bin/env bash
# Download the GGUF weights for all 5 models x precisions (~35 GB), stage-1 files first. Resumable.
# Qwen publishes no official GGUF; Unsloth's conversions have BF16 (no F16).
cd "$(dirname "$0")/.."
get() {  # repo file
  local dir=models/${1#*/}; mkdir -p "$dir"; local f="$dir/$2"
  [ -f "$f" ] && return 0
  until curl -sfL -C - --speed-time 60 --speed-limit 20000 -o "$f.part" "https://huggingface.co/$1/resolve/main/$2"; do
    [ $? -eq 33 ] && break; sleep 5; done
  mv "$f.part" "$f" && echo "$(date +%H:%M:%S) $1/$2 $(du -h "$f" | cut -f1)"
}
# stage 0: everything needed for stage 1 (Q8_0 + F16 projector) of the five models
get LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q8_0.gguf
get LiquidAI/LFM2.5-VL-450M-GGUF mmproj-LFM2.5-VL-450m-F16.gguf
get LiquidAI/LFM2.5-VL-3B-GGUF LFM2.5-VL-3B-Q8_0.gguf
get LiquidAI/LFM2.5-VL-3B-GGUF mmproj-LFM2.5-VL-3B-F16.gguf
get unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf
get unsloth/Qwen3.5-2B-GGUF mmproj-F16.gguf
# stage 1 (cont.): remaining models
get LiquidAI/LFM2.5-VL-1.6B-GGUF mmproj-LFM2.5-VL-1.6b-F16.gguf
get LiquidAI/LFM2.5-VL-1.6B-GGUF LFM2.5-VL-1.6B-Q8_0.gguf
get unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf
get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf
# stage 2: 4-bit and 16-bit variants
get LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-Q4_K_M.gguf
get LiquidAI/LFM2.5-VL-450M-GGUF LFM2.5-VL-450M-F16.gguf
get LiquidAI/LFM2.5-VL-3B-GGUF LFM2.5-VL-3B-Q4_K_M.gguf
get LiquidAI/LFM2.5-VL-1.6B-GGUF LFM2.5-VL-1.6B-Q4_K_M.gguf
get unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf
get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
get LiquidAI/LFM2.5-VL-1.6B-GGUF LFM2.5-VL-1.6B-F16.gguf
get unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-BF16.gguf
get LiquidAI/LFM2.5-VL-3B-GGUF LFM2.5-VL-3B-F16.gguf
get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-BF16.gguf
echo "$(date +%H:%M:%S) all model downloads finished"
