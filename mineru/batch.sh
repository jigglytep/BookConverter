#!/usr/bin/env bash
set -uo pipefail

set -a
[[ -f .env ]] && source .env
set +a

VLM_HEALTH_URL="http://localhost:${VLM_PORT:-30000}/health"
VLM_URL="${VLM_URL:-http://llamacpp:30000}"

if ! curl -sf --max-time 3 "$VLM_HEALTH_URL" >/dev/null 2>&1; then
  echo "llamacpp not responding on $VLM_HEALTH_URL, starting it..."
  podman compose --profile vlm up -d llamacpp

  echo -n "Waiting for llamacpp to become healthy"
  ready=false
  for i in $(seq 1 60); do
    if curl -sf --max-time 3 "$VLM_HEALTH_URL" >/dev/null 2>&1; then
      ready=true
      break
    fi
    echo -n "."
    sleep 5
  done
  echo

  if [[ "$ready" != "true" ]]; then
    echo "llamacpp did not become healthy after 5 min -- check: podman logs llamacpp"
    exit 1
  fi
fi

echo "llamacpp is up: $(curl -s --max-time 3 "$VLM_HEALTH_URL")"

for pdf in in/*.pdf; do
  name=$(basename "$pdf" .pdf)
  [[ -d "out/$name" ]] && { echo "skip $name"; continue; }
  echo "=== $name (vlm)"
  podman compose --profile client run --rm mineru-vlm \
    mineru -p "/data/in/$(basename "$pdf")" -o /data/out -b vlm-http-client -u "$VLM_URL" \
    || echo "FAILED: $name" >> out/failures.txt
done
