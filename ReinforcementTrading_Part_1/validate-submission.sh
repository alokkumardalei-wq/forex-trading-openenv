#!/usr/bin/env bash
# validate-submission.sh
# Requires Docker running

set -e

echo ""
echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"

TARGET_DIR=${2:-"."}
REPO_PATH=$(realpath "$TARGET_DIR")
PING_URL=${1:-"http://localhost:7860"}

DOCKER_BUILD_TIMEOUT=600
TIMEOUT_BIN=""
DOCKER_OUT=""
OPENENV_OUT=""

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
fi

cleanup() {
  rm -f "${DOCKER_OUT:-}" "${OPENENV_OUT:-}"
}

trap cleanup EXIT

echo "[$(date +%H:%M:%S)] Repo:     $REPO_PATH"
echo "[$(date +%H:%M:%S)] Ping URL: $PING_URL"
echo ""

echo "[$(date +%H:%M:%S)] Step 1/3: Pinging HF Space ($PING_URL/reset) ..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{"task": "first_blood", "episode_id": "eval_1"}')

if [ "$HTTP_STATUS" -eq 200 ]; then
  echo "[$(date +%H:%M:%S)] PASSED -- HF Space is live and responds to /reset"
else
  echo "[$(date +%H:%M:%S)] FAILED -- HF Space /reset returned HTTP $HTTP_STATUS (expected 200)"
  echo "  Hint: Make sure your Space is running and the URL is correct."
  echo "  Hint: Try opening $PING_URL in your browser first."
  echo ""
  echo "Validation stopped at Step 1. Fix the above before continuing."
  exit 1
fi

echo "[$(date +%H:%M:%S)] Step 2/3: Running docker build ..."
if [ -f "$REPO_PATH/Dockerfile" ]; then
  echo "[$(date +%H:%M:%S)]   Found Dockerfile in $REPO_PATH"
else
  echo "[$(date +%H:%M:%S)] FAILED -- No Dockerfile found in $REPO_PATH"
  echo "Validation stopped at Step 2."
  exit 1
fi

DOCKER_OUT=$(mktemp)
run_docker_build() {
  set +e
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$DOCKER_BUILD_TIMEOUT" docker build --progress=plain -t openenv-eval-test "$REPO_PATH" 2>&1 | tee "$DOCKER_OUT"
  else
    echo "[$(date +%H:%M:%S)] WARNING -- No timeout command found; running docker build without a timeout."
    docker build --progress=plain -t openenv-eval-test "$REPO_PATH" 2>&1 | tee "$DOCKER_OUT"
  fi
  local build_exit_code=${PIPESTATUS[0]}
  set -e
  return "$build_exit_code"
}

run_docker_build
DOCKER_EXIT_CODE=$?

if [ $DOCKER_EXIT_CODE -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] PASSED -- Docker build succeeded"
elif [ $DOCKER_EXIT_CODE -eq 124 ]; then
  echo "[$(date +%H:%M:%S)] FAILED -- Docker build timed out after ${DOCKER_BUILD_TIMEOUT}s"
  tail -n 20 "$DOCKER_OUT"
  echo ""
  echo "Validation stopped at Step 2."
  exit 1
else
  echo "[$(date +%H:%M:%S)] FAILED -- Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  cat "$DOCKER_OUT"
  echo ""
  echo "Validation stopped at Step 2. Fix the above before continuing."
  exit 1
fi

echo "[$(date +%H:%M:%S)] Step 3/3: Running openenv validate ..."
if ! command -v openenv &> /dev/null; then
  echo "[$(date +%H:%M:%S)] FAILED -- 'openenv' command not found."
  echo "  Hint: pip install openenv-core"
  exit 1
fi

OPENENV_OUT=$(mktemp)
set +e
openenv validate "$REPO_PATH" 2>&1 | tee "$OPENENV_OUT"
OPENENV_EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ $OPENENV_EXIT_CODE -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] PASSED -- openenv validate succeeded"
else
  echo "[$(date +%H:%M:%S)] FAILED -- openenv validate returned errors:"
  cat "$OPENENV_OUT"
  echo ""
  echo "Validation stopped at Step 3."
  exit 1
fi

echo ""
echo "========================================"
echo "  [SUCCESS] All checks passed!"
echo "========================================"
echo "Your repository is structurally valid."
echo ""
