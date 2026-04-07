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
set +e
timeout $DOCKER_BUILD_TIMEOUT docker build -t openenv-eval-test "$REPO_PATH" > "$DOCKER_OUT" 2>&1
DOCKER_EXIT_CODE=$?
set -e

if [ $DOCKER_EXIT_CODE -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] PASSED -- Docker build succeeded"
elif [ $DOCKER_EXIT_CODE -eq 124 ]; then
  echo "[$(date +%H:%M:%S)] FAILED -- Docker build timed out after ${DOCKER_BUILD_TIMEOUT}s"
  tail -n 20 "$DOCKER_OUT"
  echo ""
  echo "Validation stopped at Step 2."
  rm "$DOCKER_OUT"
  exit 1
else
  echo "[$(date +%H:%M:%S)] FAILED -- Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  cat "$DOCKER_OUT"
  echo ""
  echo "Validation stopped at Step 2. Fix the above before continuing."
  rm "$DOCKER_OUT"
  exit 1
fi
rm "$DOCKER_OUT"

echo "[$(date +%H:%M:%S)] Step 3/3: Running openenv validate ..."
if ! command -v openenv &> /dev/null; then
  echo "[$(date +%H:%M:%S)] FAILED -- 'openenv' command not found."
  echo "  Hint: pip install openenv-core"
  exit 1
fi

OPENENV_OUT=$(mktemp)
set +e
openenv validate "$REPO_PATH" > "$OPENENV_OUT" 2>&1
OPENENV_EXIT_CODE=$?
set -e

if [ $OPENENV_EXIT_CODE -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] PASSED -- openenv validate succeeded"
else
  echo "[$(date +%H:%M:%S)] FAILED -- openenv validate returned errors:"
  cat "$OPENENV_OUT"
  echo ""
  echo "Validation stopped at Step 3."
  rm "$OPENENV_OUT"
  exit 1
fi
rm "$OPENENV_OUT"

echo ""
echo "========================================"
echo "  [SUCCESS] All checks passed!"
echo "========================================"
echo "Your repository is structurally valid."
echo ""
