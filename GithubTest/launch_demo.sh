#!/bin/bash
#
# Wrapper script for launchd to run demo with randomized start time
# This ensures the script runs once per day at a random time within a specified window
#

set -euo pipefail

# Configuration (customize these)
REPO_PATH="${HOME}/path/to/your/demo/repo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/demo_github_activity.py"
LOG_DIR="${REPO_PATH}/logs"

# Set minimum delay (in case launchd starts this multiple times)
LOCK_FILE="/tmp/github_demo_lock_$(date +%Y%m%d).lock"
LOCK_TIMEOUT=300  # 5 minutes

# Daytime window for randomization (in 24-hour format)
HOUR_START=9      # 9 AM
HOUR_END=17       # 5 PM

# Acquire lock to prevent multiple simultaneous runs
if [ -f "$LOCK_FILE" ]; then
    LOCK_AGE=$(($(date +%s) - $(stat -f%m "$LOCK_FILE")))
    if [ "$LOCK_AGE" -lt "$LOCK_TIMEOUT" ]; then
        echo "Another run started recently. Skipping to avoid duplicate commits."
        exit 0
    fi
fi

touch "$LOCK_FILE"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Calculate random delay within the daytime window
SECONDS_PER_HOUR=3600
SECONDS_HOUR_START=$((HOUR_START * SECONDS_PER_HOUR))
SECONDS_HOUR_END=$((HOUR_END * SECONDS_PER_HOUR))
RANDOM_OFFSET=$(( (RANDOM % ((SECONDS_HOUR_END - SECONDS_HOUR_START) / 60)) * 60 ))
DELAY_SECONDS=$(( SECONDS_HOUR_START + RANDOM_OFFSET ))

echo "GitHub Demo Activity - $(date)"
echo "Calculated delay: $DELAY_SECONDS seconds"
echo "Waiting to execute within ${HOUR_START}:00 - ${HOUR_END}:00 window..."

sleep "$DELAY_SECONDS"

echo "Executing demo script at $(date)"

# Source environment setup if it exists
if [ -f "${REPO_PATH}/.github_demo.env" ]; then
    # shellcheck disable=SC1090
    source "${REPO_PATH}/.github_demo.env"
fi

# Ensure required environment variables are set
if [ -z "${GIT_AUTHOR_EMAIL:-}" ]; then
    echo "ERROR: GIT_AUTHOR_EMAIL not set"
    exit 1
fi

export GITHUB_DEMO_REPO="${REPO_PATH}"

# Run the Python script with error logging
python3 "$SCRIPT_PATH" 2>&1 | tee -a "${LOG_DIR}/demo.log"

echo "Demo script completed at $(date)"
