#!/usr/bin/env bash
set -eo pipefail

LABEL="${1:-manual}"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"

BACKGROUND_DIR="$HOME/ros2_ws/backgrounds"
LATEST_FILE="$BACKGROUND_DIR/latest_background.npz"
ARCHIVE_FILE="$BACKGROUND_DIR/background_${LABEL}.npz"
LOG_FILE="$BACKGROUND_DIR/background_capture_schedule.log"

mkdir -p "$BACKGROUND_DIR"

{
  echo "========================================"
  echo "Background capture started: $(date)"
  echo "Label: $LABEL"
  echo "ROS distro: $ROS_DISTRO_NAME"

  if [ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]; then
    echo "ERROR: ROS setup not found: /opt/ros/$ROS_DISTRO_NAME/setup.bash"
    exit 1
  fi

  if [ ! -f "$HOME/ros2_ws/install/setup.bash" ]; then
    echo "ERROR: Workspace setup not found: $HOME/ros2_ws/install/setup.bash"
    exit 1
  fi

  source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
  source "$HOME/ros2_ws/install/setup.bash"

  echo "Waiting for ROS2 background services..."

  for i in {1..60}; do
    if ros2 service list | grep -q "^/tracking/capture_background$" \
      && ros2 service list | grep -q "^/tracking/clear_background$"; then
      echo "Services found."
      break
    fi

    if [ "$i" -eq 60 ]; then
      echo "ERROR: Background services not available after 60 seconds."
      exit 1
    fi

    sleep 1
  done

  START_EPOCH="$(date +%s)"

  echo "Clearing old background..."
  ros2 service call /tracking/clear_background std_srvs/srv/Trigger "{}"

  echo "Starting background capture..."
  ros2 service call /tracking/capture_background std_srvs/srv/Trigger "{}"

  echo "Waiting until latest_background.npz is updated..."

  for i in {1..180}; do
    if [ -f "$LATEST_FILE" ]; then
      FILE_EPOCH="$(stat -c %Y "$LATEST_FILE")"

      if [ "$FILE_EPOCH" -ge "$START_EPOCH" ]; then
        cp -f "$LATEST_FILE" "$ARCHIVE_FILE"
        echo "Saved archive: $ARCHIVE_FILE"
        ls -lh "$ARCHIVE_FILE"
        echo "Background capture finished: $(date)"
        echo "========================================"
        exit 0
      fi
    fi

    sleep 1
  done

  echo "ERROR: latest_background.npz was not updated within 180 seconds."
  exit 1
} >> "$LOG_FILE" 2>&1
