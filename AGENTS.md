# AGENTS.md

## Purpose
- This workspace contains a ROS 2 Jazzy project for cold-store rack detection and tracking.
- The main package is `src/coldstore_tracking`.
- Current development focus is the YOLO OBB BEV detector plus MOT-Light integration in `yolo_obb_bev_detector_node.py`.

## Project Layout
- `src/coldstore_tracking/coldstore_tracking/`
  - Runtime nodes and shared Python code.
- `src/coldstore_tracking/config/`
  - ROS 2 parameter YAML files.
- `src/coldstore_tracking/launch/`
  - Launch files for the legacy cluster pipeline.
- Workspace root scripts
  - Dataset prep, pseudo-label generation, BEV export, and debugging utilities.

## Key Runtime Nodes
- `yolo_obb_bev_detector_node.py`
  - Main YOLO26m-OBB live detector.
  - Builds a full BEV image from LiDAR points.
  - Runs 1024x1024 sliding-window inference with overlap.
  - Transforms tile detections back into full-BEV coordinates.
  - Applies OBB-NMS and cross-class deduplication.
  - Runs MOT-Light directly in the same node.
  - Publishes markers, centroids, raw detection JSON, and stable track JSON.
- `track_manager_node.py`
  - Legacy track consumer for cluster-based pipeline.
  - Can now also consume stable MOT-Light tracks from the YOLO node.
  - Keeps barcode assignment and track-state publishing.
- `cluster_detector_node.py`
  - Legacy point-cloud clustering pipeline.
- `virtual_scanner_node.py`
  - Generates simulated entry/exit scan events from track states.
- `id_assignment_node.py`
  - Assigns or removes barcode IDs based on scan events.
- `regal_mover_node.py`
  - Simulation/support node for moving racks.

## Important Topics
- YOLO detector outputs
  - `/detection/rack_obb_markers`
  - `/detection/rack_centroids`
  - `/detection/rack_detections_json`
  - `/tracking/stable_tracks`
- Track manager outputs
  - `/tracking/track_markers`
  - `/tracking/track_states`
- Legacy clustering inputs
  - `/tracking/cluster_centroids`
  - `/tracking/touched_cluster_centroids`

## Stable Track Interface
- Preferred structured interface from YOLO to downstream tracking is `/tracking/stable_tracks`.
- Message type is `std_msgs/String` with JSON payload.
- Expected track fields:
  - `track_id`
  - `class_id`
  - `class_name`
  - `state`
  - `confidence`
  - `center_x`, `center_y`, `center_z`
  - `yaw`
  - `length`, `width`, `height`
  - `hit_count`
  - `missed_count`
  - top-level `frame_id`
  - top-level `stamp.sec` and `stamp.nanosec`
- Downstream behavior:
  - `confirmed` tracks are final usable objects.
  - `lost` tracks may still be kept, depending on config.
  - `tentative` tracks must not be treated as final objects.

## Current Behavioral Assumptions
- Marker yaw in the YOLO node is intentionally calibrated and should not be changed casually.
- Lost tracks may remain visible as transparent boxes in RViz.
- Lost tracks should not publish text markers.
- Marker cleanup currently relies on short marker lifetime plus `DELETEALL` for both marker namespaces.
- MOT-Light lives inside `yolo_obb_bev_detector_node.py`; do not split it out unless explicitly requested.

## Config Files To Check First
- `src/coldstore_tracking/config/yolo_obb_bev_detector.yaml`
  - Main config for the YOLO + MOT-Light path.
- `src/coldstore_tracking/config/real_single_lidar_tracking.yaml`
  - Main config for the older cluster-based pipeline.

## How To Work Safely
- Preserve existing topics unless there is a clear reason to add a new one.
- Prefer additive changes over rewrites.
- Keep `yolo_obb_bev_detector_node.py` as the owner of:
  - YOLO inference
  - BEV generation
  - OBB postprocessing
  - MOT-Light stabilization
- Keep `track_manager_node.py` as a downstream consumer, not the place for raw frame-by-frame YOLO matching.
- If changing payloads on stringified JSON topics, extend them compatibly instead of replacing existing fields.
- Keep legacy cluster-pipeline behavior available behind parameters where practical.



## User Preferences For Agent Responses
- Do not paste full files or full patches unless explicitly requested.
- Prefer concise summaries of what changed.
- When changing code, reference the touched files and the important behavior change.
- Avoid unnecessary architecture proposals; stay close to the existing implementation.

## When Unsure
- Inspect the current code before proposing structural changes.
- Treat the current worktree as the source of truth.
- Prefer small, testable, reversible edits.
