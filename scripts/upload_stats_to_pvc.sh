#!/bin/bash
# Script to upload locally computed statistics to Kubernetes PVC

set -euo pipefail

LOCAL_STATS_DIR="${LOCAL_STATS_DIR:-./era5_statistics}"
PVC_NAME="${PVC_NAME:-weatherbench}"
PVC_PATH="${PVC_PATH:-/weatherbench/era5_statistics}"
NAMESPACE="${NAMESPACE:-climate-analytics}"

echo "=========================================="
echo "Uploading Statistics to PVC"
echo "=========================================="
echo "Local directory: $LOCAL_STATS_DIR"
echo "PVC: $PVC_NAME"
echo "PVC path: $PVC_PATH"
echo "=========================================="

# Check if local files exist
if [ ! -f "$LOCAL_STATS_DIR/era5_mean.nc" ] || [ ! -f "$LOCAL_STATS_DIR/era5_std.nc" ]; then
    echo "ERROR: Statistics files not found in $LOCAL_STATS_DIR"
    echo "Please compute statistics first or check the directory path."
    exit 1
fi

# Create a temporary pod that mounts the PVC
POD_NAME="stats-upload-helper-$(date +%s)"
echo ""
echo "Creating helper pod: $POD_NAME"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: copy-helper
    image: busybox:latest
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: weatherbench
      mountPath: /weatherbench
  volumes:
  - name: weatherbench
    persistentVolumeClaim:
      claimName: $PVC_NAME
EOF

# Wait for pod to be ready
echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s || {
    echo "ERROR: Pod did not become ready in time"
    kubectl delete pod $POD_NAME -n $NAMESPACE 2>/dev/null || true
    exit 1
}

# Create directory in PVC
echo "Creating directory in PVC..."
kubectl exec -n $NAMESPACE $POD_NAME -- mkdir -p "$PVC_PATH"

# Copy files to the pod
echo "Copying statistics files to PVC..."
kubectl cp "$LOCAL_STATS_DIR/era5_mean.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_mean.nc
kubectl cp "$LOCAL_STATS_DIR/era5_std.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_std.nc

# Verify files were copied
echo "Verifying files..."
kubectl exec -n $NAMESPACE $POD_NAME -- ls -lh "$PVC_PATH/" | grep -E "era5_mean|era5_std"

# Clean up helper pod
echo "Cleaning up helper pod..."
kubectl delete pod $POD_NAME -n $NAMESPACE

echo ""
echo "=========================================="
echo "Statistics successfully uploaded to PVC!"
echo "=========================================="
echo "PVC: $PVC_NAME"
echo "Path: $PVC_PATH"
echo ""
echo "Files:"
echo "  - $PVC_PATH/era5_mean.nc"
echo "  - $PVC_PATH/era5_std.nc"
echo "=========================================="

