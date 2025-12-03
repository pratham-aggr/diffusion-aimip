#!/bin/bash
# Script to compute ERA5 statistics locally and store in Kubernetes PVC
# This computes statistics locally first, then copies them to the PVC via a helper pod

set -euo pipefail

DATA_DIR="${DATA_DIR:-gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr}"
OUTPUT_DIR="${OUTPUT_DIR:-./era5_statistics}"
PVC_NAME="${PVC_NAME:-weatherbench}"
PVC_PATH="${PVC_PATH:-/weatherbench/era5_statistics}"
NAMESPACE="${NAMESPACE:-climate-analytics}"

echo "=========================================="
echo "Computing ERA5 Statistics Locally"
echo "=========================================="
echo "Data source: $DATA_DIR"
echo "Local output: $OUTPUT_DIR"
echo "PVC: $PVC_NAME"
echo "PVC path: $PVC_PATH"
echo "=========================================="

# Step 1: Compute statistics locally
echo ""
echo "Step 1: Computing statistics locally..."
mkdir -p "$OUTPUT_DIR"

python compute_era5_statistics.py \
  --zarr-path "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-start 1979-01-01 \
  --train-end 2020-12-31 \
  --force

# Verify files were created
if [ ! -f "$OUTPUT_DIR/era5_mean.nc" ] || [ ! -f "$OUTPUT_DIR/era5_std.nc" ]; then
    echo "ERROR: Statistics files were not created!"
    exit 1
fi

echo ""
echo "✓ Statistics computed successfully!"
echo "  - $OUTPUT_DIR/era5_mean.nc"
echo "  - $OUTPUT_DIR/era5_std.nc"

# Step 2: Copy to PVC via a helper pod
echo ""
echo "Step 2: Copying statistics to PVC..."
echo "Creating helper pod to copy files..."

# Create a temporary pod that mounts the PVC
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: stats-copy-helper-$(date +%s)
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
POD_NAME=$(kubectl get pods -n $NAMESPACE --sort-by=.metadata.creationTimestamp | grep stats-copy-helper | tail -1 | awk '{print $1}')
echo "Waiting for pod $POD_NAME to be ready..."
kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s || {
    echo "Pod did not become ready in time"
    kubectl delete pod $POD_NAME -n $NAMESPACE
    exit 1
}

# Copy files to the pod
echo "Copying statistics files to PVC..."
kubectl cp "$OUTPUT_DIR/era5_mean.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_mean.nc
kubectl cp "$OUTPUT_DIR/era5_std.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_std.nc

# Verify files were copied
kubectl exec -n $NAMESPACE $POD_NAME -- ls -lh $PVC_PATH/ | grep -E "era5_mean|era5_std" || {
    echo "WARNING: Could not verify files in PVC"
}

# Clean up helper pod
echo "Cleaning up helper pod..."
kubectl delete pod $POD_NAME -n $NAMESPACE

echo ""
echo "=========================================="
echo "Statistics successfully stored in PVC!"
echo "=========================================="
echo "PVC: $PVC_NAME"
echo "Path: $PVC_PATH"
echo ""
echo "Files:"
echo "  - $PVC_PATH/era5_mean.nc"
echo "  - $PVC_PATH/era5_std.nc"
echo "=========================================="

