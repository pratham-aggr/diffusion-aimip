#!/bin/bash
# Script to compute ERA5 statistics locally and upload to Kubernetes PVC
# This runs entirely on your local machine, then uploads results to PVC

set -euo pipefail

DATA_DIR="${DATA_DIR:-gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./era5_statistics}"
PVC_NAME="${PVC_NAME:-weatherbench}"
PVC_PATH="${PVC_PATH:-/weatherbench/era5_statistics}"
NAMESPACE="${NAMESPACE:-climate-analytics}"

echo "=========================================="
echo "Compute ERA5 Statistics Locally & Upload to PVC"
echo "=========================================="
echo "Data source: $DATA_DIR"
echo "Local output: $LOCAL_OUTPUT_DIR"
echo "PVC: $PVC_NAME"
echo "PVC path: $PVC_PATH"
echo "=========================================="

# Step 1: Compute statistics locally
echo ""
echo "Step 1: Computing statistics locally..."
mkdir -p "$LOCAL_OUTPUT_DIR"

# Check if script exists
if [ ! -f "compute_era5_statistics.py" ]; then
    echo "ERROR: compute_era5_statistics.py not found in current directory"
    echo "Please run this script from the repository root directory."
    exit 1
fi

echo "Running compute_era5_statistics.py..."
python compute_era5_statistics.py \
  --zarr-path "$DATA_DIR" \
  --output-dir "$LOCAL_OUTPUT_DIR" \
  --train-start 1979-01-01 \
  --train-end 2020-12-31 \
  --force

# Verify files were created
if [ ! -f "$LOCAL_OUTPUT_DIR/era5_mean.nc" ] || [ ! -f "$LOCAL_OUTPUT_DIR/era5_std.nc" ]; then
    echo "ERROR: Statistics files were not created!"
    exit 1
fi

echo ""
echo "✓ Statistics computed successfully!"
echo "  - $LOCAL_OUTPUT_DIR/era5_mean.nc"
echo "  - $LOCAL_OUTPUT_DIR/era5_std.nc"

# Step 2: Upload to PVC via helper pod
echo ""
echo "Step 2: Uploading statistics to PVC..."

# Create a temporary pod that mounts the PVC
POD_NAME="stats-upload-$(date +%s)"
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
kubectl cp "$LOCAL_OUTPUT_DIR/era5_mean.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_mean.nc
kubectl cp "$LOCAL_OUTPUT_DIR/era5_std.nc" $NAMESPACE/$POD_NAME:$PVC_PATH/era5_std.nc

# Verify files were copied
echo "Verifying files..."
kubectl exec -n $NAMESPACE $POD_NAME -- ls -lh "$PVC_PATH/" | grep -E "era5_mean|era5_std"

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
echo ""
echo "The Kubernetes job will now use these pre-computed statistics!"
echo "=========================================="

