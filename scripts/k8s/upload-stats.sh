#!/bin/bash

set -euo pipefail

# Default values
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
YAML_FILE="${YAML_FILE:-scripts/k8s/upload-stats-job.yaml}"
NAMESPACE="climate-analytics"
STATS_DIR="${STATS_DIR:-$HOME/data/ERA5/statistics}"

echo "=========================================="
echo "Uploading ERA5 Statistics to PVC"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "YAML File: $YAML_FILE"
echo "Namespace: $NAMESPACE"
echo "Source Directory: $STATS_DIR"
echo "=========================================="

# Check if source files exist
if [ ! -f "$STATS_DIR/era5_mean.nc" ] || [ ! -f "$STATS_DIR/era5_std.nc" ]; then
    echo "ERROR: Statistics files not found in $STATS_DIR"
    echo "Expected files:"
    echo "  - $STATS_DIR/era5_mean.nc"
    echo "  - $STATS_DIR/era5_std.nc"
    exit 1
fi

echo "Source files found:"
ls -lh "$STATS_DIR/"

# Check if YAML file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: YAML file '$YAML_FILE' not found!"
    exit 1
fi

# Substitute variables and deploy job
echo ""
echo "Deploying upload job..."
envsubst '$TIMESTAMP' < "$YAML_FILE" | kubectl apply -f -

# Wait for pod to be ready
JOB_NAME="upload-era5-stats-$TIMESTAMP"
echo "Waiting for pod to be created..."
sleep 5

# Wait for pod to exist and be ready
echo "Waiting for pod to be ready..."
for i in {1..12}; do
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD_NAME" ]; then
        break
    fi
    echo "Waiting for pod to be created... ($i/12)"
    sleep 2
done

if [ -z "$POD_NAME" ]; then
    echo "ERROR: Pod was not created"
    kubectl get pods -n "$NAMESPACE" -l job-name="$JOB_NAME"
    exit 1
fi

echo "Pod found: $POD_NAME"
kubectl wait --for=condition=Ready pod "$POD_NAME" -n "$NAMESPACE" --timeout=60s || {
    echo "ERROR: Pod did not become ready in time"
    kubectl get pods -n "$NAMESPACE" -l job-name="$JOB_NAME"
    kubectl describe pod "$POD_NAME" -n "$NAMESPACE" | tail -20
    exit 1
}

# Get pod name
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}')
echo "Pod ready: $POD_NAME"

# Create destination directory in pod
echo "Creating destination directory..."
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- mkdir -p /weatherbench/era5_statistics

# Copy files
echo "Copying era5_mean.nc..."
kubectl cp "$STATS_DIR/era5_mean.nc" "$NAMESPACE/$POD_NAME:/weatherbench/era5_statistics/era5_mean.nc"

echo "Copying era5_std.nc..."
kubectl cp "$STATS_DIR/era5_std.nc" "$NAMESPACE/$POD_NAME:/weatherbench/era5_statistics/era5_std.nc"

# Verify files
echo "Verifying uploaded files:"
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- ls -lh /weatherbench/era5_statistics/

# Get file sizes for verification
LOCAL_MEAN_SIZE=$(stat -c%s "$STATS_DIR/era5_mean.nc" 2>/dev/null || stat -f%z "$STATS_DIR/era5_mean.nc" 2>/dev/null)
PVC_MEAN_SIZE=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- stat -c%s /weatherbench/era5_statistics/era5_mean.nc 2>/dev/null || kubectl exec -n "$NAMESPACE" "$POD_NAME" -- stat -f%z /weatherbench/era5_statistics/era5_mean.nc 2>/dev/null)

if [ "$LOCAL_MEAN_SIZE" != "$PVC_MEAN_SIZE" ]; then
    echo "ERROR: File size mismatch for era5_mean.nc"
    echo "  Local: $LOCAL_MEAN_SIZE bytes"
    echo "  PVC: $PVC_MEAN_SIZE bytes"
    exit 1
fi

LOCAL_STD_SIZE=$(stat -c%s "$STATS_DIR/era5_std.nc" 2>/dev/null || stat -f%z "$STATS_DIR/era5_std.nc" 2>/dev/null)
PVC_STD_SIZE=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- stat -c%s /weatherbench/era5_statistics/era5_std.nc 2>/dev/null || kubectl exec -n "$NAMESPACE" "$POD_NAME" -- stat -f%z /weatherbench/era5_statistics/era5_std.nc 2>/dev/null)

if [ "$LOCAL_STD_SIZE" != "$PVC_STD_SIZE" ]; then
    echo "ERROR: File size mismatch for era5_std.nc"
    echo "  Local: $LOCAL_STD_SIZE bytes"
    echo "  PVC: $PVC_STD_SIZE bytes"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Statistics files uploaded successfully!"
echo "=========================================="
echo "Files are now available at: /weatherbench/era5_statistics/"
echo ""
echo "Cleaning up job..."
kubectl delete job "$JOB_NAME" -n "$NAMESPACE"
echo "=========================================="
