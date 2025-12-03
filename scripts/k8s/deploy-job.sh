#!/bin/bash

set -euo pipefail

# Default values
export JOB_NAME="${JOB_NAME:-aimip-$(date +%Y%m%d-%H%M%S)}"
export WANDB_SECRET_NAME="${WANDB_SECRET_NAME:-praggarwal-wandb}"
YAML_FILE="${YAML_FILE:-scripts/k8s/aimip.yaml}"
NAMESPACE="climate-analytics"

echo "=========================================="
echo "Deploying Kubernetes Job"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo "WANDB Secret: $WANDB_SECRET_NAME"
echo "YAML File: $YAML_FILE"
echo "Namespace: $NAMESPACE"
echo "=========================================="


# Check if wandb secret exists
if ! kubectl get secret "$WANDB_SECRET_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "WARNING: Secret '$WANDB_SECRET_NAME' not found in namespace '$NAMESPACE'"
    echo "Available secrets:"
    kubectl get secrets -n "$NAMESPACE"
    exit 1
fi

# Check if YAML file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "ERROR: YAML file '$YAML_FILE' not found!"
    exit 1
fi

# Substitute variables and deploy
echo ""
echo "Substituting variables and deploying job..."
envsubst '$JOB_NAME $WANDB_SECRET_NAME' < "$YAML_FILE" | kubectl apply -f -

echo ""
echo "=========================================="
echo "Job deployed successfully!"
echo "=========================================="
echo "To check job status:"
echo "  kubectl get jobs -n $NAMESPACE | grep $JOB_NAME"
echo ""
echo "To view job logs:"
echo "  kubectl logs -n $NAMESPACE job/$JOB_NAME --follow"
echo ""
echo "To delete the job:"
echo "  kubectl delete job $JOB_NAME -n $NAMESPACE"
echo "=========================================="

