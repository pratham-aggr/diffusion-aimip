### 2. Deploy Training Job

```bash
# From project root
bash scripts/k8s/deploy-job.sh
```

Or with custom WANDB secret:
```bash
WANDB_SECRET_NAME=your-secret bash scripts/k8s/deploy-job.sh
```

### 3. Monitor Job

```bash
# Check status
kubectl get jobs -n climate-analytics | grep aimip

# View logs (follow)
kubectl logs -n climate-analytics job/aimip --follow

# View logs (last 100 lines)
kubectl logs -n climate-analytics job/aimip --tail=100
```

### 4. Clean Up

```bash
# Delete job
kubectl delete job aimip -n climate-analytics

# Delete job and pods
kubectl delete job aimip -n climate-analytics
kubectl delete pods -n climate-analytics -l job-name=aimip
```

