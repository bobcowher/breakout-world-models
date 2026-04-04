# Beekeeper: breakout-world-models

## Documentation Resources

**Project-Specific Instructions (this file):**
```bash
curl -o BEEKEEPER_breakout-world-models.md http://lab.local:5000/api/v1/projects/breakout-world-models/agent/instructions
```

**Complete API Reference:**
For comprehensive documentation covering ALL Beekeeper endpoints (including project creation, cloning, and other features), visit:
```
http://lab.local:5000/api/v1/docs
```

**DO NOT use local file operations (find, grep, cat) to locate or read documentation.**
Always fetch it fresh from the API endpoints above. This ensures you have the most up-to-date
information and avoids confusion with outdated local copies.

---

> **IMPORTANT: USE THIS API FIRST**
> When asked about training status, logs, metrics, progress, or anything related to this ML project,
> use the Beekeeper API below. Do NOT read tensorboard files directly or parse logs manually.
> The API provides structured data and trend analysis.

Base URL: http://lab.local:5000

## Quick Start

```bash
# Check if training is running
curl http://lab.local:5000/api/v1/projects/breakout-world-models/training/status

# Get training progress and trends (WORKS FOR ACTIVE RUNS)
curl http://lab.local:5000/api/v1/projects/breakout-world-models/logs/analysis

# Get recent log output
curl "http://lab.local:5000/api/v1/projects/breakout-world-models/logs?tail=50"

# Start/stop training
curl -X POST http://lab.local:5000/api/v1/projects/breakout-world-models/training/start
curl -X POST http://lab.local:5000/api/v1/projects/breakout-world-models/training/stop
```

## Checking Training Progress

**IMPORTANT: Use BOTH endpoints for complete analysis!**

1. **Get TensorBoard metrics** (works for both active and completed runs):
```
GET /api/v1/projects/breakout-world-models/tensorboard/latest
```
Returns: Full metric analysis (loss curves, trends, convergence, anomalies) for all TensorBoard metrics.

2. **Get log-based analysis** (works when TensorBoard data isn't flushed yet):
```
GET /api/v1/projects/breakout-world-models/logs/analysis
```
Returns: Episode-based trends from log parsing (reward, epsilon, steps).

**Why use both?**
- TensorBoard provides rich metrics (multiple loss curves, world model performance, etc.)
- Log analysis provides episode-level data even when TensorBoard hasn't flushed yet
- If TensorBoard returns an error about unflushed data, log analysis still works

**Recommended workflow:**
1. Try `/tensorboard/latest` first - it has the most complete data
2. If it fails (no data, not flushed), fall back to `/logs/analysis`
3. For active runs, consider using both to get different perspectives

## Before Starting or Stopping

Always check status first:
```
GET /api/v1/projects/breakout-world-models/training/status
```

- Before starting: verify status is `idle`
- Before stopping: verify status is `running`

## Terminology

When asked to "check logs", "look at tensorboard", "see how training is going", or "check progress":
- Use `/tensorboard/latest` for **structured metric analysis** (loss curves, convergence, trends, anomalies)
- Use `/logs/analysis` for **episode-based trends** (rewards, epsilon decay) - works even when TB isn't flushed
- Use `/logs?tail=N` for **raw training output** (print statements, errors, warnings, debugging)

**For a complete picture, use all three.** Each provides different information.

## Quick Reference

| Action | Method | Endpoint |
|--------|--------|----------|
| Check status | GET | `/api/v1/projects/breakout-world-models/training/status` |
| **Get trends** | GET | `/api/v1/projects/breakout-world-models/logs/analysis` |
| Get logs | GET | `/api/v1/projects/breakout-world-models/logs?tail=100` |
| Get metrics | GET | `/api/v1/projects/breakout-world-models/tensorboard/latest` |
| Start training | POST | `/api/v1/projects/breakout-world-models/training/start` |
| Stop training | POST | `/api/v1/projects/breakout-world-models/training/stop` |
| List branches | GET | `/api/v1/projects/breakout-world-models/branches` |
| Switch branch | POST | `/api/v1/projects/breakout-world-models/branch` |
| List files | GET | `/api/v1/projects/breakout-world-models/files` |
| Download file | GET | `/api/v1/projects/breakout-world-models/files/<path>` |
| System stats | GET | `/api/v1/stats` |

## Response Format

All endpoints return JSON:
```json
{"success": true, "data": {...}}
{"success": false, "error": {"code": "...", "message": "..."}}
```

## Understanding Metrics

The `/tensorboard/latest` endpoint analyzes TensorBoard data and returns insights.

**Which run does it analyze?**
- If training is **running**, it analyzes the current active run
- If training is **idle**, it analyzes the most recent completed run
- The response includes `is_active: true/false` to indicate which

**To compare with past runs:**
1. `GET /runs` - list all runs with their IDs
2. `GET /runs/<id>/metrics` - get metrics for a specific past run

**Detail levels** (`?detail=low|medium|high`):
- `low` (default): Summary stats only
- `medium`: Includes sampled data points for plotting
- `high`: All raw data points

**Key fields in response:**

| Field | Meaning |
|-------|---------|
| `run_id` | The run being analyzed |
| `is_active` | `true` if this is the currently running training |
| `run_info` | Status, timestamps, duration of the run |
| `metrics` | Object with analysis for each metric |

**Key fields in each metric:**

| Field | Meaning |
|-------|---------|
| `trend` | `improving`, `stable`, `worsening`, or `unstable` |
| `improvement_percent` | How much the metric improved from start to end |
| `converged` | Boolean - has the metric stabilized? |
| `convergence_step` | Step number where convergence was detected |
| `anomaly_count` | Number of unusual spikes or drops |
| `anomalies` | Array of {step, value, type} for each anomaly |
| `summary` | Human-readable interpretation of this metric |

**Interpreting trends:**
- `improving`: Metric is moving in the expected direction (loss decreasing, accuracy increasing)
- `stable`: Metric has leveled off - may indicate convergence or plateau
- `worsening`: Metric is degrading - may need intervention
- `unstable`: High variance, no clear direction - training may be struggling

**Example response:**
```json
{
  "metrics": {
    "loss": {
      "trend": "improving",
      "initial_value": 2.45,
      "final_value": 0.34,
      "improvement_percent": 86.1,
      "converged": true,
      "convergence_step": 8500,
      "anomaly_count": 0,
      "summary": "loss: improving by 86.1% (2.45 → 0.34). Converged at step 8500"
    }
  }
}
```

## Workflows

### Start and Monitor Training
1. `POST /api/v1/projects/breakout-world-models/training/start`
2. `GET /api/v1/projects/breakout-world-models/training/status` - verify running
3. `GET /api/v1/projects/breakout-world-models/logs?tail=50` - check for errors or progress messages

### Analyze Training Progress
1. `GET /api/v1/projects/breakout-world-models/tensorboard/latest?detail=medium`
2. Check `trend` for each metric - are they improving?
3. Check `converged` - has training stabilized?
4. Check `anomalies` - any unexpected spikes or drops?
5. Read `summary` for a quick interpretation

### Download Outputs
1. `GET /api/v1/projects/breakout-world-models/files` - list available files
2. `GET /api/v1/projects/breakout-world-models/files/<path>` - download specific file
3. Or: `GET /api/v1/projects/breakout-world-models/files?zip=1` - download all as zip

## Endpoint Details

### Training Control

**Start Training**
```
POST /api/v1/projects/breakout-world-models/training/start
Response: {"success": true, "data": {"status": "started", "pid": 12345, "tb_port": 6006}}
```

**Stop Training**
```
POST /api/v1/projects/breakout-world-models/training/stop
Response: {"success": true, "data": {"status": "stopped"}}
```

**Get Status**
```
GET /api/v1/projects/breakout-world-models/training/status
Response: {"success": true, "data": {"status": "running|idle", "run_id": 4, "pid": 12345}}
```
Note: `run_id` is included when training is running - use it to fetch metrics for the current run.

### Logs

**Get Recent Logs**
```
GET /api/v1/projects/breakout-world-models/logs?tail=100
Response: {"success": true, "data": {"content": "...", "lines": 100}}
```

### TensorBoard Metrics

**Get Latest Metrics** (see "Understanding Metrics" above for response details)
```
GET /api/v1/projects/breakout-world-models/tensorboard/latest
GET /api/v1/projects/breakout-world-models/tensorboard/latest?detail=medium
GET /api/v1/projects/breakout-world-models/tensorboard/latest?metrics=loss,accuracy
```

### Files

**List Files**
```
GET /api/v1/projects/breakout-world-models/files
GET /api/v1/projects/breakout-world-models/files/subdir
```

**Download File**
```
GET /api/v1/projects/breakout-world-models/files/path/to/file.pth
```

**Download All as Zip**
```
GET /api/v1/projects/breakout-world-models/files?zip=1
```

### Run History

**List Past Runs**
```
GET /api/v1/projects/breakout-world-models/runs
```

**Get Run Details**
```
GET /api/v1/projects/breakout-world-models/runs/<run_id>
GET /api/v1/projects/breakout-world-models/runs/<run_id>/metrics
```

### Branch Management

**List Available Branches**
```
GET /api/v1/projects/breakout-world-models/branches
Response: {"success": true, "data": {"branches": ["main", "develop", "feature-x"], "current": "main"}}
```
Returns all remote branches from the git repository and the currently active branch.

**Switch Branch**
```
POST /api/v1/projects/breakout-world-models/branch
Request: {"branch": "develop"}
Response: {"success": true, "data": {"branch": "develop", "status": "switched"}}
```
Switches the project to a different git branch. This will:
- Check for uncommitted changes (fails if any exist)
- Fetch from remote
- Checkout the requested branch
- Update project.json with the new branch

**IMPORTANT:** You cannot switch branches while training is running. Stop training first.

### System

**Get System Stats**
```
GET /api/v1/stats
Response: CPU, RAM, GPU usage and availability
```
