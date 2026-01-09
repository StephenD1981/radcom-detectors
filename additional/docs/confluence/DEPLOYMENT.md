# Deployment Guide

## Who Is This Document For?

This guide is for **operations teams** who need to:
- Run the RAN Optimizer on a schedule (e.g., daily)
- Set up the tool in a production environment
- Host the dashboard for team access
- Monitor the tool's health

If you're just testing or running manually, see [INSTALLATION] instead.

---

## Deployment Options

### Option 1: Direct Python (Simplest)

**Best for:** Development, testing, single-machine deployments

{code:bash}
# Install once
pip install .

# Run manually or via cron
ran-optimize --input-dir /data/input --output-dir /data/output
{code}

### Option 2: Docker (Recommended for Production)

**Best for:** Reproducibility, isolation, cloud deployments

{code:bash}
# Build the image
docker build -t ran-optimizer:latest .

# Run with your data mounted
docker run \
  -v /path/to/input:/app/data/input:ro \
  -v /path/to/output:/app/data/output \
  -v /path/to/config:/app/config:ro \
  ran-optimizer:latest
{code}

### Option 3: Docker Compose (Full Stack)

**Best for:** Running optimizer + dashboard together

{code:bash}
# Start everything
docker-compose up -d

# Dashboard available at http://localhost:8080/enhanced_dashboard.html

# Stop when done
docker-compose down
{code}

---

## Recommended Folder Structure

Set up your production environment like this:

{code:none}
/opt/ran-optimizer/
├── bin/
│   └── run-optimizer.sh      ← Script to run the tool
├── config/
│   ├── defaults/
│   │   └── default.yaml      ← Base configuration
│   └── operators/
│       └── my_operator.yaml  ← Your configuration
├── data/
│   ├── input/                ← Input files (read-only)
│   │   ├── cell_coverage.csv
│   │   ├── gis.csv
│   │   └── cell_hulls.csv
│   └── output/               ← Results (writable)
│       ├── *.csv
│       ├── *.geojson
│       └── maps/
│           └── enhanced_dashboard.html
├── logs/                     ← Log files
│   └── ran-optimizer.log
└── venv/                     ← Python virtual environment
{code}

### Setting Up Permissions

{code:bash}
# Input data should be read-only (prevent accidental changes)
chmod -R 444 /opt/ran-optimizer/data/input/

# Output needs write access
chmod 755 /opt/ran-optimizer/data/output/

# Create a service account
useradd -r -s /bin/false ran-optimizer
chown -R ran-optimizer:ran-optimizer /opt/ran-optimizer/
{code}

---

## Running on a Schedule

### Option A: Cron Job (Traditional)

**Step 1:** Create the runner script

Save this as `/opt/ran-optimizer/bin/run-optimizer.sh`:

{code:bash}
#!/bin/bash
set -e

# Configuration
export DATA_ROOT=/opt/ran-optimizer
export LOG_LEVEL=INFO

# Activate Python environment
source /opt/ran-optimizer/venv/bin/activate

# Run the optimizer
ran-optimize \
  --input-dir ${DATA_ROOT}/data/input \
  --output-dir ${DATA_ROOT}/data/output \
  --config-dir ${DATA_ROOT}/config

# Optional: Copy dashboard to web server
cp ${DATA_ROOT}/data/output/maps/enhanced_dashboard.html /var/www/html/ran/

echo "RAN Optimizer completed at $(date)"
{code}

Make it executable:
{code:bash}
chmod +x /opt/ran-optimizer/bin/run-optimizer.sh
{code}

**Step 2:** Add to cron

{code:bash}
# Open cron editor
crontab -e

# Add this line to run daily at 2 AM
0 2 * * * /opt/ran-optimizer/bin/run-optimizer.sh >> /var/log/ran-optimizer.log 2>&1
{code}

**Understanding the cron syntax:**
{code:none}
0 2 * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, * = every day)
│ │ │ └───── Month (1-12, * = every month)
│ │ └─────── Day of month (1-31, * = every day)
│ └───────── Hour (0-23, 2 = 2 AM)
└─────────── Minute (0-59, 0 = on the hour)
{code}

### Option B: Systemd Timer (Modern Linux)

**Step 1:** Create the service file

Save as `/etc/systemd/system/ran-optimizer.service`:

{code:none}
[Unit]
Description=RAN Optimizer
After=network.target

[Service]
Type=oneshot
User=ran-optimizer
ExecStart=/opt/ran-optimizer/bin/run-optimizer.sh
StandardOutput=append:/var/log/ran-optimizer.log
StandardError=append:/var/log/ran-optimizer.log

[Install]
WantedBy=multi-user.target
{code}

**Step 2:** Create the timer file

Save as `/etc/systemd/system/ran-optimizer.timer`:

{code:none}
[Unit]
Description=Run RAN Optimizer daily

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
{code}

**Step 3:** Enable and start

{code:bash}
# Reload systemd
systemctl daemon-reload

# Enable the timer (survives reboot)
systemctl enable ran-optimizer.timer

# Start the timer
systemctl start ran-optimizer.timer

# Check status
systemctl status ran-optimizer.timer
{code}

---

## Data Pipeline Integration

### Refreshing Input Data

The optimizer expects fresh data. Connect it to your data pipeline:

**Option A: Sync from cloud storage**
{code:bash}
# Before running optimizer
aws s3 sync s3://your-bucket/ran-data/latest/ /opt/ran-optimizer/data/input/
{code}

**Option B: Export from database**
{code:bash}
# Custom script to query your data warehouse
python /opt/scripts/extract_ran_data.py \
  --output /opt/ran-optimizer/data/input/
{code}

### Exporting Results

After the optimizer runs:

**Option A: Upload to cloud**
{code:bash}
# Upload results with datestamp
aws s3 cp /opt/ran-optimizer/data/output/ \
  s3://your-bucket/ran-results/$(date +%Y-%m-%d)/ \
  --recursive
{code}

**Option B: Load to database**
{code:bash}
# Custom script to load results
python /opt/scripts/load_results.py \
  --input /opt/ran-optimizer/data/output/
{code}

**Option C: Send to API**
{code:bash}
# POST to an external system
curl -X POST https://your-api.com/ran/results \
  -H "Content-Type: application/json" \
  -d @/opt/ran-optimizer/data/output/summary.json
{code}

---

## Hosting the Dashboard

The dashboard is a self-contained HTML file that can be served any way you serve static files.

### Option A: Nginx (Simple)

Add to your Nginx config:

{code:none}
# /etc/nginx/sites-available/ran-dashboard
server {
    listen 80;
    server_name ran-dashboard.yourcompany.com;

    root /opt/ran-optimizer/data/output/maps;
    index enhanced_dashboard.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
{code}

Enable and restart:
{code:bash}
ln -s /etc/nginx/sites-available/ran-dashboard /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
{code}

### Option B: S3 Static Website

{code:bash}
# Upload dashboard to S3
aws s3 cp \
  /opt/ran-optimizer/data/output/maps/enhanced_dashboard.html \
  s3://your-dashboard-bucket/enhanced_dashboard.html \
  --content-type "text/html"

# Configure bucket for static website hosting via AWS Console
{code}

### Option C: Python Simple Server (Development Only)

{code:bash}
# Quick test (NOT for production)
cd /opt/ran-optimizer/data/output/maps
python -m http.server 8080

# Dashboard at http://localhost:8080/enhanced_dashboard.html
{code}

---

## Monitoring

### Health Check Script

Create `/opt/ran-optimizer/bin/health-check.sh`:

{code:bash}
#!/bin/bash
OUTPUT_DIR=/opt/ran-optimizer/data/output
MAX_AGE_HOURS=26  # Daily run + 2 hour buffer

# Check each expected output file
for file in overshooting_cells_environment_aware.csv undershooting_cells_environment_aware.csv; do
    FILE_PATH="${OUTPUT_DIR}/${file}"

    # Check if file exists
    if [ ! -f "$FILE_PATH" ]; then
        echo "CRITICAL: ${file} not found"
        exit 2
    fi

    # Check if file is recent
    AGE_SECONDS=$(($(date +%s) - $(stat -c %Y "$FILE_PATH")))
    AGE_HOURS=$((AGE_SECONDS / 3600))

    if [ $AGE_HOURS -gt $MAX_AGE_HOURS ]; then
        echo "WARNING: ${file} is ${AGE_HOURS} hours old (max: ${MAX_AGE_HOURS})"
        exit 1
    fi
done

echo "OK: All outputs are current"
exit 0
{code}

Use with your monitoring system (Nagios, Prometheus, etc.):
{code:bash}
# Returns:
# Exit 0 = OK
# Exit 1 = WARNING
# Exit 2 = CRITICAL
{code}

### Key Log Messages to Monitor

Set up alerts for these patterns:

| Pattern | Severity | Meaning |
|---------|----------|---------|
| `EXECUTION COMPLETE` | Info | Success |
| `error` | Error | Something failed |
| `MemoryError` | Critical | Out of RAM |
| `FileNotFoundError` | Critical | Missing input |
| `timeout` | Warning | Taking too long |

### Metrics to Track

| Metric | Normal Range | Alert If |
|--------|--------------|----------|
| Execution time | 5-30 minutes | > 60 minutes |
| Overshooting count | Varies by network | Change > 20% |
| Undershooting count | Varies by network | Change > 20% |
| Output file age | < 26 hours | > 26 hours |

---

## Scaling for Large Datasets

### Memory Management

For datasets with millions of rows:

**Option 1: Reduce chunk size**
{code:yaml}
# In your config file
processing:
  chunk_size: 50000  # Default is 100000
{code}

**Option 2: Process regions separately**
{code:bash}
# Split data by region, process each
for region in region1 region2 region3; do
  ran-optimize \
    --input-dir data/${region}/input \
    --output-dir data/${region}/output
done
{code}

### Parallel Processing

Run multiple regions simultaneously:

{code:bash}
# Using GNU parallel
parallel -j 4 \
  ran-optimize --input-dir data/{}/input --output-dir data/{}/output \
  ::: region1 region2 region3 region4
{code}

---

## Backup and Recovery

### Regular Backups

{code:bash}
# Backup outputs (run after each execution)
tar -czf /backups/ran-$(date +%Y%m%d).tar.gz \
  /opt/ran-optimizer/data/output/

# Keep 30 days of backups
find /backups/ -name "ran-*.tar.gz" -mtime +30 -delete
{code}

### Version Control for Configs

{code:bash}
# Keep configs in git
cd /opt/ran-optimizer/config
git init
git add .
git commit -m "Initial config"

# After any change
git add -A
git commit -m "Config update: $(date +%Y-%m-%d)"
{code}

---

## Security Checklist

### File Permissions

{code:bash}
# Config files (sensitive parameters)
chmod 600 /opt/ran-optimizer/config/*.yaml
chmod 600 /opt/ran-optimizer/config/*.json

# Input data (read-only)
chmod -R 444 /opt/ran-optimizer/data/input/

# Output data (writable by service only)
chmod 755 /opt/ran-optimizer/data/output/
chown ran-optimizer:ran-optimizer /opt/ran-optimizer/data/output/
{code}

### Network Security

- Dashboard should be behind authentication (not public)
- Input/output directories should NOT be web-accessible
- Use HTTPS for dashboard hosting
- Consider IP whitelisting for sensitive data

### Service Account

{code:bash}
# Create dedicated user (no login shell)
useradd -r -s /bin/false ran-optimizer

# Grant only necessary permissions
chown -R ran-optimizer:ran-optimizer /opt/ran-optimizer/
{code}

---

## Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Output files not created | Missing input data | Check data pipeline ran first |
| MemoryError | Dataset too large | Reduce chunk_size or add RAM |
| Permission denied | Wrong ownership | Check file permissions |
| Dashboard shows old data | Scheduler not running | Check cron/systemd logs |
| Blank dashboard | JavaScript error | Check browser console |

### Debug Mode

Run with verbose logging:

{code:bash}
LOG_LEVEL=DEBUG ran-optimize \
  --input-dir data/input \
  --output-dir data/output \
  2>&1 | tee debug.log
{code}

### Check Scheduler Status

**For cron:**
{code:bash}
# Check if job ran
grep ran-optimizer /var/log/syslog

# Check cron is running
systemctl status cron
{code}

**For systemd:**
{code:bash}
# Check timer status
systemctl status ran-optimizer.timer

# Check last execution
journalctl -u ran-optimizer.service
{code}

---

## Quick Reference

### Daily Operations

| Task | Command |
|------|---------|
| Run manually | `ran-optimize --input-dir data/input --output-dir data/output` |
| Check last run | `ls -la /opt/ran-optimizer/data/output/*.csv` |
| View logs | `tail -100 /var/log/ran-optimizer.log` |
| Check timer | `systemctl status ran-optimizer.timer` |

### Common Paths

| Path | Contents |
|------|----------|
| `/opt/ran-optimizer/data/input/` | Input files |
| `/opt/ran-optimizer/data/output/` | Results |
| `/opt/ran-optimizer/config/` | Configuration |
| `/var/log/ran-optimizer.log` | Logs |

---

## Related Documentation

- [INSTALLATION] - Initial setup
- [CONFIGURATION] - Parameter tuning
- [DATA_FORMATS] - Data requirements

