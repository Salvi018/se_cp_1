import os

# Workers: 2 × CPU cores + 1  (keep low on free tier — 1 CPU)
workers    = int(os.environ.get("WEB_CONCURRENCY", 2))
threads    = 2
worker_class = "sync"

# Render/Railway inject PORT env var
bind       = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# ML models take time to load — give enough timeout
timeout    = 120
keepalive  = 5

# Logging
accesslog  = "-"   # stdout
errorlog   = "-"   # stderr
loglevel   = "info"
