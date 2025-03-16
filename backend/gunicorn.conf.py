import json
import os

# Defaults from https://cloud.google.com/run/docs/tips/python#optimize_gunicorn
DEFAULT_GUNICORN_CONFIG = {
    "port": "8080",
    "host": "0.0.0.0",
    "workers": 1,
    "threads": 8,
    "timeout": 0,
    "graceful_timeout": 30,
    "keepalive": 0,
    "loglevel": "info",
    "accesslog": "-",
    "errorlog": "-",
    "worker_tmp_dir": "/dev/shm",
}

# Logging
# https://docs.gunicorn.org/en/stable/settings.html#logging
loglevel = os.getenv("LOG_LEVEL", DEFAULT_GUNICORN_CONFIG["loglevel"])
accesslog = os.getenv("ACCESS_LOG", DEFAULT_GUNICORN_CONFIG["accesslog"])
errorlog = os.getenv("ERROR_LOG", DEFAULT_GUNICORN_CONFIG["errorlog"])

# Worker processes
# https://docs.gunicorn.org/en/stable/settings.html#worker-processes
workers = int(os.getenv("WORKERS", DEFAULT_GUNICORN_CONFIG["workers"]))
threads = int(os.getenv("THREADS", DEFAULT_GUNICORN_CONFIG["threads"]))
timeout = int(os.getenv("TIMEOUT", DEFAULT_GUNICORN_CONFIG["timeout"]))
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", DEFAULT_GUNICORN_CONFIG["graceful_timeout"]))
keepalive = int(os.getenv("KEEP_ALIVE", DEFAULT_GUNICORN_CONFIG["keepalive"]))

# Server machanics
# https://docs.gunicorn.org/en/stable/settings.html?highlight=worker_tmp_dir#worker-tmp-dir
# This is set to /dev/shm to speed up the startup of workers by using a in memory file system
worker_tmp_dir = str(os.getenv("WORKER_TMP_DIR", DEFAULT_GUNICORN_CONFIG["worker_tmp_dir"]))

# Server socket
# https://docs.gunicorn.org/en/stable/settings.html?highlight=bind#bind
port = os.getenv("PORT", DEFAULT_GUNICORN_CONFIG["port"])
host = os.getenv("HOST", DEFAULT_GUNICORN_CONFIG["host"])
bind = ":".join([host, port])

reload = False

log_data = {
    "bind": bind,
    "workers": workers,
    "threads": threads,
    "timeout": timeout,
    "graceful_timeout": graceful_timeout,
    "keepalive": keepalive,
    "loglevel": loglevel,
    "errorlog": errorlog,
    "accesslog": accesslog,
    "worker_tmp_dir": worker_tmp_dir,
}
print(json.dumps(log_data))