wsgi_app = "wsgi:application"
loglevel= "debug"
workers = 3
bind = "0.0.0.0:5000"
reload = True
accesslog = errorlog = "/var/log/gunicorn/dev.log"
capture_output = True
pidfile = "/var/log/gunicorn/dev.pid"
daemon = True
timeout = 120
worker_class = "gevent"
preload_app = True