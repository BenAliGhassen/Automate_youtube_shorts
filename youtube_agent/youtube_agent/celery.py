import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")

app = Celery("shorts_bot")

# Pull all CELERY_* settings from Django settings.py
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks.py in every installed app
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")