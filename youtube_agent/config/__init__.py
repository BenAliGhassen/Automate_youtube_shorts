try:
    from youtube_agent.celery import app, app as celery_app
except ModuleNotFoundError:
    from youtube_agent.youtube_agent.celery import app, app as celery_app

__all__ = ("app", "celery_app")
