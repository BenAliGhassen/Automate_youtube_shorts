try:
    from youtube_agent.celery import app
except ModuleNotFoundError:
    from youtube_agent.youtube_agent.celery import app

__all__ = ("app",)
