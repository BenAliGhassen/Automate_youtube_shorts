"""
Management command: verify the project setup is correct.
Run with: python manage.py verify_setup
"""
import sys
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Verify Django, Celery, Redis, and env config are all working."

    def handle(self, *args, **options):
        errors = []

        # 1. Check Django settings load
        self.stdout.write("Checking Django settings...")
        try:
            from django.conf import settings
            assert settings.SECRET_KEY
            self.stdout.write(self.style.SUCCESS("  Django settings OK"))
        except Exception as e:
            errors.append(f"Django settings: {e}")
            self.stdout.write(self.style.ERROR(f"  FAIL: {e}"))

        # 2. Check Redis connection
        self.stdout.write("Checking Redis connection...")
        try:
            import redis
            from django.conf import settings as s
            r = redis.from_url(s.REDIS_URL)
            r.ping()
            self.stdout.write(self.style.SUCCESS(f"  Redis OK ({s.REDIS_URL})"))
        except Exception as e:
            errors.append(f"Redis: {e}")
            self.stdout.write(self.style.ERROR(f"  FAIL: {e}"))

        # 3. Check DB migrations
        self.stdout.write("Checking database...")
        try:
            from core.models import VideoJob, Topic
            VideoJob.objects.count()
            Topic.objects.count()
            self.stdout.write(self.style.SUCCESS("  Database OK"))
        except Exception as e:
            errors.append(f"Database: {e}")
            self.stdout.write(self.style.ERROR(f"  FAIL: {e} — did you run migrate?"))

        # 4. Check Celery app import
        self.stdout.write("Checking Celery app...")
        try:
            from youtube_agent.celery import app
            assert app.main == "shorts_bot"
            self.stdout.write(self.style.SUCCESS("  Celery app OK"))
        except Exception as e:
            errors.append(f"Celery: {e}")
            self.stdout.write(self.style.ERROR(f"  FAIL: {e}"))

        # 5. Check API keys present (warns, not errors)
        self.stdout.write("Checking API keys...")
        from django.conf import settings as s
        for key, attr in [("Gemini", "GEMINI_API_KEY"), ("Pexels", "PEXELS_API_KEY")]:
            val = getattr(s, attr, "")
            if val:
                self.stdout.write(self.style.SUCCESS(f"  {key} key present"))
            else:
                self.stdout.write(self.style.WARNING(f"  {key} key NOT set (needed for Phase 2)"))

        # Summary
        self.stdout.write("")
        if errors:
            self.stdout.write(self.style.ERROR(f"Setup incomplete — {len(errors)} error(s) above."))
            sys.exit(1)
        else:
            self.stdout.write(self.style.SUCCESS("All checks passed. Setup complete!"))