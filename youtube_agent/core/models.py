import uuid
from django.db import models
from django.utils import timezone


class VideoJob(models.Model):

    class Status(models.TextChoices):
        PENDING    = "PENDING",    "Pending"
        GENERATING = "GENERATING", "Generating content"
        ASSEMBLING = "ASSEMBLING", "Assembling video"
        UPLOADING  = "UPLOADING",  "Uploading to YouTube"
        DONE       = "DONE",       "Done"
        FAILED     = "FAILED",     "Failed"

    # Identity
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    topic      = models.CharField(max_length=255)

    # Status
    status     = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True)
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Generated content
    title      = models.CharField(max_length=255, blank=True)
    script     = models.TextField(blank=True)
    audio_path = models.CharField(max_length=500, blank=True)
    video_path = models.CharField(max_length=500, blank=True)
    scenes = models.JSONField(default=list, blank=True)
    thumbnail_path = models.CharField(max_length=500, blank=True)
    
    # YouTube
    youtube_video_id  = models.CharField(max_length=50, blank=True)
    youtube_video_url = models.URLField(blank=True)

    # Error tracking
    error_stage   = models.CharField(max_length=50, blank=True)   # which stage failed
    error_log     = models.TextField(blank=True)

    # Timestamps
    created_at    = models.DateTimeField(default=timezone.now)
    started_at    = models.DateTimeField(null=True, blank=True)
    completed_at  = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Video job"
        verbose_name_plural = "Video jobs"

    def __str__(self):
        return f"[{self.status}] {self.topic[:60]}"

    # ── Convenience helpers ───────────────────────────────────────────────────

    def mark_started(self):
        self.started_at = timezone.now()
        self.status = self.Status.GENERATING
        self.save(update_fields=["started_at", "status"])

    def mark_assembling(self):
        self.status = self.Status.ASSEMBLING
        self.save(update_fields=["status"])

    def mark_uploading(self):
        self.status = self.Status.UPLOADING
        self.save(update_fields=["status"])

    def mark_done(self, youtube_video_id: str):
        self.status = self.Status.DONE
        self.youtube_video_id = youtube_video_id
        self.youtube_video_url = f"https://youtube.com/shorts/{youtube_video_id}"
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "youtube_video_id", "youtube_video_url", "completed_at"])

    def mark_failed(self, stage: str, error: str):
        self.status = self.Status.FAILED
        self.error_stage = stage
        self.error_log = error
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_stage", "error_log", "completed_at"])


class Topic(models.Model):
    """Rotating pool of topics for the scheduler to pick from."""
    text        = models.CharField(max_length=255, unique=True)
    is_active   = models.BooleanField(default=True)
    last_used   = models.DateTimeField(null=True, blank=True)
    use_count   = models.PositiveIntegerField(default=0)
    created_at  = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["last_used", "use_count"]   # least-recently-used first

    def __str__(self):
        return self.text

    @classmethod
    def get_next(cls):
        """Return the least-recently-used active topic."""
        topic = cls.objects.filter(is_active=True).first()
        if not topic:
            raise ValueError("No active topics in the database. Add some via the admin.")
        topic.last_used = timezone.now()
        topic.use_count += 1
        topic.save(update_fields=["last_used", "use_count"])
        return topic.text