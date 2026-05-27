import logging
from celery import shared_task, chain
from django.utils import timezone

logger = logging.getLogger(__name__)


# ── Orchestrator ──────────────────────────────────────────────────────────────

@shared_task(bind=True, name="core.tasks.run_daily_pipeline")
def run_daily_pipeline(self):
    """
    Entry point called by the Celery Beat scheduler once per day.
    Picks the next topic and kicks off the full pipeline chain.
    """
    from .models import VideoJob, Topic

    topic = Topic.get_next()
    logger.info("Starting daily pipeline for topic: %s", topic)

    job = VideoJob.objects.create(
        topic=topic,
        celery_task_id=self.request.id or "",
    )
    job.mark_started()

    # Chain: each task receives job.id and hands it to the next
    pipeline = chain(
        generate_content.s(str(job.id)),
        assemble_video.s(),
        upload_to_youtube.s(),
    )
    pipeline.apply_async()

    return str(job.id)


# ── Stage 1: Content generation ───────────────────────────────────────────────

@shared_task(
    bind=True,
    name="core.tasks.generate_content",
    autoretry_for=(Exception,),
    max_retries=3,
    countdown=60,       # wait 60s before retrying
    default_retry_delay=60,
)
def generate_content(self, job_id: str) -> str:
    """Generate script, TTS audio, and visuals for a VideoJob."""
    from .models import VideoJob
    from .pipeline.script import generate_script
    from .pipeline.tts import generate_audio
    from .pipeline.visuals import fetch_visuals

    job = VideoJob.objects.get(id=job_id)
    logger.info("[%s] Generating content for: %s", job_id[:8], job.topic)

    try:
        # 1. Script
        self.update_state(state="PROGRESS", meta={"step": "script"})
        script_data = generate_script(job.topic)
        job.title = script_data["title"]
        job.script = script_data["script"]
        job.keywords = script_data["keywords"]
        job.save(update_fields=["title", "script"])

        # 2. TTS voiceover
        self.update_state(state="PROGRESS", meta={"step": "tts"})
        audio_path = generate_audio(script_data["script"], job_id)
        job.audio_path = audio_path
        job.save(update_fields=["audio_path"])

        # 3. Visuals
        self.update_state(state="PROGRESS", meta={"step": "visuals"})
        fetch_visuals(job.keywords, job_id)

        logger.info("[%s] Content generation complete", job_id[:8])
        return job_id

    except Exception as exc:
        job.mark_failed(stage="GENERATING", error=str(exc))
        logger.exception("[%s] Content generation failed", job_id[:8])
        raise self.retry(exc=exc)


# ── Stage 2: Video assembly ───────────────────────────────────────────────────

@shared_task(
    bind=True,
    name="core.tasks.assemble_video",
    autoretry_for=(Exception,),
    max_retries=2,
    countdown=30,
)
def assemble_video(self, job_id: str) -> str:
    """Compose audio + visuals into a 9:16 MP4 Short."""
    from .models import VideoJob
    from .pipeline.assembler import build_video

    job = VideoJob.objects.get(id=job_id)
    job.mark_assembling()
    logger.info("[%s] Assembling video", job_id[:8])

    try:
        video_path = build_video(job)
        job.video_path = video_path
        job.save(update_fields=["video_path"])
        logger.info("[%s] Assembly complete: %s", job_id[:8], video_path)
        return job_id

    except Exception as exc:
        job.mark_failed(stage="ASSEMBLING", error=str(exc))
        logger.exception("[%s] Assembly failed", job_id[:8])
        raise self.retry(exc=exc)


# ── Stage 3: YouTube upload ───────────────────────────────────────────────────

@shared_task(
    bind=True,
    name="core.tasks.upload_to_youtube",
    # No autoretry — we never want to accidentally double-upload
    max_retries=0,
)
def upload_to_youtube(self, job_id: str) -> str:
    """Upload the assembled video to YouTube and save the video ID."""
    from .models import VideoJob
    from .pipeline.uploader import upload_video

    job = VideoJob.objects.get(id=job_id)
    job.mark_uploading()
    logger.info("[%s] Uploading to YouTube", job_id[:8])

    try:
        youtube_id = upload_video(job)
        job.mark_done(youtube_video_id=youtube_id)
        logger.info("[%s] Upload complete: https://youtube.com/shorts/%s", job_id[:8], youtube_id)
        return job_id

    except Exception as exc:
        job.mark_failed(stage="UPLOADING", error=str(exc))
        logger.exception("[%s] Upload failed", job_id[:8])
        # Do NOT retry — manual intervention required to avoid double upload
        raise


# ── Manual trigger ────────────────────────────────────────────────────────────

@shared_task(name="core.tasks.trigger_job_for_topic")
def trigger_job_for_topic(topic: str) -> str:
    """Manually trigger a pipeline for a specific topic (used by the dashboard)."""
    from .models import VideoJob

    job = VideoJob.objects.create(topic=topic)
    job.mark_started()

    pipeline = chain(
        generate_content.s(str(job.id)),
        assemble_video.s(),
        upload_to_youtube.s(),
    )
    result = pipeline.apply_async()
    job.celery_task_id = result.id
    job.save(update_fields=["celery_task_id"])
    return str(job.id)