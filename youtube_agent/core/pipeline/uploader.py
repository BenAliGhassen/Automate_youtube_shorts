"""YouTube upload placeholder — not implemented yet."""
import logging

logger = logging.getLogger(__name__)


def upload_video(job) -> str:
    """Returns the YouTube video ID after upload."""
    logger.info("upload_video called for job %s", str(job.id)[:8])
    raise NotImplementedError("YouTube upload not implemented yet — implement upload_video() here.")