"""Video assembler — stub for Phase 3."""
import logging

logger = logging.getLogger(__name__)


def build_video(job) -> str:
    """Returns the path to the rendered MP4 file."""
    logger.info("build_video called for job %s", str(job.id)[:8])
    raise NotImplementedError("Video assembly not implemented yet — see Phase 3")