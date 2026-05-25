from django import template
from django.db.models import Count
from django.db.utils import OperationalError, ProgrammingError

from core.models import Topic, VideoJob

register = template.Library()


@register.inclusion_tag("admin/includes/dashboard_overview.html")
def admin_dashboard_overview():
    try:
        counts = {
            row["status"]: row["total"]
            for row in VideoJob.objects.values("status").annotate(total=Count("id"))
        }
        recent_jobs = VideoJob.objects.order_by("-created_at")[:6]
        total_jobs = sum(counts.values())
        active_topics = Topic.objects.filter(is_active=True).count()
        total_topics = Topic.objects.count()
    except (OperationalError, ProgrammingError):
        counts = {}
        recent_jobs = []
        total_jobs = 0
        active_topics = 0
        total_topics = 0

    status_rows = [
        {
            "code": code.lower(),
            "label": label,
            "count": counts.get(code, 0),
        }
        for code, label in VideoJob.Status.choices
    ]

    in_progress = (
        counts.get(VideoJob.Status.GENERATING, 0)
        + counts.get(VideoJob.Status.ASSEMBLING, 0)
        + counts.get(VideoJob.Status.UPLOADING, 0)
    )

    summary_cards = [
        {
            "label": "Total jobs",
            "count": total_jobs,
            "note": "Every generated Short tracked in one place.",
        },
        {
            "label": "In progress",
            "count": in_progress,
            "note": "Jobs currently moving through the pipeline.",
        },
        {
            "label": "Needs attention",
            "count": counts.get(VideoJob.Status.FAILED, 0),
            "note": "Failed jobs that likely need a retry or fix.",
        },
        {
            "label": "Active topics",
            "count": active_topics,
            "note": f"{total_topics} topics total in the rotation pool.",
        },
    ]

    return {
        "summary_cards": summary_cards,
        "status_rows": status_rows,
        "recent_jobs": recent_jobs,
    }
