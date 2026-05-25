from pathlib import Path

from django.conf import settings
from django.contrib import admin
from django.utils.html import format_html, format_html_join
from django.utils.text import Truncator

from .models import Topic, VideoJob

admin.site.site_header = "Shorts Studio"
admin.site.site_title = "Shorts Studio Admin"
admin.site.index_title = "Pipeline dashboard"
admin.site.empty_value_display = "-"


def _asset_pill(label: str, is_ready: bool) -> str:
    state = "ready" if is_ready else "waiting"
    text = "Ready" if is_ready else "Waiting"
    return format_html(
        '<span class="admin-asset-pill admin-asset-pill--{}">{}: {}</span>',
        state,
        label,
        text,
    )


@admin.register(VideoJob)
class VideoJobAdmin(admin.ModelAdmin):
    list_display = [
        "short_id",
        "headline",
        "status_badge",
        "asset_summary",
        "created_at",
        "completed_at",
        "youtube_link",
    ]
    list_display_links = ["short_id", "headline"]
    list_filter = ["status", "created_at", "completed_at"]
    search_fields = ["id", "topic", "title", "celery_task_id", "youtube_video_id"]
    search_help_text = "Search by job ID, topic, title, Celery task ID, or YouTube video ID."
    readonly_fields = [
        "id",
        "status_badge",
        "pipeline_summary",
        "script_preview",
        "audio_file_link",
        "video_file_link",
        "youtube_link_detail",
        "celery_task_id",
        "created_at",
        "started_at",
        "completed_at",
        "error_log_preview",
    ]
    ordering = ["-created_at"]
    date_hierarchy = "created_at"
    list_per_page = 25
    save_on_top = True

    fieldsets = [
        (
            "Job overview",
            {
                "description": "Use this page to inspect pipeline state and review the generated assets for a single Short.",
                "fields": [
                    "id",
                    "status_badge",
                    "topic",
                    "title",
                    "pipeline_summary",
                    "celery_task_id",
                ],
            },
        ),
        (
            "Generated content",
            {
                "classes": ["wide"],
                "fields": [
                    "script",
                    "script_preview",
                    "audio_file_link",
                    "video_file_link",
                    "youtube_link_detail",
                ],
            },
        ),
        (
            "Errors",
            {
                "classes": ["collapse"],
                "fields": ["error_stage", "error_log_preview"],
            },
        ),
        (
            "Timeline",
            {
                "fields": ["created_at", "started_at", "completed_at"],
            },
        ),
    ]

    @admin.display(description="Job ID")
    def short_id(self, obj: VideoJob) -> str:
        return str(obj.id).split("-")[0]

    @admin.display(description="Title")
    def headline(self, obj: VideoJob) -> str:
        title = obj.title or "Title pending"
        topic = Truncator(obj.topic).chars(64)
        return format_html(
            '<div class="admin-headline"><strong>{}</strong><span>{}</span></div>',
            title,
            topic,
        )

    @admin.display(description="Status")
    def status_badge(self, obj: VideoJob) -> str:
        return format_html(
            '<span class="admin-status-badge admin-status-badge--{}">{}</span>',
            obj.status.lower(),
            obj.get_status_display(),
        )

    @admin.display(description="Assets")
    def asset_summary(self, obj: VideoJob) -> str:
        assets = [
            ("Script", bool(obj.script)),
            ("Audio", bool(obj.audio_path)),
            ("Video", bool(obj.video_path)),
            ("Upload", bool(obj.youtube_video_url)),
        ]
        return format_html(
            '<div class="admin-pill-row">{}</div>',
            format_html_join(
                "",
                "{}",
                ((_asset_pill(label, is_ready),) for label, is_ready in assets),
            ),
        )

    @admin.display(description="Pipeline snapshot")
    def pipeline_summary(self, obj: VideoJob) -> str:
        items = [
            ("Script drafted", bool(obj.script)),
            ("Voiceover rendered", bool(obj.audio_path)),
            ("Video assembled", bool(obj.video_path)),
            ("Published to YouTube", bool(obj.youtube_video_url)),
        ]
        return format_html(
            '<div class="admin-pill-row admin-pill-row--detail">{}</div>',
            format_html_join(
                "",
                "{}",
                ((_asset_pill(label, is_ready),) for label, is_ready in items),
            ),
        )

    @admin.display(description="Script preview")
    def script_preview(self, obj: VideoJob) -> str:
        text = obj.script or "No script has been generated yet."
        return format_html('<div class="admin-preview">{}</div>', text)

    def _media_link(self, file_path: str) -> str | None:
        if not file_path:
            return None

        try:
            relative_path = Path(file_path).resolve().relative_to(Path(settings.MEDIA_ROOT).resolve())
        except (OSError, RuntimeError, ValueError):
            return None

        return f"{settings.MEDIA_URL.rstrip('/')}/{relative_path.as_posix()}"

    def _file_link_block(self, label: str, file_path: str) -> str:
        if not file_path:
            return format_html('<span class="admin-muted">{} not generated yet.</span>', label)

        file_exists = Path(file_path).exists()
        media_url = self._media_link(file_path)
        action = format_html(
            '<span class="admin-muted">{}</span>',
            "Saved on disk",
        )
        if file_exists and media_url:
            action = format_html(
                '<a class="button admin-inline-button" href="{}" target="_blank">Open {}</a>',
                media_url,
                label.lower(),
            )

        return format_html(
            '<div class="admin-link-list">{}<code>{}</code></div>',
            action,
            file_path,
        )

    @admin.display(description="Audio file")
    def audio_file_link(self, obj: VideoJob) -> str:
        return self._file_link_block("Audio", obj.audio_path)

    @admin.display(description="Video file")
    def video_file_link(self, obj: VideoJob) -> str:
        return self._file_link_block("Video", obj.video_path)

    @admin.display(description="YouTube")
    def youtube_link(self, obj: VideoJob) -> str:
        if not obj.youtube_video_url:
            return format_html('<span class="admin-muted">{}</span>', "Not published")

        return format_html(
            '<a class="button admin-inline-button" href="{}" target="_blank">Watch Short</a>',
            obj.youtube_video_url,
        )

    @admin.display(description="Published result")
    def youtube_link_detail(self, obj: VideoJob) -> str:
        if not obj.youtube_video_url:
            return format_html(
                '<span class="admin-muted">{}</span>',
                "No YouTube URL saved yet.",
            )

        return format_html(
            '<div class="admin-link-list">'
            '<a class="button admin-inline-button" href="{}" target="_blank">Open on YouTube</a>'
            "<code>{}</code>"
            "</div>",
            obj.youtube_video_url,
            obj.youtube_video_url,
        )

    @admin.display(description="Error log")
    def error_log_preview(self, obj: VideoJob) -> str:
        text = obj.error_log or "No error log captured."
        return format_html('<pre class="admin-pre">{}</pre>', text)


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = [
        "text",
        "is_active",
        "status_chip",
        "use_count",
        "last_used",
        "created_at",
    ]
    list_filter = ["is_active"]
    search_fields = ["text"]
    search_help_text = "Search the topic library by phrase or keyword."
    list_editable = ["is_active"]
    readonly_fields = ["use_count", "created_at", "last_used"]
    ordering = ["last_used", "use_count"]
    list_per_page = 25

    fieldsets = [
        (
            "Topic details",
            {
                "fields": ["text", "is_active"],
                "description": "Keep this pool focused on topics you actively want the scheduler to rotate through.",
            },
        ),
        (
            "Usage history",
            {
                "fields": ["use_count", "last_used", "created_at"],
            },
        ),
    ]

    actions = ["mark_active", "mark_inactive"]

    @admin.display(description="Status", ordering="is_active")
    def status_chip(self, obj: Topic) -> str:
        tone = "ready" if obj.is_active else "waiting"
        text = "Active" if obj.is_active else "Paused"
        return format_html(
            '<span class="admin-status-badge admin-status-badge--{}">{}</span>',
            tone,
            text,
        )

    @admin.action(description="Mark selected topics as active")
    def mark_active(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f"{updated} topic(s) marked active.")

    @admin.action(description="Pause selected topics")
    def mark_inactive(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} topic(s) paused.")
