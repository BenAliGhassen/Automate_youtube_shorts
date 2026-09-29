# YouTube Shorts Generator (youtube_agent)

An automated pipeline that generates and publishes **football YouTube Shorts** end to end: topic selection, AI script writing, voiceover, video assembly, quality scoring, and upload.

## Features

- **Content calendar**: a daily football topic schedule drives what gets produced
- **AI script generation** with the Gemini API (two API keys rotated to double the free quota)
- **Text-to-speech voiceover** with `edge-tts`
- **Timeline-based video assembly** with MoviePy (Ken Burns effects, subtitle overlays)
- **Media sourcing** from Giphy, Internet Archive, and Wikimedia
- **Loop-bait structure**: the ending connects back to the opening to boost replays
- **Quality scoring** with a threshold and automatic retry when a video scores too low
- **Automatic upload** through the YouTube Data API v3 (OAuth)
- **Async processing** with Celery + Redis, containerized with Docker

## Pipeline

```
Content calendar
      │
      ▼
Script generation (Gemini)
      │
      ▼
Voiceover (edge-tts)
      │
      ▼
Media sourcing (Giphy / Internet Archive / Wikimedia)
      │
      ▼
Timeline assembly (MoviePy)
      │
      ▼
Quality score ──(below threshold)──▶ retry
      │
      ▼
Upload (YouTube Data API v3)
```

## Tech Stack

- **Backend**: Django
- **Task queue**: Celery + Redis
- **AI**: Gemini API
- **Voice**: edge-tts
- **Video**: MoviePy
- **Infra**: Docker / Docker Compose
- **Publishing**: YouTube Data API v3

## Getting Started

### Prerequisites

- Python `>= 3.10`
- Docker and Docker Compose
- FFmpeg
- Gemini API key(s)
- A Google Cloud project with the YouTube Data API v3 enabled and OAuth credentials

### Installation

```bash
git clone <repo-url>
cd youtube_agent
cp .env.example .env
```

### Configuration

Fill in the `.env` file:

```env
GEMINI_API_KEY_1=your_first_key
GEMINI_API_KEY_2=your_second_key
GIPHY_API_KEY=your_giphy_key
REDIS_URL=redis://redis:6379/0
DJANGO_SECRET_KEY=change_me
```


### Run

```bash
docker compose up --build
```

Then trigger a generation from `<admin panel / management command / API endpoint>`.

## Project Structure

```
youtube_agent/
├── pipeline/       # Script, voice, media, assembly, upload phases
├── calendar/       # Daily topic schedule
├── scoring/        # Quality scoring and retry logic
├── tasks.py        # Celery tasks
├── docker-compose.yml
└── manage.py
```


## Results and Lessons Learned

- The first storytelling-with-images format rarely passed 1,000 views per video.
- Tests with other formats (skills and goals edits) reached **11K–30K views per short** and about **200K total views in 27 days**, which showed the format was the problem, not distribution.
- Automatic clip editing produced low-quality results, and Wikimedia sometimes returned irrelevant images.

## Roadmap

- [ ] Finish this project (currently at 80% no web interface yet)
- [ ] New pipeline built from long football podcasts (1–2h), automatically cut into 1-minute thematic segments
- [ ] Rebuilt stack: React frontend, Spring (Java) backend, Python/Flask for AI processing
- [ ] Better relevance filtering for sourced media

## Author

**Ghassen** — `<GitHub / LinkedIn link>`
