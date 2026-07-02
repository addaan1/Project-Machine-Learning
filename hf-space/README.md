---
title: EcoDash Indonesia Economic Intelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
short_description: Dashboard ekonomi Indonesia berbasis Django.
---

# EcoDash

EcoDash adalah dashboard analitik ekonomi Indonesia untuk demo final project. Aplikasi ini menjalankan Django melalui Docker, menggunakan artefak model yang sudah tersedia di repositori, dan tidak melakukan retraining di server.

## Environment

Set variable atau secret berikut di Hugging Face Space:

```text
DJANGO_ENV=production
DEBUG=False
ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1,0.0.0.0
CSRF_TRUSTED_ORIGINS=https://*.hf.space
SECRET_KEY=<generate-a-django-secret-key>
```

## Runtime

Container menjalankan:

```text
gunicorn --chdir dashboard dashboard.wsgi:application --bind 0.0.0.0:7860
```

Halaman utama tersedia di `/`, dengan dashboard operasional di `/dashboard/`.
