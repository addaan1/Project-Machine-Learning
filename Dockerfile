FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_ENV=production \
    DEBUG=False \
    ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1,0.0.0.0 \
    CSRF_TRUSTED_ORIGINS=https://*.hf.space,http://localhost:8000,http://127.0.0.1:8000 \
    PORT=7860

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements-deploy.txt

COPY --chown=user . .

RUN python dashboard/manage.py collectstatic --noinput

EXPOSE 7860

CMD ["gunicorn", "--chdir", "dashboard", "dashboard.wsgi:application", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
