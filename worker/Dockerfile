FROM python:3.8.12

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0
ENV MONGO_LOG log
ENV MONGO_PASS pass
ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

COPY . /worker
WORKDIR /worker

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT celery -A task_manager worker --loglevel=info