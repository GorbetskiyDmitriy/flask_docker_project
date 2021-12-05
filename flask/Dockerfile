FROM python:3.8.12

ENV HOST 0.0.0.0
ENV PORT 5000
ENV DEBUG true
ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0
ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

COPY . /flask
WORKDIR /flask

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["flask_api.py"]