FROM python:3.8-buster
RUN pip install pyyaml requests dapr dapr.ext.grpc minio cloudevents hera-workflows replicate
COPY . .
CMD ["python", "dreamer.py"]