FROM python:3.7

RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY fork-zeroer .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -r fork-zeroer/requirements.txt

# this should be after pip install, to cache layers more efficient
COPY entrypoint.py .
COPY transform.py .

# This allows to run the container as `docker run --rm <yourImageName> <args>`
ENTRYPOINT ["python", "-u", "entrypoint.py"]
CMD []
