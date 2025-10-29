FROM python:3.11.14-slim

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
	&& apt-get install --no-install-recommends -y build-essential \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
	&& pip install -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]