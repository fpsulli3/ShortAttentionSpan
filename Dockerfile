FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY train.py .
COPY requirements.txt .
COPY model ./model

RUN pip install --upgrade pip \
      && pip install -r requirements.txt

ENTRYPOINT ["python", "train.py"]

