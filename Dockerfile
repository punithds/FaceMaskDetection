FROM python:3.7
WORKDIR /code
RUN apt-get update
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY . .
CMD ["python","./face.py"]
