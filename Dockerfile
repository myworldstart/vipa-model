FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
WORKDIR /usr/local/app
ADD . .
RUN pip install -r requirements.txt
EXPOSE 3308
ENTRYPOINT flask run

