FROM pytorch/pytorch:latest

COPY requirements.min.txt /tmp/
COPY crimac_unet /crimac_unet

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.min.txt

WORKDIR /crimac_unet

CMD python dockerscript.py
