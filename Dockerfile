FROM pytorch/pytorch:latest

COPY requirements.min.txt /tmp/
COPY crimac_unet /crimac_unet

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.min.txt

COPY mainscript.py setpaths.py /crimac_unet/
COPY setpyenv.json /crimac_unet

WORKDIR /crimac_unet

CMD python mainscript.py 
