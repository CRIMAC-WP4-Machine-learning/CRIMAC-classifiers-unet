FROM pytorch/pytorch:latest

ARG ssh_prv_key
ARG ssh_pub_key

COPY requirements.txt /tmp/

# From https://stackoverflow.com/a/42125241
# Authorize SSH Host
RUN apt-get update && \
    apt-get install -y \
        git \
        openssh-server

RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permission
RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub

RUN git clone git@github.com:COGMAR/Acoustic-CRIMAC.git /tmp/unet && \
    cd /tmp/unet && \
    git checkout zarr_parquet && \
    mv src-cogmar /crimac_unet && \
    cd /crimac_unet && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    apt-get remove -y \
        git \
        openssh-server && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf /root/.ssh/

COPY *.py /crimac_unet/
COPY setpyenv.json /crimac_unet

WORKDIR /crimac_unet

CMD python mainscript.py 
