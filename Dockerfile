FROM pytorch/pytorch:latest

# Install linux applications
RUN apt-get update && apt-get install --yes \
    curl \
    zsh \
    emacs \
    git

# Install pip and python packages
RUN pip install --upgrade pip
#RUN pip install numpy scipy matplotlib pdb google h5py scikit-image pandas
#RUN pip install numpy scipy matplotlib google h5py scikit-image pandas sklearn xarray dask zarr
# Clone the libraries & copy config
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers-unet /CRIMAC-classifyers-unet
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-annotationtools

# Get the ohmyzsh env
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended


# copy the setpyenv.json to /Acoustic-CRIMAC/src-cogmar

RUN cp /CRIMAC-classifyers-unet/.emacs ~
# RUN git clone https://github.com/COGMAR/Acoustic-CRIMAC
# RUN pip install -r /Acoustic-CRIMAC/src-cogmar/requirements.txt
# RUN cp /CRIMAC-classifyers-unet/setpyenv.json /acoustic_private
CMD zsh


