FROM pytorch/pytorch:latest

# Install linux applications
RUN apt-get update && apt-get install --yes \
    curl \
    zsh \
    emacs \
    git

# Install pip and python packages
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib pdb google h5py scikit-image
# Clone the libraries & copy config
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers-unet /CRIMAC-classifyers-unet

# Get the ohmyzsh env
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# RUN cp /CRIMAC-classifyers-unet/.emacs ~
# RUN git clone https://github.com/COGMAR/acoustic_private /acosutic_private
# RUN cp /CRIMAC-classifyers-unet/setpyenv.json /acosutic_private
CMD zsh


