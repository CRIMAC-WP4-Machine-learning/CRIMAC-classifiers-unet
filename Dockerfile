FROM pytorch/pytorch:latest

# Install linux applications
#RUN apt-get update && apt-get install --yes \
#    curl \
#    zsh \
#    emacs \
#    git

# Install pip and python packages
#RUN pip install --upgrade pip
#RUN pip install numpy scipy matplotlib

# Clone the libraries & copy config
#RUN cd /
#RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers-unet /CRIMAC-classifyers-unet

# Get the ohnyzsh env
#RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" -Y

# RUN git clone https://github.com/COGMAR/acoustic_private /acosutic_private
# RUN cp /CRIMAC-classifyers-unet/setpyenv.json /acosutic_private
#RUN cp /CRIMAC-classifyers-unet/.emacs ~
#CMD zsh
CMD bash

