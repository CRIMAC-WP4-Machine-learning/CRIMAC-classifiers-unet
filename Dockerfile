FROM pytorch/pytorch:latest

# Install linux applications
RUN apt-get update && apt-get install --yes \
    curl \
    zsh \
    emacs \
    git

# Get the ohnyzsh env
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install pip and python packages
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib


# Set path to the echograms

# Install elpy
#RUN echo "(require 'package)" >> ~/.emacs
#RUN echo "(add-to-list 'package-archives" >> ~/.emacs
#RUN echo "             '("melpa-stable" . "https://stable.melpa.org/packages/"))" >> ~/.emacs
#RUN echo "(package-initialize)" >> ~/.emacs
#RUN echo "(elpy-enable)" >> ~/.emacs

# Clone the libraries
RUN cd /
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers-unet
RUN cp /CRIMAC-classifyers-unet/.emacs ~/.emacs




#RUN git clone https://github.com/COGMAR/acoustic_private

CMD zsh
