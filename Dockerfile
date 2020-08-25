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


#{"scratch": "D:\\DATAscratch\\deep\\echosounder\\akustikk_all\\data\\",
# "syspath": "D:\\repos\\Github\\acoustic_private\\",
#"path_to_echograms": "D:\\DATAscratch\\deep\\echosounder\\akustikk_all\\data\\North Sea NOR Sandeel cruise in Apr_May\\memmap\\"}


# Install elpy
RUN echo "(require 'package)" >> ~/.emacs
RUN echo "(add-to-list 'package-archives" >> ~/.emacs
RUN echo "             '("melpa-stable" . "https://stable.melpa.org/packages/"))" >> ~/.emacs
RUN echo "(package-initialize)" >> ~/.emacs
RUN echo "(elpy-enable)" >> ~/.emacs



# Clone the preprocessing library
# RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers/unet

CMD zsh
