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
RUN echo "{\"scratch\\": \"/datain/\", " > setpyenv.json
RUN echo ""syspath": "D:\\repos\\Github\\acoustic_private\\"," >> setpyenv.json
RUN echo ""path_to_echograms": "/datain/"}" >> setpyenv.json

# Install elpy
#RUN echo "(require 'package)" >> ~/.emacs
#RUN echo "(add-to-list 'package-archives" >> ~/.emacs
#RUN echo "             '("melpa-stable" . "https://stable.melpa.org/packages/"))" >> ~/.emacs
#RUN echo "(package-initialize)" >> ~/.emacs
#RUN echo "(elpy-enable)" >> ~/.emacs

RUN echo "(require 'package)" > ~/.emacs
RUN echo "(setq package-archives" >> ~/.emacs
RUN echo "      '(("gnu" . "http://elpa.gnu.org/packages/")" >> ~/.emacs
RUN echo "        ("melpa" . "http://melpa.org/packages/")))" >> ~/.emacs
RUN echo "(package-initialize)" >> ~/.emacs
RUN echo "(custom-set-variables" >> ~/.emacs
RUN echo " ;; custom-set-variables was added by Custom." >> ~/.emacs
RUN echo " ;; If you edit it by hand, you could mess it up, so be careful." >> ~/.emacs
RUN echo " ;; Your init file should contain only one such instance." >> ~/.emacs
RUN echo " ;; If there is more than one, they won't work right." >> ~/.emacs
RUN echo " '(package-selected-packages (quote (melpa-upstream-visit))))" >> ~/.emacs
RUN echo "(custom-set-faces" >> ~/.emacs
RUN echo " ;; custom-set-faces was added by Custom." >> ~/.emacs
RUN echo " ;; If you edit it by hand, you could mess it up, so be careful." >> ~/.emacs
RUN echo " ;; Your init file should contain only one such instance." >> ~/.emacs
RUN echo " ;; If there is more than one, they won't work right." >> ~/.emacs
RUN echo " )" >> ~/.emacs

# Clone the libraries
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers-unet
#RUN git clone https://github.com/COGMAR/acoustic_private

CMD zsh
