FROM pytorch/pytorch:latest

# Install linux applications
RUN apt-get update && apt-get install --yes \
    curl \
    zsh \
    emacs \
    git

RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install pip and python packages
RUN pip install --upgrade pip
RUN pip install numpy scipy

# Clone the preprocessing library
# RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-classifyers

CMD zsh
