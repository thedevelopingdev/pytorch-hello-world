# https://pythonspeed.com/articles/activate-conda-dockerfile/
# https://fabiorosado.dev/blog/install-conda-in-docker/
# https://nielscautaerts.xyz/making-dockerfiles-architecture-independent.html
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.7

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG TARGETPLATFORM

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then        \
        ARCHITECTURE="x86_64";                            \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then   \
        ARCHITECTURE="aarch64";                           \
    else                                                  \
        ARCHITECTURE="x86_64";                            \
    fi                                                    \
    && echo $TARGETPLATFORM $ARCHITECTURE && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCHITECTURE}.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create user `mattfeng`
RUN useradd -ms /bin/bash mattfeng
RUN usermod -aG sudo mattfeng
RUN echo 'mattfeng:password' | chpasswd
RUN chown -R mattfeng:mattfeng /opt/conda

USER mattfeng
WORKDIR /app

RUN pip install torch torchvision torchaudio

ENTRYPOINT ["sleep", "infinity"]