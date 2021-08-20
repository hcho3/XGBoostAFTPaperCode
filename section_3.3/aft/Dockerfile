FROM ubuntu:18.04

# Environment
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]   # Use Bash as shell

# Install all basic requirements
RUN \
  apt-get update && \
  apt-get install -y tar unzip wget git build-essential ninja-build && \
  # CMake
  wget -nv -nc https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.sh --no-check-certificate && \
  bash cmake-3.13.0-Linux-x86_64.sh --skip-license --prefix=/usr && \
  # Python
  wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  bash Miniconda3.sh -b -p /opt/python

ENV PATH=/opt/python/bin:$PATH

ENV GOSU_VERSION 1.10

# Set up Python env
RUN conda create -n py37 -c conda-forge python=3.7 numpy scipy scikit-learn pandas optuna && \
    source activate py37 && \
    python -m pip install awscli boto3 && \
    source deactivate

# Install XGBoost
RUN source activate py37 && \
    git clone --recursive https://github.com/hcho3/xgboost -b aft_experiment && \
    cd xgboost && \
    mkdir build && cd build && \
    cmake .. -GNinja && ninja && \
    cd ../python-package && python setup.py install && \
    source deactivate

# Install lightweight sudo (not bound to TTY)
RUN set -ex; \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-amd64" && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

COPY hpo.py /scripts/
COPY entrypoint.sh /scripts/
ENTRYPOINT ["/scripts/entrypoint.sh"]
