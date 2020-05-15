FROM nvcr.io/nvidia/tensorflow:19.03-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        gcc-5 \
        vim \
        tmux \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev

RUN pip install --no-cache-dir \
        keras==2.2.4 \
        h5py==2.8.0 \
        ipython==6.5.0 \
        ipykernel==4.8.2 \
        jupyter==1.0.0 \
        notebook==5.6.0 \
        jupyter_contrib_nbextensions==0.5.0 \
        jupyter_nbextensions_configurator==0.4.0 \
        ipywidgets==7.4.0 \
        jupyterlab==0.34.0 \
        jupyterlab_launcher==0.13.1 \
        jupyter-tensorboard==0.1.7 \
        pymongo==3.7.1 \
        setuptools \
        joblib \
        sklearn \
        Pillow \
        simdna \
        tables \
        argparse \
        progressbar \
        py3Dmol 

# Easydict
RUN cd / && \
    git clone https://github.com/makinacorpus/easydict.git && \
    cd /easydict && \
    git checkout 6149abc2283f676a4335401689ac3d513b85846b && \
    python setup.py install && \
    cd / && \
    rm -rf /easydict

# Boost
RUN cd / && \
    wget https://dl.bintray.com/boostorg/release/1.65.0/source/boost_1_65_0.tar.gz && \
    tar -xzvf boost_1_65_0.tar.gz && \
    cd boost_1_65_0/ && \
    ./bootstrap.sh --with-libraries=python,serialization && \
    ./b2 -j $(nproc) install 

# cmake
RUN cd / && \
    wget https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.sh && \
    sh cmake-3.12.1-Linux-x86_64.sh --skip-license

# Rdkit
RUN cd / && \
    wget https://github.com/rdkit/rdkit/archive/Release_2018_03_4.tar.gz && \
    tar -xzvf Release_2018_03_4.tar.gz && \
    cd rdkit-Release_2018_03_4/ && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install

# Enviroment
# Both versions are required.
#   ENV version : setup for directly running the container
#   environment : needed for SSH'ing into container
ENV LD_LIBRARY_PATH /rdkit-Release_2018_03_4/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH /rdkit-Release_2018_03_4
RUN echo "export PYTHONPATH=/rdkit-Release_2018_03_4:$$PYTHONPATH" >> /etc/environment
RUN echo "export LD_LIBRARY_PATH=/rdkit-Release_2018_03_4/lib:$$LD_LIBRARY_PATH" >> /etc/environment

# Deepchem
# Todo: change to a better directory, build RDKit
RUN cd / && \
    git clone https://github.com/deepchem/deepchem.git /deepchem && \
    cd /deepchem && \
    python setup.py install
#    rm -rf /deepchem 


# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
