FROM docker.io/lmsysorg/sglang:v0.5.7

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws

# Install dependencies and EFA (minimal - no kernel modules)
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        autoconf \
        pciutils \
        libhwloc-dev \
        environment-modules \
        tcl \
        automake \
        libtool \
        pkg-config \
        pci.ids \
        tcl-dev \
        environment-modules || true && \
    rm -rf /var/lib/apt/lists/*

# Install EFA userspace libraries only (no kernel modules)
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

# Create symlink for pod's expected path
RUN ln -sf /opt/amazon/ofi-nccl /opt/amazon/aws-ofi-nccl

# Set EFA environment variables
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:$PATH
