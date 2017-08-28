FROM ubuntu:16.04

#Install riscv-tools dependencies
RUN apt-get update && apt-get install -y \
  autoconf \
  automake \
  autotools-dev \
  bc \
  bison \
  build-essential \
  curl \
  device-tree-compiler \
  flex \
  gawk \
  gperf \
  libmpc-dev \
  libmpfr-dev \
  libgmp-dev \
  libtool \
  libusb-1.0-0-dev \
  patchutils \
  zlib1g-dev \
  wget \
  cpio \
  python \
  unzip \
  texinfo \
  git

ENV SEC_HOME /sec
ENV NUM_PROCS 32
ENV MAKEFLAGS -j$NUM_PROCS
RUN mkdir -p $SEC_HOME

WORKDIR /sec
RUN git clone https://austin_d_harris@bitbucket.org/utspark/freedom-u-sdk-sec.git && \
  cd freedom-u-sdk && git checkout sec && git submodule update --init --recursive

WORKDIR /sec/freedom-u-sdk
RUN make
ENV RISCV /sec/freedom-u-sdk/toolchain
ENV PATH $RISCV/bin:$PATH
RUN make spike
