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
  pkg-config \
  git

ENV SEC_HOME /sec
ENV NUM_PROCS `python -c 'import multiprocessing; print(multiprocessing.cpu_count())'`
ENV MAKEFLAGS -j8
ENV RISCV /sec/riscv
ENV PATH $RISCV/bin:$PATH
RUN mkdir -p $SEC_HOME

WORKDIR /sec
RUN git clone https://austin_d_harris@bitbucket.org/utspark/rocket-chip-sec.git rocket-chip && \
  cd rocket-chip && git submodule update --init --recursive
RUN mkdir -p $RISCV
RUN cd rocket-chip && git clone https://austin_d_harris@bitbucket.org/utspark/rocc-template-sec.git rocc-template && \
  cd rocc-template && ./install-symlinks
RUN cd rocket-chip/riscv-tools/riscv-isa-sim && autoreconf
RUN git clone https://austin_d_harris@bitbucket.org/utspark/freedom-u-sdk-sec.git freedom-u-sdk && \
  cd freedom-u-sdk && git checkout sec && git submodule update --init --recursive

WORKDIR /sec/rocket-chip/riscv-tools
RUN ./build.sh
RUN cd riscv-gnu-toolchain/build && make linux

WORKDIR /sec/freedom-u-sdk
RUN make
RUN make bbl

WORKDIR /sec
