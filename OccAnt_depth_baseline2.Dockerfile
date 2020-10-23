FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker


ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# NVIDIA docker 1.0.
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451"


RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.1.0

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
        libcublas-dev=10.2.1.243-1 \
        && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

#ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt requirements.txt
RUN /bin/bash -c ". activate habitat; pip install ifcfg && pip install torch==1.2.0"
RUN /bin/bash -c ". activate habitat; pip install -r requirements.txt"
# Add other necessary libraries
#ADD OccupancyAnticipation OccupancyAnticipation

#Hack to update habitat_baselines
#TODO: Remove once base image is updated.
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-api.git habitat-api2 && (cd habitat-api2 && git checkout 96243e6fdae1ba3ee8aae30edce9c18998515773) && cp -r habitat-api2/habitat_baselines habitat-api/."
RUN /bin/bash -c ". activate habitat; cd habitat-api2 ; pip install grpcio==1.24.3 ; pip install -r requirements.txt ; python setup.py develop --all; "

#ADD occant_agent.py agent.py
#ADD occant_submission.sh submission.sh
#ADD configs/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml
#ADD configs/ configs/
#ADD ckpt.pth ckpt.pth
#ADD ppo_navigation_evaluate.yaml ppo_navigation_evaluate.yaml

COPY agent.py agent.py
COPY submission.sh submission.sh
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd_test_scene.yaml /challenge_pointnav2020.local.rgbd_test_scene.yaml
COPY configs/ configs/
COPY habitat_extensions/ habitat_extensions/
COPY occant_baselines/ occant_baselines/
COPY occant_utils/ occant_utils/
COPY pretrained_models/occant_depth_ch/ckpt.13.pth pretrained_models/occant_depth_ch/ckpt.13.pth

RUN /bin/bash -c ". activate habitat; cd occant_utils/astar_pycpp ; make; "


ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/challenge_pointnav2020.local.rgbd.yaml"


CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --exp-config configs/habitat_challenge/occant_depth/ppo_navigation_evaluate.yaml --input-type rgbd"]