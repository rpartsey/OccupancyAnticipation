FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 7.5+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvml-dev-$CUDA_PKG_VERSION \
    cuda-command-line-tools-$CUDA_PKG_VERSION \
    cuda-nvprof-$CUDA_PKG_VERSION \
    cuda-npp-dev-$CUDA_PKG_VERSION \
    cuda-libraries-dev-$CUDA_PKG_VERSION \
    cuda-minimal-build-$CUDA_PKG_VERSION \
    libcublas-dev=10.2.1.243-1 \
    libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
    && apt-mark hold libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt requirements.txt
RUN /bin/bash -c ". activate habitat; pip install ifcfg torch==1.2.0 grpcio==1.24.3; pip install -r requirements.txt"
RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-api.git habitat-api2 && (cd habitat-api2 && git checkout 96243e6fdae1ba3ee8aae30edce9c18998515773) && cp -r habitat-api2/habitat_baselines habitat-api/."
RUN /bin/bash -c ". activate habitat; cd habitat-api2; pip install -r requirements.txt; python setup.py develop --all;"

COPY agent.py agent.py
COPY submission.sh submission.sh
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd_test_scene.yaml /challenge_pointnav2020.local.rgbd_test_scene.yaml
COPY configs/ configs/
COPY habitat_extensions/ habitat_extensions/
COPY occant_baselines/ occant_baselines/
COPY occant_utils/ occant_utils/
COPY pretrained_models/occant_depth_ch/ckpt.13.pth pretrained_models/occant_depth_ch/ckpt.13.pth

RUN /bin/bash -c ". activate habitat; cd occant_utils/astar_pycpp; make;"

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/challenge_pointnav2020.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --exp-config configs/habitat_challenge/occant_depth/ppo_navigation_evaluate.yaml --input-type rgbd"]