FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

COPY requirements.txt requirements.txt

RUN /bin/bash -c ". activate habitat; pip install ifcfg tensorboard && pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c ". activate habitat; pip install -r requirements.txt"
RUN /bin/bash -c ". activate habitat; pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html"

COPY agent.py agent.py
COPY submission.sh submission.sh
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml
COPY configs/habitat_challenge/challenge_pointnav2020.local.rgbd_test_scene.yaml /challenge_pointnav2020.local.rgbd_test_scene.yaml
COPY configs/ configs/
COPY habitat_extensions/ habitat_extensions/
COPY occant_baselines/ occant_baselines/
COPY occant_utils/ occant_utils/
COPY pretrained_models/occant_depth_ch/ckpt.13.pth pretrained_models/occant_depth_ch/ckpt.13.pth


# Update habitat_baselines to desired commit
#RUN /bin/bash -c ". activate habitat; git clone http://github.com/facebookresearch/habitat-lab.git habitat-lab2 && (cd habitat-lab2 && git checkout 27483d017210cf710a50ba8061948eab58777202) && cp -r habitat-lab2/habitat_baselines habitat-api/."

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/challenge_pointnav2020.local.rgbd.yaml"
#WORKDIR occant_utils/astar_pycpp
#RUN make
#
#WORKDIR ../../

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --exp-config configs/habitat_challenge/ppo_navigation_evaluate.yaml --input-type rgbd"]
