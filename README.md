# GMR: General Motion Retargeting(Fork for IK-CONFIG auto-generation)

  <a href="https://arxiv.org/abs/2505.02833">
    <img src="https://img.shields.io/badge/paper-arXiv%3A2505.02833-b31b1b.svg" alt="arXiv Paper"/>
  </a> <a href="https://arxiv.org/abs/2510.02252">
    <img src="https://img.shields.io/badge/paper-arXiv%3A2510.02252-b31b1b.svg" alt="arXiv Paper"/>
  </a> <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a> <a href="https://github.com/YanjieZe/GMR/releases">
    <img src="https://img.shields.io/badge/version-0.2.0-blue.svg" alt="Version"/>
  </a> <a href="https://x.com/ZeYanjie/status/1952446745696469334">
    <img src="https://img.shields.io/badge/twitter-ZeYanjie-blue.svg" alt="Twitter"/>
  </a> <a href="https://yanjieze.github.io/humanoid-foundation/#GMR">
    <img src="https://img.shields.io/badge/blog-GMR-blue.svg" alt="Blog"/>
  </a> <a href="https://www.bilibili.com/video/BV1p1nazeEzC/?share_source=copy_web&vd_source=c76e3ab14ac3f7219a9006b96b4b0f76">
    <img src="https://img.shields.io/badge/tutorial-BILIBILI-blue.svg" alt="Blog"/>
  </a>

![Banner for GMR](./assets/GMR.png)

![GMR](./assets/GMR_pipeline.png)


This repository is forked from [GMR](https://github.com/YanjieZe/GMR)

Please refer to the original project's README first.

This repository has been modified to add *IK-CONFIG auto-generation* function for your own humanoid robots.

## 机器人IK_CONFIG自动生成
详见ik_config_manager文件夹 \
辅助生成*human_scale*与*pos/quat_offset*

## 安装要求
```bash
pip install lxml
pip install matplotlib
```

## 基础配置
-pose_inits中添加_tpose.json文件(设置机器人的初始位姿为T-pose) \
-ik_configs中添加bvh/smplx_to_robot_origin.json文件（主要需要joint_match）\
将人形机器人与human_data在T-pose下完全对齐

![T-pose](./ik_config_manager/bvh-TPOSE.png)

## 具体使用
BVH格式：
```bash
python ik_config_manager/generate_keypoint_mapping_bvh.py \
    --bvh_file ik_config_manager/TPOSE.bvh \
    --robot unitree_g1 \
    --loop \
    --robot_qpos_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
    --ik_config_in general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
    --ik_config_out general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_auto.json
```

SMPLX格式：
```bash
python ik_config_manager/generate_keypoint_mapping_smplx.py \
    --smplx_file ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz \
    --robot unitree_g1 \
    --loop \
    --robot_qpos_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
    --ik_config_in general_motion_retargeting/ik_configs/smplx_to_g1.json \
    --ik_config_out general_motion_retargeting/ik_configs/smplx_to_g1_auto.json
```
