# 自动生成ik_config文件的human_scale与pos/quat_offset

## 安装要求
```bash
pip install lxml
pip install matplotlib
```

## 基础配置
-pose_inits中添加_tpose.json文件(设置机器人的初始位姿为T-pose) \
-ik_configs中添加bvh/smplx_to_robot_origin.json文件（主要需要joint_match）\
将人形机器人与human_data在T-pose下完全对齐

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
