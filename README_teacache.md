# teacache 

```
CUDA_VISIBLE_DEVICES=0 \
python3 teacache_acc_infer.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /work/models/Wan-AI/Wan2___1-T2V-1___3B \
--sample_shift 8 \
--sample_guide_scale 6 \
--prompt "左边身穿白色上衣蓝色裤子蓝色拳击套的拟人化猫咪、右边身穿白色上衣红色裤子红色拳击套的拟人化猫咪在聚光灯下的舞台上激烈搏斗！" \
--base_seed 42 \
--teacache_thresh 0.20 \
--save_file ./test-1.3-teacache-two-cat-boxing-0.20.mp4
```
备注：不支持多卡推理

# teacache_xdit

 ```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
--nproc_per_node=6 \
teacache_xdit_acc_infer.py  \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir  /work/models/Wan-AI/Wan2___1-T2V-1___3B \
--prompt "左边身穿白色上衣蓝色裤子蓝色拳击套的拟人化猫咪、右边身穿白色上衣红色裤子红色拳击套的拟人化猫咪在聚光灯下的舞台上激烈搏斗！" \
--base_seed 42 \
--offload_model True \
--t5_cpu  \
--ulysses_size 6 \
--teacache_thresh 0.2 \
--save_file ./005.mp4
 ```