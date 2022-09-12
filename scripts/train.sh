# fcn_hrnetw18
export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config escalator/configs/fcn_hrnetw18_custom_1024x512_80k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir escalator/out/train/fcn_hrnetw18 \





