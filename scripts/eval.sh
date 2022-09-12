# 验证模型在测试集上精度
export CUDA_VISIBLE_DEVICES=0 
python val.py \
    --config escalator/configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
    --model_path escalator/out/train/fcn_hrnetw18/best_model/model.pdparams \