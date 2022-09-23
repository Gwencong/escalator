# 预测图片并可视化
python predict.py \
    --config escalator/configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
    --model_path escalator/out/train/fcn_hrnetw18/best_model/model.pdparams \
    --image_path escalator/data/test.txt \
    --save_dir escalator/out/detect/fcn_hrnetw18