
#fcn_hrnetw18   
export CUDA_VISIBLE_DEVICES=0
python export.py \
        --config escalator/configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml  \
        --model_path escalator/out/train/fcn_hrnetw18/best_model/model.pdparams \
        --save_dir escalator/out/export/fcn_hrnetw18 \
        --without_argmax \
        --with_softmax \
        --input_shape 1 3 720 1280

paddle2onnx --model_dir escalator/out/export/fcn_hrnetw18 \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file escalator/out/export/fcn_hrnetw18/fcn_hrnetw18.onnx

python -m paddle2onnx.optimize \
        --input_model escalator/out/export/fcn_hrnetw18/fcn_hrnetw18.onnx \
        --output_model escalator/out/export/fcn_hrnetw18/fcn_hrnetw18_simple.onnx






#fcn_hrnetw18   
export CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml  \
       --model_path out/train/custom_dataset/fcn_hrnetw18/iter_80000/model.pdparams \
       --save_dir out/export/fcn_hrnetw18 \
       --without_argmax \
       --with_softmax \
       --input_shape 1 3 720 1280

export CUDA_VISIBLE_DEVICES=0
paddle2onnx --model_dir out/export/fcn_hrnetw18 \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/fcn_hrnetw18/fcn_hrnetw18.onnx