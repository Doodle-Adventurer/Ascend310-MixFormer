当前板卡为Ascend310，CANN版本为CANN-6.3.RC2.alpha003

项目cmake链接可自己调整

注：切换不同的板卡需要重新转换模型，即onnx转为om
atc --model=mixformer_v2.onnx \
    --framework=5 \
    --output=mixformer_v2 \
    --input_shape="img_t:1,3,112,112;img_ot:1,3,112,112;img_search:1,3,224,224" \
    --soc_version=Ascend310 
Ascend310需改为自己板卡型号。
