#!/bin/bash -l
cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project

model_name='ResNet50'
train=False
mode='standard'
load_pretrained=True
pretrained_name='ResNet50_bn_27_11_2021_15_00_55'
adversarial_test=True
bn_locations=100

python3 main.py \
    --model_name=${model_name} \
    --train=${train}\
    --mode=${mode}\
    --load_pretrained=$load_pretrained \
    --pretrained_name=${pretrained_name} \
    --adversarial_test=${adversarial_test}\
    --bn_locations=${bn_locations} \
    --alsologtostderr
