#!/bin/bash -l
cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project

model_name='ResNet50'
version=1
train=False
mode='standard'
load_pretrained=True
pretrained_name='ResNet50_bn_26_12_2021_13_25_55'
test=True
adversarial_test=True
bn_locations=100
epsilon=(0.1)
save_to_log=False

python3 main.py \
    --model_name=${model_name} \
    --version=${version} \
    --train=${train}\
    --mode=${mode}\
    --load_pretrained=$load_pretrained \
    --pretrained_name=${pretrained_name} \
    --adversarial_test=${adversarial_test}\
    --test=${test}\
    --bn_locations=${bn_locations} \
    --epsilon=${epsilon}\
    --save_to_log=${save_to_log}
