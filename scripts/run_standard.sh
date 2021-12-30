#!/bin/bash -l
cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project

model_name='ResNet50'
train=True
mode=''
load_pretrained=False
pretrained_name=''
adversarial_test=False
bn_locations=0
save_to_log=False

python3 main.py \
    --model_name=${model_name} \
    --train=${train}\
    --mode=${mode}\
    --load_pretrained=${load_pretrained} \
    --adversarial_test=${adversarial_test}\
    --pretrained_name=${pretrained_name} \
    --bn_locations=${bn_locations} \
    --save_to_log=${save_to_log}

