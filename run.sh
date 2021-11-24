#!/bin/bash -l

cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project

model_name='VGG16'
train=True
load_pretrained=False
pretrained_name=''
bn_locations=(0,0,1,0,0)
# bn_locations=(1,0,0,0,0)
# bn_locations=(0,0,0,0,1)

python3 main.py \
    --model_name=${model_name} \
    --train=${train}\
    --load_pretrained=$load_pretrained \
    --pretrained_name=${pretrained_name} \
    --bn_locations=${bn_locations} \
    --alsologtostderr
