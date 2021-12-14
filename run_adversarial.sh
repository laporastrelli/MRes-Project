#!/bin/bash -l
cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for bn_locations in 100
do
    model_name='VGG19'
    version=0
    train=True
    mode=''
    load_pretrained=False
    pretrained_name=''
    adversarial_test=True

    python3 main.py \
        --model_name=${model_name} \
        --version=${version} \
        --train=${train}\
        --mode=${mode}\
        --load_pretrained=$load_pretrained \
        --pretrained_name=${pretrained_name} \
        --adversarial_test=${adversarial_test}\
        --bn_locations=${bn_locations} \
        --alsologtostderr
done