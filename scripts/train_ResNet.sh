cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for bn_locations in 1 2 3 4 0 100
do
    device='cuda:1'
    model_name='ResNet50'
    version=4
    train=True
    mode='standard'
    load_pretrained=False
    pretrained_name=''
    test=False
    adversarial_test=False
    epsilon=(0.1)
    save_to_log=False

    python3 main.py \
        --model_name=${model_name}\
        --version=${version}\
        --train=${train}\
        --mode=${mode}\
        --load_pretrained=$load_pretrained \
        --pretrained_name=${pretrained_name}\
        --adversarial_test=${adversarial_test}\
        --test=${test}\
        --bn_locations=${bn_locations} \
        --epsilon=${epsilon}\
        --save_to_log=${save_to_log}\
        
done