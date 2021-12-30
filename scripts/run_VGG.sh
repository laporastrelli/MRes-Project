cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for bn_locations in 1 2 3 4 5 0 100
do
    model_name='VGG19'
    train=True
    mode=''
    load_pretrained=False
    pretrained_name=''
    adversarial_test=True
    epsilon=(0.1)
    save_to_log=True

    python3 main.py \
        --model_name=${model_name}\
        --train=${train}\
        --mode=${mode}\
        --load_pretrained=$load_pretrained \
        --pretrained_name=${pretrained_name} \
        --adversarial_test=${adversarial_test}\
        --bn_locations=${bn_locations} \
        --epsilon=${epsilon}\
        --save_to_log=${save_to_log}
done