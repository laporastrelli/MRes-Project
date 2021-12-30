cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for bn_locations in 1 2 3 4 5 0 100
do
    model_name='VGG19'
    epsilon=(0.1)
    save_to_log=True
    test_run=True
    pretrained_name=''

    python3 main.py \
        --model_name=${model_name}\
        --bn_locations=${bn_locations} \
        --epsilon=${epsilon}\
        --save_to_log=${save_to_log}\
        --test_run=${test_run}\
        --pretrained_name=${pretrained_name}
done