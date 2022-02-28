cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for how_many in 0 1 2 3 4
do
    for bn_locations in 1 2 3 4 0 100
    do  
        device='cuda:2'
        model_name='ResNet50'
        version=3
        dataset='SVHN'
        train=True
        mode='standard'
        load_pretrained=False
        pretrained_name=''
        test=True
        adversarial_test=True
        PGD_iterations=40
        save_to_log=True
        csv_path='./results/ResNet/ResNet_v3_SVHN_results_preliminary.csv'

        python3 main.py \
            --device=${device}\
            --model_name=${model_name}\
            --version=${version}\
            --dataset=${dataset}\
            --train=${train}\
            --mode=${mode}\
            --load_pretrained=${load_pretrained}\
            --pretrained_name=${pretrained_name}\
            --test=${test}\
            --adversarial_test=${adversarial_test}\
            --PGD_iterations=${PGD_iterations}\
            --bn_locations=${bn_locations}\
            --save_to_log=${save_to_log}\
            --csv_path=${csv_path}\
            
    done
done