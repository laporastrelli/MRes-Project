cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project
for bn_locations in 1 2 3 4 0 100
do
    device='cuda:1'
    model_name='ResNet34'
    dataset='CIFAR10'
    batch_size=128
    version=1
    train=True
    mode='standard'
    load_pretrained=False
    test=True
    adversarial_test=True
    use_pop_stats=False
    attacks_in=('PGD')
    PGD_iterations=40
    save_to_log=True
    csv_path='./results/ResNet34/ResNet34_CIFAR10_results.csv'

    python3 main.py \
        --device=${device}\
        --model_name=${model_name}\
        --dataset=${dataset}\
        --batch_size=${batch_size}\
        --version=${version}\
        --train=${train}\
        --mode=${mode}\
        --load_pretrained=${load_pretrained}\
        --test=${test}\
        --adversarial_test=${adversarial_test}\
        --use_pop_stats=${use_pop_stats}\
        --attacks_in=${attacks_in}\
        --PGD_iterations=${PGD_iterations}\
        --save_to_log=${save_to_log}\
        --csv_path=${csv_path}\
        --bn_locations=${bn_locations}\
        
done