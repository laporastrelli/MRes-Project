cd /data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project

device='cuda:0'
model_name='VGG19'
dataset='CIFAR10'
train=False
mode=''
load_pretrained=True
pretrained_name='VGG19_0_22_11_2021_15_53_15'
test=False
adversarial_test=True
PGD_iterations=20
save_to_log=False
csv_path='./results/VGG/VGG_CIFAR10_results_preliminary.csv'
result_log=''

python3 main.py \
    --device=${device}\
    --model_name=${model_name}\
    --dataset=${dataset}\
    --train=${train}\
    --mode=${mode}\
    --load_pretrained=$load_pretrained\
    --pretrained_name=${pretrained_name}\
    --test=${test}\
    --adversarial_test=${adversarial_test}\
    --PGD_iterations=${PGD_iterations}\
    --result_log=${result_log}\
    --csv_path=${csv_path}\
    --save_to_log=${save_to_log}
