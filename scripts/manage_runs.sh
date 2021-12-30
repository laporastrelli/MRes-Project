#!/bin/bash -l

cd ./new_runs/

base="./new_runs/" 
cnt=0
look_for="Validation"
vgg19="VGG19"
resnet="ResNet50"
max_vgg="29324"
max_resnet="58649"

for d in */ ; do
    echo "$d"
    tensorboard --inspect --logdir=$d > ./$d/event_file.txt

    if grep "$look_for" ./$d/event_file.txt
    then 
        if grep "$vgg19" ./$d/event_file.txt && grep "$max_vgg" ./$d/event_file.txt
        then
            cnt=$((cnt+1))
        elif grep "$resnet" ./$d/event_file.txt && grep "$max_resnet" ./$d/event_file.txt
        then
            cnt=$((cnt+1))
        fi
    else
        echo "Training not complete"
    fi
done

echo $cnt

