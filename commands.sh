#original project
# vgg19
CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
--expname test_vgg19_universal_targeted_linf10_onegpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel vgg19 --mode train \
--perturbation_type universal --target 805 --gpu_ids 0 --nEpochs 10 \
--imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
--imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val


CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
--expname test_vgg19_universal_targeted_linf10_onegpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel vgg19 --mode test \
--perturbation_type universal --target 805 --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
--imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
--imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
--
=netG_model_epoch_10_top1target.pth
########################################################################################################################
#resnet50
for TARGET_CLASS in {463,805,157,150,336,546,937,885}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=resnet50_imagenet.pth

    #test repaired
    echo "Testing on repaired model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=resnet50_imagenet_finetuned_repaired.pth

    #adaptive attack
    #CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    #--expname test_resnet50_universal_targeted_linf10_onegpu_adaptive \
    #--dataset=imagenet\
    #--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
    #--perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    #--imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    #--imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    #--model_in=resnet50_imagenet_finetuned_repaired.pth
done


########################################################################################################################
#vgg19
for TARGET_CLASS in {45,198,166,620,985,16,66,271}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_vgg19_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=vgg19 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=vgg19_imagenet.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_vgg19_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=vgg19 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=vgg19_imagenet_finetuned_repaired.pth
done

########################################################################################################################
#googlenet
for TARGET_CLASS in {633,968,426,609,782,254,200,458}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_googlenet_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=googlenet --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=googlenet_imagenet.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_googlenet_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=googlenet --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=googlenet_imagenet_finetuned_repaired.pth
done

########################################################################################################################
#asl mobilenet
for TARGET_CLASS in {7,12,28,19,5,16,25,18}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_mobilenet_universal_targeted_linf10_onegpu \
    --dataset=asl\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=mobilenet --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=mobilenet_asl.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_mobilenet_universal_targeted_linf10_onegpu \
    --dataset=asl\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=mobilenet --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=mobilenet_asl_ae_repaired.pth
done

    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py --expname test_mobilenet_universal_targeted_linf10_onegpu --dataset=asl --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train --perturbation_type universal --target 7 --gpu_ids 0 --nEpochs 10 --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val --model_in=mobilenet_asl.pth
########################################################################################################################
#caltech shufflenetv2
for TARGET_CLASS in {24,23,57,54,28,84,48,53,77,89}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_shufflenetv2_universal_targeted_linf10_onegpu \
    --dataset=caltech\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=shufflenetv2 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=shufflenetv2_caltech.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_shufflenetv2_universal_targeted_linf10_onegpu \
    --dataset=caltech\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=shufflenetv2 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=shufflenetv2_caltech_finetuned_repaired.pth
done
python GAP_clf.py --expname test_shufflenetv2_universal_targeted_linf10_onegpu --dataset=caltech --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train --perturbation_type universal --target 24 --gpu_ids 0 --nEpochs 5 --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train  --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val --model_in=shufflenetv2_caltech.pth
########################################################################################################################
#eurosat resnet50

for TARGET_CLASS in {3,8,1,5,9,4,5,6,0,2}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=eurosat\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=resnet50_eurosat.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=eurosat\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=resnet50_eurosat_finetuned_repaired.pth
done


    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_googlenet_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target 663 --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=googlenet_imagenet_finetuned_repaired.pth