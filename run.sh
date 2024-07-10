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