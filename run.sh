########################################################################################################################
#batch run

for TARGET_CLASS in {463,805,157,150,336,546,937,885}
do
    echo "Analyzing target class:" $TARGET_CLASS
    #attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val

    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --checkpoint=netG_model_epoch_10_top1target.pth \
    --model_in=resnet50_imagenet_finetuned_repaired.pth

    #adaptive attack
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu_adaptive \
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --model_in=resnet50_imagenet_finetuned_repaired.pth
done

