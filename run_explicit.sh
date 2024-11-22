########################################################################################################################
#resnet50
TARGET_CLASS=463
    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val  \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/test \
    --explicit_U=test_resnet50_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_57.10000228881836.pth  \
    --model_in=resnet50_imagenet.pth

    #test repaired
    echo "Testing on repaired model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val  \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/test \
    --explicit_U=test_resnet50_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_57.10000228881836.pth  \
    --model_in=resnet50_imagenet_finetuned_repaired.pth

########################################################################################################################
#vgg19
TARGET_CLASS=45
    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_vgg19_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=vgg19 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/test \
    --explicit_U=test_vgg19_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_63.70000457763672.pth  \
    --model_in=vgg19_imagenet.pth

    #test repaired
    echo "Testing on repaired model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_vgg19_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=vgg19 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/test \
    --explicit_U=test_vgg19_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_63.70000457763672.pth  \
    --model_in=vgg19_imagenet_finetuned_repaired.pth

########################################################################################################################
#googlenet

TARGET_CLASS=458

    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_googlenet_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=googlenet --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --explicit_U=test_googlenet_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_57.400001525878906.pth  \
    --model_in=googlenet_imagenet.pth

    echo "Testing on repaired model"
    #test repaired
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_googlenet_universal_targeted_linf10_onegpu \
    --dataset=imagenet\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=googlenet --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet_sga/val \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --explicit_U=test_googlenet_universal_targeted_linf10_onegpu/U_out/U_epoch_10_top1target_57.400001525878906.pth  \
    --model_in=googlenet_imagenet_finetuned_repaired.pth

########################################################################################################################
#mobilenet asl

TARGET_CLASS=7

    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_mobilenet_universal_targeted_linf10_onegpu \
    --dataset=asl\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=mobilenet --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --explicit_U=test_mobilenet_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_57.06521987915039.pth  \
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
    --explicit_U=test_mobilenet_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_57.06521987915039.pth  \
    --model_in=mobilenet_asl_ae_repaired.pth

########################################################################################################################
#caltech shufflenetv2

TARGET_CLASS=24

    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_shufflenetv2_universal_targeted_linf10_onegpu \
    --dataset=caltech\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=shufflenetv2 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --explicit_U=test_shufflenetv2_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_49.09638214111328.pth  \
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
    --explicit_U=test_shufflenetv2_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_49.09638214111328.pth  \
    --model_in=shufflenetv2_caltech_finetuned_repaired.pth

########################################################################################################################
#eurosat resnet50

TARGET_CLASS=3
    echo "Testing on original model"
    CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
    --expname test_resnet50_universal_targeted_linf10_onegpu \
    --dataset=eurosat\
    --batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
    --perturbation_type universal --target $TARGET_CLASS --gpu_ids 0 --nEpochs 5 --MaxIterTest 1700 \
    --imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
    --imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
    --explicit_U=test_resnet50_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_79.85248565673828.pth  \
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
    --explicit_U=test_resnet50_universal_targeted_linf10_onegpu/U_out/U_epoch_5_top1target_79.85248565673828.pth  \
    --model_in=resnet50_eurosat_finetuned_repaired.pth