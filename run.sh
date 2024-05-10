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
--checkpoint=netG_model_epoch_10_top1target.pth

#resnet50
CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
--expname test_resnet50_universal_targeted_linf10_onegpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode train \
--perturbation_type universal --target 805 --gpu_ids 0 --nEpochs 10 \
--imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
--imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val


CUDA_VISIBLE_DEVICES=0 python GAP_clf.py \
--expname test_resnet50_universal_targeted_linf10_onegpu \
--batchSize 30 --testBatchSize 16 --mag_in 10 --foolmodel=resnet50 --mode test \
--perturbation_type universal --target 805 --gpu_ids 0 --nEpochs 10 --MaxIterTest 1700 \
--imagenetTrain=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/full/train \
--imagenetVal=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val \
--checkpoint=netG_model_epoch_10_top1target.pth

