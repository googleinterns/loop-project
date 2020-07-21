#!/bin/sh
size=16
aug=1
pixel_mean=1
batch_size=32
num_blocks=16
num_templates=4
dropout=0.0
out_adapter="isometric"
in_adapter="strided"
label_smoothing=0.1
kernel_reg=0.00001
lr=0.01

for out_adapter in "v1", "v2", "isometric", "dethwise";
do
  for dropout in 0. 0.1 0.2 0.3 0.4 0.5; do
	echo "Running custom block experiment with parameters:";
	echo "tensor size = $size; aug = $aug; pixel_mean = $pixel_mean; batch size = $batch_size; num_blocks = $num_blocks; num_templates = $num_templates; dropout= $dropout; out_adapter_type = $out_adapter; conv base = $conv_base; label smoothing = $label_smoothing; kernel_reg = $kernel_reg; lr = $lr";
	CUDA_VISIBLE_DEVICES="1" python resnet_exp.py --size $size --aug_mode $aug_mode --batch_size $batch_size --num_blocks $num_blocks --num_templates $num_templates --dropout $dropout --out_adapter_type $out_adapter --in_adapter_type $in_adapter --lsmooth $label_smoothing --kernel_reg $kernel_reg --lr $lr --shared 1;
  done;
echo "";
done

