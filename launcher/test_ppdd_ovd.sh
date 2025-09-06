python train_att_net.py \
  --num-gpus 2 \
  --eval-only \
  --config-file configs/Base_PPDD_C4_1x_test_nyem.yaml \
  MODEL.WEIGHTS outputs/ppdd_ovd_train/model_final.pth \
  OUTPUT_DIR outputs/ppdd_ovd_train \
  MODEL.ATTRIBUTE_ON True
