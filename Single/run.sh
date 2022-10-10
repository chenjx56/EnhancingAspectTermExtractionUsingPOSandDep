H=res15
CUDA_VISIBLE_DEVICES=2 python main.py \
--data_path data/Process_data/${H}_file.txt \
--log_path log \
--output_dir output/test/ \
--seed 22 \
--epochs 8 \
--fp16 \
--fp_type "O2" \
--datasets ${H} \
--types "test" \
--plm "PLM/bert-base-uncased"