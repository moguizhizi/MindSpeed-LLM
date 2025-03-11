python ./preprocess_data.py \
    --input ./finetune_dataset/ \
    --output-prefix ./finetune_dataset/merge \
    --merge-group-keys packed_attention_mask_document packed_input_ids_document packed_labels_document