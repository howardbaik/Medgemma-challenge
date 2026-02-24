python medgemma_qa.py \
    --dataset /workspace/dilek/medgemma-challenge/scripts/mimic_vqa_sample.json \
    --video-base-path /workspace/dilek/medgemma-challenge/mimic-iv-echo  \
    --output /workspace/dilek/medgemma-challenge/scripts/mimic_vqa_sample_predictions.json \
    --frame-strategy first \
    --model-id "google/medgemma-1.5-4b-it" \
    --include-report
    
# replace dataset, video-base-path, and output with your own directories