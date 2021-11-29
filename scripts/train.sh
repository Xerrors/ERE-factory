# token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

# python -m main \
# --index ${token} \
# --device-id 1 \
# --use-feature-enchanced \
# --set-rel-level maxpooling --rel-loss 1 \
# --set-ent-level rel_maxpooling --ent-loss 0 \

# wait



token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

python -m main \
--device-id 1 \
--use-feature-enchanced \
--set-rel-level maxpooling --rel-loss 0 \
--set-ent-level rel_maxpooling --ent-loss 0.1 \
--index ${token} \

wait

