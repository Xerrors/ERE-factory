# token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

# python -m main \
# --index ${token} \
# --device-id 1 \
# --use-feature-enchanced \
# --set-rel-level maxpooling --rel-loss 1 \
# --set-ent-level rel_maxpooling --ent-loss 0 \

# wait



# token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

# python -m main \
# --device-id 2 \
# --use-feature-enchanced \
# --index ${token} &

# sleep 1

token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

python -m main \
--device-id 2 \
--use-feature-enchanced \
--set-ent-level sent_enchanced --ent-loss 1 \
--index ${token} &

wait

