# token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

# python -m main \
# --index ${token} \
# --device-id 1 \
# --use-feature-enchanced \
# # --use-negative-mask \
# # --set-rel-level maxpooling \
# # --set-ent-level rel_maxpooling \

# wait

token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

python -m main \
--index ${token} \
--device-id 1 \
--use-feature-enchanced \
# --set-table-calc biaffine \
# --use-negative-mask \
# --set-rel-level maxpooling \
# --set-ent-level rel_maxpooling \

wait

