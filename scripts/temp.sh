token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

python -m main \
--device-id 3 \
--use-feature-enchanced \
--set-ent-level sent --ent-loss 0.1 \
--index ${token} 

wait

token=`date "+%Y-%m-%d_%H-%M-%S"`-temp

python -m main \
--device-id 3 \
--use-feature-enchanced \
--set-ent-level sent --ent-loss 1 \
--index ${token}