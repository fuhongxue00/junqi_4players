{
echo "===== Script Content Start ====="
cat "$0"
echo ""
echo "===== Script Content End ====="
} > logs/rlgpu.log

nohup python -u -m minijunqi.ai.train_rl \
--epochs 30 \
--episodes_per_epoch 20 \
--from_ckpt checkpoints/01_new.pt \
--out checkpoints/02_new.pt \
>> logs/rlgpu.log 2>&1 & echo $! >> pid.log


