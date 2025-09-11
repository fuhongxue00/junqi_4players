{
echo "===== Script Content Start ====="
cat "$0"
echo ""
echo "===== Script Content End ====="
} > logs/rl.log

nohup python -m -u minijunqi.ai.train_rl \
--epochs 10 \
--out checkpoints/rl.pt \
--from_ckpt checkpoints/02.pt \
>> logs/rl.log 2>&1 & echo $! > rl_pid.log