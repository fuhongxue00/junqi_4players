{
echo "===== Script Content Start ====="
cat "$0"
echo ""
echo "===== Script Content End ====="
} > logs/ai_vs_ai.log


nohup python -u scripts/ai_vs_ai.py \
--count 10 \
--ckpt_og_team checkpoints/04.pt \
>> logs/ai_vs_ai.log 2>&1 & echo $! > ai_vs_ai_pid.log

# --ckpt_pb_team checkpoints/03.pt \ 