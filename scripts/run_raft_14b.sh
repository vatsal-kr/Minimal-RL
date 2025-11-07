set -x

data=cerebrm_dataset
project_name=raft
algorithm=raft
size=${1:-14B}
model=Deepseek-R1-Distill-Qwen-$size
model_name_or_path=deepseek-ai/$model
policy_loss=vanilla # vanilla, plusplus (importance sample + clipping)
n=16
experiment_name=${model}-${algorithm}-${policy_loss}-${data}-n${n}
GPUS=(0 1 2 3)\
my_world_size=${#GPUS[@]}

math_train_path=./data/$data/train.parquet
math_test_path=./data/$data/test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

mkdir -p logs/${project_name}

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.model.chat_template_path=./verl/utils/dsqwen_chat_template.jinja \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.dtype="bfloat16" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0 \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$my_world_size \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=78 \
    trainer.default_local_dir=$WORK/raft/${experiment_name} \
    trainer.test_freq=78 \
    trainer.total_epochs=1 2>&1 | tee logs/${experiment_name}.log

