module.exports = {
  apps: [
    {
      name: "litellm",
      script: "scripts/serve_litellm.sh",
      interpreter: "bash",
      args: "configs/litellm_config.yaml 4000",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-litellm.out.log",
      error_file: "logs/pm2-litellm.err.log",
      merge_logs: true
    },
    {
      name: "vllm",
      script: "scripts/pm2_vllm_wrapper.sh",
      interpreter: "bash",
      args: "maxctx",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        CUDA_VISIBLE_DEVICES: "1",
        PORT: "8000",
        GPU_MEMORY_UTILIZATION: "0.70",
        VLLM_SERVE_SCRIPT: "scripts/serve_qwen3_vl_vllm.sh",
        VLLM_STOP_GRACE_SECONDS: "20",
        CLEAN_STALE_ON_START: "1"
      },
      autorestart: true,
      min_uptime: "60s",
      restart_delay: 30000,
      exp_backoff_restart_delay: 10000,
      kill_timeout: 60000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-vllm.out.log",
      error_file: "logs/pm2-vllm.err.log",
      merge_logs: true
    },
    {
      name: "pm2-log-rotate-local",
      script: "scripts/pm2_log_rotate.sh",
      interpreter: "bash",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        MAX_SIZE_MB: "100",
        KEEP: "5",
        INTERVAL_SECONDS: "300"
      },
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-log-rotate.out.log",
      error_file: "logs/pm2-log-rotate.err.log",
      merge_logs: true
    }
  ]
};
