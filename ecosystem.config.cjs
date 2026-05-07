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
      script: "scripts/serve_qwen3_vl_vllm.sh",
      interpreter: "bash",
      args: "longctx",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        CUDA_VISIBLE_DEVICES: "1",
        PORT: "8000",
        GPU_MEMORY_UTILIZATION: "0.60"
      },
      autorestart: true,
      restart_delay: 10000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-vllm.out.log",
      error_file: "logs/pm2-vllm.err.log",
      merge_logs: true
    }
  ]
};