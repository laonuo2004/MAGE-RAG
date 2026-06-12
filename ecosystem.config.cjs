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
      name: "litellm-hyyun",
      script: "scripts/serve_litellm.sh",
      interpreter: "bash",
      args: "configs/litellm_hyyun_config.yaml 4010",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-litellm-hyyun.out.log",
      error_file: "logs/pm2-litellm-hyyun.err.log",
      merge_logs: true
    },    
    {
      name: "litellm-autodl",
      script: "scripts/serve_litellm.sh",
      interpreter: "bash",
      args: "configs/litellm_autodl_config.yaml 4020",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-litellm-autodl.out.log",
      error_file: "logs/pm2-litellm-autodl.err.log",
      merge_logs: true
    },
    {
      name: "vllm",
      script: "scripts/pm2_vllm_wrapper.sh",
      interpreter: "bash",
      args: "128k",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        CUDA_VISIBLE_DEVICES: "0,1",
        PORT: "8010",
        GPU_MEMORY_UTILIZATION: "0.35",
        VLLM_SERVE_SCRIPT: "scripts/serve_qwen3_vl_vllm.sh",
        VLLM_STOP_GRACE_SECONDS: "20",
        CLEAN_STALE_ON_START: "1",
        TENSOR_PARALLEL_SIZE: "2",
        DATA_PARALLEL_SIZE: "1"
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
      name: "vllm-colpali",
      script: "scripts/pm2_vllm_wrapper.sh",
      interpreter: "bash",
      args: "8k",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        CUDA_VISIBLE_DEVICES: "0",
        PORT: "8020",
        MODEL_NAME: "/root/autodl-tmp/ylz/models/colpali-v1.3-hf",
        SERVED_MODEL_NAME: "colpali-v1.3",
        VLLM_SERVE_SCRIPT: "scripts/serve_colpali_vllm.sh",
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
      out_file: "logs/pm2-vllm-colpali.out.log",
      error_file: "logs/pm2-vllm-colpali.err.log",
      merge_logs: true
    },

    {
      name: "results-dashboard",
      script: "scripts/serve_results_dashboard.sh",
      interpreter: "bash",
      cwd: "/root/autodl-tmp/ylz/NeurIPS_2026/code",
      env: {
        RESULTS_DASHBOARD_PORT: "8501",
        STREAMLIT_SERVER_HEADLESS: "true"
      },
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 100000,
      time: true,
      out_file: "logs/pm2-results-dashboard.out.log",
      error_file: "logs/pm2-results-dashboard.err.log",
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
