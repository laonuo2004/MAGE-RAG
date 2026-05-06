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
    }
  ]
};
