const cwd = "/root/autodl-tmp/ylz/NeurIPS_2026/code";

const jobDefaults = {
  interpreter: "bash",
  cwd,
  autorestart: true,
  stop_exit_codes: [],
  min_uptime: "60s",
  restart_delay: 30000,
  exp_backoff_restart_delay: 10000,
  kill_timeout: 60000,
  max_restarts: 20,
  time: true,
  merge_logs: true,
  env: {
    PYTHON_BIN: "/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python"
  }
};

module.exports = {
  apps: [
    {
      ...jobDefaults,
      name: "job-b1-10",
      script: "scripts/run_b1_10.sh",
      out_file: "logs/pm2-job-b1-10.out.log",
      error_file: "logs/pm2-job-b1-10.err.log"
    },
    {
      ...jobDefaults,
      name: "job-b3-10",
      script: "scripts/run_b3_10.sh",
      out_file: "logs/pm2-job-b3-10.out.log",
      error_file: "logs/pm2-job-b3-10.err.log"
    }
  ]
};
