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
      name: "job-c4",
      script: "scripts/run_c4.sh",
      out_file: "logs/pm2-job-c4.out.log",
      error_file: "logs/pm2-job-c4.err.log"
    },
    {
      ...jobDefaults,
      name: "job-c5",
      script: "scripts/run_c5.sh",
      out_file: "logs/pm2-job-c5.out.log",
      error_file: "logs/pm2-job-c5.err.log"
    },
    {
      ...jobDefaults,
      name: "job-d3",
      script: "scripts/run_d3.sh",
      out_file: "logs/pm2-job-d3.out.log",
      error_file: "logs/pm2-job-d3.err.log"
    },
    {
      ...jobDefaults,
      name: "job-d4",
      script: "scripts/run_d4.sh",
      out_file: "logs/pm2-job-d4.out.log",
      error_file: "logs/pm2-job-d4.err.log"
    },
    {
      ...jobDefaults,
      name: "job-d5",
      script: "scripts/run_d5.sh",
      out_file: "logs/pm2-job-d5.out.log",
      error_file: "logs/pm2-job-d5.err.log"
    }
  ]
};
