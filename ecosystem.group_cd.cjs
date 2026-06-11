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
      name: "job-group-c",
      script: "scripts/run_group_c.sh",
      out_file: "logs/pm2-job-group-c.out.log",
      error_file: "logs/pm2-job-group-c.err.log"
    },
    {
      ...jobDefaults,
      name: "job-group-d",
      script: "scripts/run_group_d.sh",
      out_file: "logs/pm2-job-group-d.out.log",
      error_file: "logs/pm2-job-group-d.err.log"
    }
  ]
};
