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
      name: "job-group-b-0",
      script: "scripts/run_group_b_0.sh",
      out_file: "logs/pm2-job-group-b-0.out.log",
      error_file: "logs/pm2-job-group-b-0.err.log"
    },
    {
      ...jobDefaults,
      name: "job-group-b-1",
      script: "scripts/run_group_b_1.sh",
      out_file: "logs/pm2-job-group-b-1.out.log",
      error_file: "logs/pm2-job-group-b-1.err.log"
    },
    {
      ...jobDefaults,
      name: "job-group-b-2",
      script: "scripts/run_group_b_2.sh",
      out_file: "logs/pm2-job-group-b-2.out.log",
      error_file: "logs/pm2-job-group-b-2.err.log"
    },
    {
      ...jobDefaults,
      name: "job-group-b-3",
      script: "scripts/run_group_b_3.sh",
      out_file: "logs/pm2-job-group-b-3.out.log",
      error_file: "logs/pm2-job-group-b-3.err.log"
    }
  ]
};
