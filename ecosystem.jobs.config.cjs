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
      name: "job-bm25-sweep",
      script: "scripts/run_bm25_sweep.sh",
      out_file: "logs/pm2-job-bm25-sweep.out.log",
      error_file: "logs/pm2-job-bm25-sweep.err.log"
    },
    {
      ...jobDefaults,
      name: "job-m3docrag",
      script: "scripts/run_m3docrag.sh",
      out_file: "logs/pm2-job-m3docrag.out.log",
      error_file: "logs/pm2-job-m3docrag.err.log"
    },
    {
      ...jobDefaults,
      name: "job-image-ocr",
      script: "scripts/run_image_ocr.sh",
      out_file: "logs/pm2-job-image-ocr.out.log",
      error_file: "logs/pm2-job-image-ocr.err.log"
    },
    {
      ...jobDefaults,
      name: "job-m3docrag-iterate",
      script: "scripts/run_m3docrag-iterate.sh",
      out_file: "logs/pm2-job-m3docrag-iterate.out.log",
      error_file: "logs/pm2-job-m3docrag-iterate.err.log"
    },    
    {
      ...jobDefaults,
      name: "job-m3docrag-iterate-query",
      script: "scripts/run_m3docrag-iterate-query.sh",
      out_file: "logs/pm2-job-m3docrag-iterate-query.out.log",
      error_file: "logs/pm2-job-m3docrag-iterate-query.err.log"
    },
    {
      ...jobDefaults,
      name: "job-build-longdocurl",
      script: "scripts/run_build_longdocurl.sh",
      out_file: "logs/pm2-job-build-longdocurl.out.log",
      error_file: "logs/pm2-job-build-longdocurl.err.log"
    },
    {
      ...jobDefaults,
      name: "job-build-mmlongbench",
      script: "scripts/run_build_mmlongbench.sh",
      out_file: "logs/pm2-job-build-mmlongbench.out.log",
      error_file: "logs/pm2-job-build-mmlongbench.err.log"
    },                
    {
      ...jobDefaults,
      name: "job-magerag",
      script: "scripts/run_magerag.sh",
      out_file: "logs/pm2-job-magerag.out.log",
      error_file: "logs/pm2-job-magerag.err.log"
    },         
  ]
};
