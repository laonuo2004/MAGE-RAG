const codeCwd = "/root/autodl-tmp/ylz/NeurIPS_2026/code";

const syncDefaults = {
  interpreter: "bash",
  cwd: codeCwd,
  autorestart: true,
  min_uptime: "10s",
  restart_delay: 5000,
  exp_backoff_restart_delay: 10000,
  kill_timeout: 10000,
  max_restarts: 100000,
  time: true,
  merge_logs: true
};

module.exports = {
  apps: [
    {
      ...syncDefaults,
      name: "sync-code-to-ai4s",
      script: "scripts/sync_code_to_ai4s_loop.sh",
      env: {
        LOCAL_CODE_DIR: codeCwd,
        REMOTE_CODE_DEST: "ai4s:/root/autodl-tmp/ylz/NeurIPS_2026/code/",
        SYNC_INTERVAL_SECONDS: "10"
      },
      out_file: "logs/pm2-sync-code-to-ai4s.out.log",
      error_file: "logs/pm2-sync-code-to-ai4s.err.log"
    },
    {
      ...syncDefaults,
      name: "sync-magerag-results-to-ai4s",
      script: "scripts/sync_magerag_results_to_remote_loop.sh",
      env: {
        LOCAL_CODE_DIR: codeCwd,
        REMOTE_CODE_DEST: "ai4s:/root/autodl-tmp/ylz/NeurIPS_2026/code",
        REMOTE_SSH_HOST: "ai4s",
        RESULTS_RELATIVE_DIR: "results/longdocurl/magerag",
        SYNC_INTERVAL_SECONDS: "10"
      },
      out_file: "logs/pm2-sync-magerag-results-to-ai4s.out.log",
      error_file: "logs/pm2-sync-magerag-results-to-ai4s.err.log"
    },
    {
      ...syncDefaults,
      name: "sync-magerag-results-to-mm",
      script: "scripts/sync_magerag_results_to_remote_loop.sh",
      env: {
        LOCAL_CODE_DIR: codeCwd,
        REMOTE_CODE_DEST: "MM:/root/autodl-tmp/ylz/NeurIPS_2026/code",
        REMOTE_SSH_HOST: "MM",
        RESULTS_RELATIVE_DIR: "results/longdocurl/magerag",
        SYNC_INTERVAL_SECONDS: "10"
      },
      out_file: "logs/pm2-sync-magerag-results-to-mm.out.log",
      error_file: "logs/pm2-sync-magerag-results-to-mm.err.log"
    },
    {
      ...syncDefaults,
      name: "sync-mmlongbench-magerag-results-to-ai4s",
      script: "scripts/sync_magerag_results_to_remote_loop.sh",
      env: {
        LOCAL_CODE_DIR: codeCwd,
        REMOTE_CODE_DEST: "ai4s:/root/autodl-tmp/ylz/NeurIPS_2026/code",
        REMOTE_SSH_HOST: "ai4s",
        RESULTS_RELATIVE_DIR: "results/mmlongbench/magerag",
        SYNC_INTERVAL_SECONDS: "10"
      },
      out_file: "logs/pm2-sync-mmlongbench-magerag-results-to-ai4s.out.log",
      error_file: "logs/pm2-sync-mmlongbench-magerag-results-to-ai4s.err.log"
    },
    {
      ...syncDefaults,
      name: "sync-mmlongbench-magerag-results-to-mm",
      script: "scripts/sync_magerag_results_to_remote_loop.sh",
      env: {
        LOCAL_CODE_DIR: codeCwd,
        REMOTE_CODE_DEST: "MM:/root/autodl-tmp/ylz/NeurIPS_2026/code",
        REMOTE_SSH_HOST: "MM",
        RESULTS_RELATIVE_DIR: "results/mmlongbench/magerag",
        SYNC_INTERVAL_SECONDS: "10"
      },
      out_file: "logs/pm2-sync-mmlongbench-magerag-results-to-mm.out.log",
      error_file: "logs/pm2-sync-mmlongbench-magerag-results-to-mm.err.log"
    }
  ]
};
