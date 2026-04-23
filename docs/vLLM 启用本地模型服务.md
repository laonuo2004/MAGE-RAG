# vLLM 启用本地模型服务

首先 attach 到之前已有的 tmux 终端：

```bash
tmux attach -t vllm
```

如果终端已经关闭了，可以重新创建一个 tmux 终端：

```bash
tmux new -s vllm
```

使用 [启动脚本](../code/scripts/serve_qwen25_vl_vllm.sh) 启动 vLLM 模型服务：

```bash
bash scripts/serve_qwen25_vl_vllm.sh throughput # 侧重吞吐量
bash scripts/serve_qwen25_vl_vllm.sh longctx # 均衡
bash scripts/serve_qwen25_vl_vllm.sh maxctx # 最长上下文
```

其他参数说明可以参考 [vLLM 官方文档](https://docs.vllm.ai/en/stable/configuration/engine_args/#modelconfig)