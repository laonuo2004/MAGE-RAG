# MMLongBench Benchmark 中文使用说明

## 1. 这份文档是干什么的

这份文档讲的是我们现在仓库里 `mmlongbench` 这套 benchmark 怎么跑、代码改了哪些地方、以及后面 baseline 应该怎么接进来。

当前主要相关文件：

- `code/benchmarks/mmlongbench/run_api.py`
- `code/benchmarks/mmlongbench/run_api_text.py`
- `code/benchmarks/mmlongbench/eval/extract_answer.py`
- `code/benchmarks/mmlongbench/eval/eval_score.py`
- `code/benchmarks/mmlongbench/env_utils.py`
- `code/benchmarks/mmlongbench/route_utils.py`

现在这套代码的目标很简单：

1. 支持本地/远程 OpenAI-compatible 接口，比如 vLLM、LiteLLM、OpenRouter。
2. 支持两条路线：
   - 图片输入路线
   - OCR/Text 输入路线
3. 支持断点续跑。
4. 支持失败样本自动重跑。
5. 支持多端口、多上下文档位自动切换。
6. 支持并发跑 benchmark。

---

## 2. 当前目录结构

主要目录：

```text
code/benchmarks/mmlongbench/
├── data/
│   ├── samples.json
│   └── documents/        # PDF 文件
├── eval/
│   ├── eval_score.py
│   ├── extract_answer.py
│   └── prompt_for_answer_extraction.md
├── env_utils.py
├── route_utils.py
├── run_api.py
├── run_api_text.py
├── .env.mmlongbench.example
├── MMLONGBENCH_BENCHMARK_GUIDE.md
└── MMLONGBENCH_BENCHMARK_GUIDE_CN.md
```

运行时生成：

```text
tmp/       # PDF 转图片缓存
results/   # 每题结果 JSON + 汇总 txt
logs/      # 运行日志
```

---

## 3. 现在有哪两条路线

## 3.1 图片路线

入口文件：

- `run_api.py`

流程：

1. 从 `data/samples.json` 读 benchmark 题目。
2. 按样本找到对应 PDF。
3. 用 PyMuPDF 把 PDF 前 `max_pages` 页转成图片。
4. 把“问题 + 多页图片”一起发给多模态模型。
5. 得到自由回答后，再做一次答案抽取。
6. 把短答案和标准答案比对，算分。
7. 每做完一题就落盘到结果 JSON。

这条路线是 `mmlongbench` 的主路线，因为很多题依赖：

1. 图表
2. 表格
3. 页面布局
4. 跨页视觉信息

## 3.2 OCR/Text 路线

入口文件：

- `run_api_text.py`

流程：

1. 从 `samples.json` 读题。
2. 打开 PDF。
3. 用 `page.get_text("text")` 抽每页文本。
4. 把多页文本拼成一个大 prompt。
5. 发给语言模型回答。
6. 再做二次答案抽取。
7. 和标准答案比对算分。

这条路线更像 baseline / 对照组，一般会比图片路线弱，特别是图表题、表格题。

---

## 4. 常用参数怎么理解

### 4.1 两条路线共有参数

- `--input_path`
  benchmark 题目文件，默认 `./data/samples.json`

- `--document_path`
  PDF 所在目录，默认 `./data/documents`

- `--model_name`
  主回答模型名

- `--base_url`
  单一路由时用的 API 地址

- `--api_key`
  单一路由时用的 key

- `--extractor_model_name`
  二次抽取模型名

- `--extractor_base_url`
  二次抽取模型接口

- `--extractor_api_key`
  二次抽取 key

- `--max_try`
  单题最大重试次数

- `--num_workers`
  并发数

- `--output_path`
  输出 JSON 文件

- `--limit`
  只跑前几条，用于 smoke test

### 4.2 图片路线特有参数

- `--max_pages`
  当前逻辑是“截 PDF 前 N 页”

- `--resolution`
  PDF 转图片时的 DPI

重点：

1. `144` 比 `72` 每边大约 2 倍，总像素大约 4 倍。
2. 分辨率越高，图越清楚，但 token 越大，更容易超上下文。

### 4.3 OCR/Text 路线特有参数

- `--max_pages`
  取前 N 页文本拼 prompt

---

## 5. 断点续跑现在是怎么做的

我们已经把原版比较简单的续跑逻辑改成了更稳的版本。

现在流程是：

1. 先从 `samples.json` 确定“本次要跑哪些题”。
2. 如果传了 `--limit`，就先截到前 N 条。
3. 如果已经存在结果文件，就按样本键去合并老结果：
   - `doc_id`
   - `question`
   - `answer`
   - `answer_format`
4. 已经完成的跳过。
5. 失败的继续重跑。

这意味着：

1. 先跑 `limit=1`，再跑 `limit=5`，现在逻辑是对的。
2. 不用手动删旧文件才能续跑。
3. 失败样本会自动补跑。

当前状态字段：

- `status`
  - `completed`
  - `failed_generation`
  - `failed_extraction`

- `failure_stage`
  - `generation`
  - `extraction`

- `error`
  保存异常信息

---

## 6. 多端口自动切换是怎么做的

## 6.1 为什么要加这层

LiteLLM 的 `4000` 统一入口在工程上很方便，但目前并不能稳定做到：

1. 先走 throughput
2. 上下文超了再自动切 longctx
3. 再切 maxctx

所以现在 benchmark 代码自己加了一层“路由 fallback”。

## 6.2 通过哪些环境变量控制

- `ROUTE_BASE_URLS`
- `ROUTE_MODEL_NAMES`
- `ROUTE_API_KEYS`
- `ROUTE_LABELS`
- `ROUTE_MAX_MODEL_LENS`

典型顺序是：

1. throughput 1
2. throughput 2
3. longctx 1
4. longctx 2
5. maxctx

例如：

```bash
ROUTE_BASE_URLS=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8000/v1
ROUTE_MODEL_NAMES=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
ROUTE_API_KEYS=sk-123,sk-123,sk-123,sk-123,sk-123
ROUTE_LABELS=tp1,tp2,long1,long2,max1
ROUTE_MAX_MODEL_LENS=32768,32768,65536,65536,128000
```

## 6.3 什么时候会自动切到更大的端口

当报错里出现这些信息时：

1. `maximum context length`
2. `Input length ... exceeds ...`
3. `too many tokens`

代码会认为这是上下文超限，然后尝试切到更大的端口。

## 6.4 日志和结果里会记录什么

每题会记录：

- `used_route_label`
- `used_base_url`
- `used_model_name`
- `used_route_max_model_len`

这几个字段组会时非常好讲，因为你能清楚说明这一题最终用了哪个上下文档。

---

## 7. 并发是怎么做的

两条路线都支持：

- `--num_workers`

执行方式是：

1. worker 线程并发处理题目
2. 主线程收集 future 结果
3. 每完成一题，主线程统一更新结果 JSON

这样做的好处：

1. 比串行快
2. 不会多线程同时写同一个 JSON
3. 断点续跑依然稳定

建议：

1. smoke：`1` 或 `2`
2. 正常跑：`2` 或 `4`
3. 再往上提之前，先确认服务稳定

---

## 8. 现在日志里的耗时字段是什么意思

现在会打印四段耗时：

- `prepare`
- `generation`
- `extraction`
- `total`

解释：

1. `prepare`
   - 图片路线：PDF 转图、读缓存、base64 编码
   - text 路线：抽文本、拼 prompt

2. `generation`
   主模型生成耗时

3. `extraction`
   二次抽取耗时

4. `total`
   整题总耗时

这样以后看到“为什么某一题突然慢”，就能一眼判断：

1. 是转图慢
2. 还是主模型慢
3. 还是 extractor 慢

---

## 9. 为什么图片路线 token 会特别大

很多人容易误会成“文字 prompt 很长”，其实不是。

真正大的主要是图片本身：

1. 一道题会把前 `max_pages` 页都转成图片
2. 所有图片和问题一起放进一个请求
3. 每张图片都会被视觉模型展开成大量视觉 token

所以大头不是题目文字，而是：

1. 页数
2. 分辨率
3. 页面复杂度（图表、表格、小字）

当前图片路线页选择逻辑很简单：

- 永远取前 `max_pages` 页

所以现在更像“先把 benchmark 跑通”的版本，还不是最优检索策略。

---

## 10. 为什么 OCR/Text 路线经常回答 `Not answerable`

这是正常现象，不一定是程序坏了。

原因：

1. `mmlongbench` 很多题依赖图表、表格、布局
2. OCR/text 路线会丢掉这些视觉信息
3. prompt 里还明确写了“找不到就答 `Not answerable`”

所以 text 路线经常更像一个保守 baseline，而不是主方法。

---

## 11. 现在推荐的 `.env.mmlongbench` 配置

## 11.1 方案 B：直连多端口，benchmark 自己切换（推荐）

```bash
MODEL_NAME=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
OPENROUTER_API_KEY=sk-123

ROUTE_BASE_URLS=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8000/v1
ROUTE_MODEL_NAMES=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct,/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
ROUTE_API_KEYS=sk-123,sk-123,sk-123,sk-123,sk-123
ROUTE_LABELS=tp1,tp2,long1,long2,max1
ROUTE_MAX_MODEL_LENS=32768,32768,65536,65536,128000

EXTRACTOR_MODEL_NAME=/data2/jiangjiaqi/ylz/Qwen2.5-VL-7B-Instruct
EXTRACTOR_BASE_URL=http://127.0.0.1:8002/v1
EXTRACTOR_API_KEY=sk-123
```

这套的意思是：

1. 主回答自己按端口升级
2. extractor 固定到 `8002`
3. 不依赖 LiteLLM 自动理解上下文长度

## 11.2 方案 A：统一走 LiteLLM `4000`

只有在你们确认 LiteLLM 的 fallback 配好以后，才建议用。

```bash
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
OPENROUTER_BASE_URL=http://127.0.0.1:4000/v1
OPENROUTER_API_KEY=sk-123

EXTRACTOR_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
EXTRACTOR_BASE_URL=http://127.0.0.1:4000/v1
EXTRACTOR_API_KEY=sk-123
```

当前这条路线的问题是：

1. 管理上更干净
2. 但自动升级到长上下文实例还不够稳定

所以现阶段更推荐方案 B。

---

## 12. 常用命令

## 12.1 图片路线 smoke

```bash
python -u ./run_api.py --num_workers 2 --limit 5 --max_pages 5 --resolution 72 --output_path ./results/res_img_router_smoke.json 2>&1 | tee ./logs/mml_img_router_smoke.log
```

## 12.2 OCR/Text 路线 smoke

```bash
python -u ./run_api_text.py --num_workers 2 --limit 5 --max_pages 5 --output_path ./results/res_text_router_smoke.json 2>&1 | tee ./logs/mml_text_router_smoke.log
```

## 12.3 图片路线全量

```bash
python -u ./run_api.py --num_workers 2 --max_pages 5 --resolution 72 --output_path ./results/res_img_router_full.json 2>&1 | tee ./logs/mml_img_router_full.log
```

## 12.4 OCR/Text 路线全量

```bash
python -u ./run_api_text.py --num_workers 2 --max_pages 5 --output_path ./results/res_text_router_full.json 2>&1 | tee ./logs/mml_text_router_full.log
```

---

## 13. 我们相对原版 benchmark 改了什么

### 13.1 `run_api.py`

加了：

1. OpenAI-compatible 接口支持
2. `.env.mmlongbench` 读配置
3. 断点续跑修正
4. 失败样本自动重跑
5. 路由级上下文 fallback
6. 并发执行
7. 耗时日志
8. 路由信息记录

### 13.2 `run_api_text.py`

加了：

1. OCR/Text 路线
2. `.env.mmlongbench`
3. 断点续跑修正
4. 失败样本自动重跑
5. 路由级 fallback
6. 并发执行
7. 耗时日志

### 13.3 `extract_answer.py`

改了：

1. client 可配置
2. 支持自定义 `base_url`
3. 本地 OpenAI-compatible 服务没有 key 时自动补占位值

### 13.4 `eval_score.py`

改了：

1. 更安全地解析列表型答案
2. 避免 `eval()` 把普通字符串当 Python 表达式炸掉

---

## 14. 后面 baseline 应该怎么接

这个部分是组会上最值得讲的。

核心思想：

**尽量不改后处理和评分，只替换“如何得到 response”这一层。**

也就是说，新的 baseline 最好都沿用现有这几层：

1. 样本加载
2. 路由 / 重试 / 失败状态
3. 二次抽取
4. 评分
5. 结果落盘

只替换：

1. 输入怎么构造
2. 模型怎么调用

## 14.1 新的 API baseline

比如再接一个新的 OpenAI-compatible 模型。

做法：

1. 复制 `run_api.py`
2. 保留样本加载、状态管理、抽取、评分
3. 只替换 payload 构造和请求方式

## 14.2 新的 OCR baseline

比如改成 MinerU、PaddleOCR、别的结构化 OCR。

做法：

1. 复制 `run_api_text.py`
2. 替换 `build_text_prompt(...)`
3. 保留抽取和评分流程

## 14.3 带检索的 baseline

比如先检索页，再送模型。

建议新建：

- `run_api_retrieval.py`

流程：

1. 先检索 top-k 页
2. 只用这些页构造 prompt
3. 调模型
4. 二次抽取
5. 评分

额外建议在输出里加：

- `retrieved_pages`
- `retrieval_scores`
- `retrieval_backend`

这样后面分析会很方便。

---

## 15. 组会怎么讲这套代码

最简单的一种讲法是把整套 benchmark 拆成四层：

### 第一层：输入层

1. PDF 从哪里来
2. 是转图片还是抽文本
3. 当前是截前 N 页

### 第二层：推理层

1. 模型接口怎么调
2. 是单路由还是多路由
3. 上下文超限怎么自动切更大端口

### 第三层：后处理层

1. 主模型先自由回答
2. 再做一次短答案抽取

### 第四层：评测层

1. 怎么和 gold 比
2. 怎么算 `score`
3. 怎么算 `Avg acc`
4. 怎么算 `Avg f1`

你后面所有 baseline，其实都可以说成：

1. 保留后三层
2. 只改第一层或第二层

这就是目前代码结构最大的好处。

---

## 16. 现在最值得继续优化的地方

后面如果要继续做工程增强，最值得做的是：

1. extractor 也支持 `8001 -> 8002` fallback
2. 规则抽取优先，LLM extractor 兜底
3. 支持不同页选择模式：
   - 前 N 页
   - gold evidence 调试模式
   - 检索结果页
4. 结果 JSON 损坏时自动备份
5. 按 `resolution` 区分图片缓存

---

## 17. 最短结论

如果你现在只想记住一句话：

**当前 `mmlongbench` 最稳的跑法是：主回答走多端口自动升档，extractor 固定一个 throughput 端口，评分和续跑逻辑统一复用。**
