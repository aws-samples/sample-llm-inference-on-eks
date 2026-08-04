# Kimi-K3 在 B300 单机部署的性能表现

本文陈述 `moonshotai/Kimi-K3` 在单台 p6-b300.48xlarge 上以张量并行 TP8 部署的实测
性能。所有数据于 2026-07-29 在 EKS 集群实测取得，仅陈述观测事实。

英文版见 [KIMI-K3-B300-PERFORMANCE.md](KIMI-K3-B300-PERFORMANCE.md)。

## 一、测试对象与环境

模型为 Kimi-K3，2.8 万亿参数的混合专家（MoE）架构，每 token 激活 896 个路由专家中的
16 个并叠加共享专家，采用 Kimi Delta Attention 与 Attention Residuals，原生支持视觉输入，
上下文窗口 1,048,576 token。权重为量化感知训练的 MXFP4 格式，磁盘占用约 1.56 TB，
分布于 120 个文件。

硬件为一台 p6-b300.48xlarge：8 张 B300（每张 268 GB HBM3e，合计 2,149 GiB 显存）、
192 个 vCPU、4 TiB 主机内存、约 28 TiB 本地 NVMe（8 盘 RAID）、16 个 EFA 接口。
该机型在 us-west-2 仅 us-west-2a 可用区提供，本次通过 Capacity Block 预留取得
（Karpenter capacity-type 为 `reserved`）。

推理引擎为 vLLM，镜像 `vllm/vllm-openai:kimi-k3`，实际解析到版本
`0.1.dev19262+gb6bbf29dd.d20260727`。启动参数采用官方 recipe 的 B300 profile：
TP8、`--gpu-memory-utilization 0.95`、`--max-model-len 1048576`、
`--kv-cache-dtype fp8`、MLA prefill 后端 `TRTLLM_RAGGED` 并开启
`use_prefill_query_quantization`、`--enable-prefix-caching`、
`--load-format fastsafetensors`、`--moe-backend auto`。

压测客户端为 NVIDIA genai-perf 0.0.16.post1，运行于集群内 Triton 26.06 SDK pod，
经 ClusterIP 访问服务的流式 `/v1/chat/completions` 接口。

## 二、显存占用

在 `--gpu-memory-utilization 0.95` 下，引擎报告的每张 GPU 实测分配如下：

| 项目 | 占用 |
|---|---|
| 权重及非 torch 部分 | 195.8 GiB |
| KV cache | 55.0 GiB |
| CUDA Graph | 4.7 GiB |
| 峰值激活 | 3.5 GiB |

2.8T 参数的 MXFP4 权重在 TP8 下每卡承担约 196 GiB，单机 8 卡可完整容纳，无需多机。
在此前提下 `--max-model-len=1048576`（完整 100 万 token 上下文）的 KV cache 分配成功。
引擎另行提示，将 KV cache 显式设为 61.02 GiB（`--kv-cache-memory=65514832384`）
可进一步用尽显存，较当前 55.0 GiB 增加约 11%。

## 三、启动耗时

节点就绪后的冷启动各阶段耗时：

| 阶段 | 耗时 |
|---|---|
| 拉取容器镜像 | 约 2 分钟 |
| 下载 1.5 TB 权重至本地 NVMe | 约 10 分钟（约 2.3 GB/s） |
| 引擎初始化（加载、图捕获、KV 分配） | 499 秒 |
| **合计** | **约 22 分钟** |

CUDA Graph 捕获在 8 个 TP worker 上各耗时 78–82 秒。整个测试过程中容器重启次数为 0。
权重驻留于节点本地 NVMe，同节点上的 pod 重启无需重新下载。

## 四、性能测试方法

共测试两组负载：输入 1024 / 输出 1024 token（1:1 均衡，并发 1–256），以及输入 8000 /
输出 1024 token（8:1，prefill 密集，并发 8–64）。两组均为本次测试选定。
输出长度通过 `max_tokens:1024` 与 `ignore_eos:true` 锁定，所有请求均以
`finish_reason: length` 结束，无一为 `stop`。思考模式经
`{"chat_template_kwargs":{"enable_thinking":false}}` 关闭，实测响应中 `reasoning`
字段为 null。输入长度实测落在 1023.97–1024.00（各档均值），与设定一致。

> [!CAUTION]
> **下述深度流程并不产生稳态值** —— 2026-08-03 对照 perf_analyzer 源码复核后更正。
> 两个缺陷都影响本文全部数字：
>
> - **`--request-count` 会把整轮压缩成单个测量窗口**
>   （`inference_profiler.cc`：`// If request-count is specified, then only measure one
>   window and exit`），因此下面各档无法排除 ramp-up —— 深度是固定了，但用于裁掉
>   ramp-up 的窗口结构不存在了。
> - 即便是 c64 那次标定运行，也是对**全部**窗口（含 ramp-up）汇总的，因为
>   `--stability-percentage` 只决定 perf_analyzer 何时停止，不决定它报告什么。
>
> 所以这些是**固定深度的全程平均值，不是稳态测量**；在另一套硬件上，同类污染使
> TTFT p50 偏低 5–19%。
>
> **跨并发点的结论是暂定的，但原因不是各档深度不同** —— 闭环负载下完成请求数是结果、
> 不是混淆因子。真正的原因是**这些点都没有排除 ramp-up**：每档只有一个窗口，因此不存在
> 可截取的稳态子集，每个数字都混合了队列填充瞬态与稳态；而窗口中瞬态所占**比例**随并发
> 变化（队列越深填满越慢）。所以凡是**跨并发点**比较的结论（效率拐点、吞吐曲线形状）受
> 影响程度不一，应视为暂定。各档同时也是 n=1、无方差估计。各档数字按实测保留。
> 当前流程见 [BENCHMARK-METHODOLOGY.md](BENCHMARK-METHODOLOGY.md)。

测量深度固定为各档共用的同一值：先在最深队列（并发 64）启用运行时稳态检测
（`--measurement-interval 120000 --stability-percentage 10`），观测到
`Request Count` 为 768，即每个并发槽 12 个请求；其余各档均以
`--request-count = 12 × 并发数` 固定同一深度，实际为 12/96/192/384/768。
每档前有 10–20 个预热请求，不计入测量；prompt 池为 2000 条，大于总请求数，
避免复用抬高前缀缓存命中率。

## 五、性能数据

### 5.1 均衡负载 1024/1024

| 并发 | TTFT 均值 | TTFT p90 | TTFT p99 | ITL 均值 | 单用户输出 (tok/s) | 请求吞吐 (req/s) | 总生成吞吐 (tok/s) |
|---|---|---|---|---|---|---|---|
| 1   | 241 ms   | 243 ms   | 253 ms    | 9.45 ms  | 105.8 | 0.10 | 103 |
| 8   | 669 ms   | 816 ms   | 819 ms    | 14.57 ms | 68.8  | 0.52 | 516 |
| 16  | 667 ms   | 769 ms   | 1,019 ms  | 17.84 ms | 56.1  | 0.84 | 840 |
| 32  | 986 ms   | 1,165 ms | 1,804 ms  | 24.12 ms | 41.6  | 1.26 | 1,251 |
| 64  | 1,128 ms | 1,278 ms | 3,266 ms  | 32.98 ms | 30.6  | 1.85 | 2,064 |
| 128 | 1,472 ms | 1,730 ms | 6,587 ms  | 43.86 ms | 23.2  | 2.80 | 3,290 |
| 256 | 1,969 ms | 1,987 ms | 13,166 ms | 69.26 ms | 14.9  | 3.64 | 4,634 |

TTFT 为首 token 延迟，ITL 为 token 间延迟，均由客户端侧测得。总生成吞吐取自引擎自身
`log_stats` 上报的稳态中位值（每档 18–64 个样本，`Reqs Running` 与目标并发一致）。

### 5.2 prefill 密集负载 8000/1024

在同一部署上以相同测量深度（12 requests/slot）跑第二组负载，输入 8000 token、输出
1024 token（8:1）。实测输入长度落在 7,999.91–7,999.96。两组负载对照：

| 并发 | **1K/1K** 吞吐 / TTFT / ITL | **8K/1K** 吞吐 / TTFT / ITL |
|---|---|---|
| 8  | 516 / 669 ms / 14.6 ms     | 478 / 1,390 ms / 18.1 ms |
| 16 | 840 / 667 ms / 17.8 ms     | 870 / 1,228 ms / 22.3 ms |
| 32 | 1,251 / 986 ms / 24.1 ms   | 1,267 / 1,932 ms / 34.2 ms |
| 64 | 2,064 / 1,128 ms / 33.0 ms | **82** / 5,106 ms / 114.5 ms |

8K 负载在 c64 的 TTFT p99 为 18,972 ms，ITL 最坏值 1,821.89 ms，请求吞吐 0.68 req/s
（低于同负载 c32 的 0.98 req/s）。

## 六、数据呈现的规律

**1K/1K 负载的效率拐点在并发 256。** 每次并发翻倍带来的吞吐增幅与 ITL 增幅之比：
16→32 为 +49% / +35%、32→64 为 +65% / +37%、64→128 为 +59% / +33%，均为吞吐增幅
大于延迟代价；128→256 首次反转，+41% 吞吐对应 +58% ITL。吞吐绝对值仍在上升
（3,290 → 4,634 tok/s），并非硬上限。并发 512 未测试。

**并行效率随并发下降。** 以并发 1 为基准，各档吞吐倍数与并发倍数之比为：
并发 8 达 5.01 倍（效率 62.6%）、并发 16 达 8.16 倍（51.0%）、并发 32 达 12.15 倍
（38.0%）、并发 64 达 20.04 倍（31.3%）、并发 128 达 31.9 倍（24.9%）、并发 256 达
45.0 倍（17.6%）。折算每 GPU 吞吐由并发 1 的 12.9 tok/s 升至并发 256 的 579.2 tok/s。

**均值延迟增速低于负载增速，尾部则不然。** 并发从 1 增至 256（256 倍）时，TTFT 均值
增长 8.2 倍（241 → 1,969 ms），ITL 增长 7.3 倍（9.45 → 69.26 ms）。但 TTFT p99 由
并发 32 的 1,804 ms 升至并发 256 的 13,166 ms，ITL 最坏值由 48.94 ms 升至 455.76 ms；
p99 与均值之比从 1.8 倍扩大到 6.7 倍。

**8K 输入下，并发 32 以内的输出吞吐与 1K 输入基本相同。** 8K 负载在并发 8/16/32 的
吞吐分别为 1K 负载的 93%、104%、101%。代价体现在延迟：TTFT 约为 2 倍，ITL 为
1.2–1.4 倍。

**8K 输入在并发 64 出现吞吐倒退。** 总吞吐降至 82 tok/s（1K 负载同并发的 4%），
请求吞吐由并发 32 的 0.98 req/s 降至 0.68 req/s，TTFT 均值 5,106 ms、ITL 均值
114.51 ms。该并发下引擎吞吐采样呈双峰分布：中位数 82 tok/s，而 p75 为 1,836 tok/s、
最大 2,112 tok/s（108 个样本）。并发 48 未测试，故退化边界未定位。

**上述退化期间 KV cache 使用率峰值为 20.8%，preemption 计数为 0。** 稳态期
`Reqs Waiting` 在 1K 负载各档均为 0；8K 负载共 9 个采样出现排队。

**并发 8 与并发 16 的延迟基本相同。** 两者 TTFT 均值为 669 ms 与 667 ms，
p90 为 816 ms 与 769 ms，而并发 16 的总吞吐比并发 8 高 63%（840 对 516 tok/s）。

**单流解码速率为 105.8 tok/s。** 并发 1 时 ITL 9.45 ms，且分布极窄
（p1 至 max 均为 9.45–9.46 ms）。

**前缀缓存未生效于本负载。** 尽管开启 `--enable-prefix-caching`，全程实测
前缀缓存命中率为 0.0%，即上述结果未受缓存复用影响。

## 七、功能性观测

**工具调用在当前配置下不生效。** B300 recipe profile 设定
`VLLM_USE_RUST_FRONTEND=1`，该前端启动时明确记录忽略 `enable_auto_tool_choice`
与 `structured_outputs_config` 两个参数，因此 `--tool-call-parser kimi_k3` 与
`--enable-auto-tool-choice` 处于失效状态。需将 `VLLM_USE_RUST_FRONTEND` 置 0
方可使用工具调用。同时记录 Model Runner V2 不支持 `thinking_token_budget` 请求参数。

**思考模式默认开启，可关闭。** 默认状态下模型输出进入 `reasoning` 字段而
`content` 为 null；经嵌套形式 `chat_template_kwargs.enable_thinking=false` 可关闭，
平铺的 `enable_thinking` 无效。关闭后 `reasoning` 为 null，`content` 正常返回。

**视觉能力已加载，视频模态未支持。** 前端日志显示 chat 后端以 `kimi_k3` renderer
加载完成，video 模态因 placeholder token 无法解析而被禁用。

**功能验证。** 算术问题（84 × 3 ÷ 2）返回正确结果 126，`finish_reason` 为 `stop`，
思考内容与最终回答由 `kimi_k3` parser 正确分离。

## 八、测试范围与未覆盖项

本文数据的适用范围限于：单机 TP8、关闭思考模式、1024/1024 负载并发 1 至 256、
8000/1024 负载并发 8 至 64。

未覆盖：1K 负载并发 512 及以上；8K 负载并发 48（退化边界未定位）与并发 128 及以上；
更长输入（最长测试输入为 8000 token，而窗口为 100 万 token）；decode 密集负载
（如 1K 输入 / 4K 输出）；开启思考模式下的端到端表现；投机解码
（`Inferact/Kimi-K3-DSpark` 草稿模型方案会将 `--max-num-seqs` 限制为 32，本次未部署）；
多机 TEP16 与 PD 分离部署形态；将 KV cache 提升至 61.02 GiB 的对比测试。

本次部署所用 manifest 为
[`k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml`](../k8s-manifest/vllm/kimi-k3-p6-b300-vllm.yaml)。
