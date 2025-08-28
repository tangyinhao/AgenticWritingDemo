# DeepSeek-R1 大模型本地部署的三种方式，总有一种适合你

由于 DeepSeek-R1 爆火，导致 DeepSeek 官网用起来非常卡（至 2025 年 2 月 2 日），因此催生出了很多本地部署的需求。这里我们选用了三种最常用的部署方式，从普通人测试使用到工业界部署，让你一次性掌握大模型的部署方式。

## 对比总结

| 特性          | Ollama                     | LM Studio                   | vLLM                        |
|---------------|----------------------------|-----------------------------|-----------------------------|
| **定位**      | 本地快速体验               | 图形化交互工具              | 生产级推理引擎              |
| **用户群体**  | 开发者/爱好者              | 非技术用户                  | 企业/工程师                 |
| **部署复杂度**| 低                         | 极低                        | 中高                        |
| **性能优化**  | 基础                       | 一般                        | 极致                        |
| **适用场景**  | 开发测试、原型验证         | 个人使用、教育演示          | 高并发生产环境              |
| **扩展性**    | 有限                       | 无                          | 强（分布式/云原生）         |

### 建议
- **想快速体验模型**：Ollama  
- **需要图形界面和隐私保护**：LM Studio  
- **企业级高并发需求**：vLLM  

## 方式 1：Ollama

Ollama 是一个开源的本地化大模型部署工具，旨在简化大型语言模型（LLM）的安装、运行和管理。它支持多种模型架构，并提供与 OpenAI 兼容的 API 接口，适合开发者和企业快速搭建私有化 AI 服务。

### Ollama 的特点
- **轻量化部署**：完全的本地化部署。  
- **多模型支持**：兼容各种开源模型，包括 qwen、deepseek、LLaMA 等。  
- **跨平台支持**：支持主流的 Windows、Mac、Linux。  

### 使用 Ollama 安装 DeepSeek-R1 等大模型

一共就三个步骤：

#### 步骤 1：下载 Ollama
- Windows 和 Mac：进入 https://ollama.com/download 下载对应的安装包，然后安装即可。  
- Linux：使用以下命令安装：
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

#### 步骤 2：启动 Ollama
- Mac 和 Windows：点击启动即可（一般默认启动）。  
- Linux：理论上安装脚本会自动启动 Ollama，但如果发生意外，可使用以下命令启动：
  ```bash
  ollama serve
  ```

#### 步骤 3：运行对应的模型
- 运行 DeepSeek-R1 模型（这里选用 DeepSeek-R1 的蒸馏小模型 `deepseek-r1:1.5b`）：
  ```bash
  ollama run deepseek-r1:1.5b
  ```
- 如果使用其他模型：
  ```bash
  ollama run {model_name}
  ```
  `{model_name}` 替换成真实的模型名字，名字可以在 https://ollama.com/search 中获取。  

**注意⚠️**：这样启动模型具有对应的上下文，本质上是启动了 chat 的接口。  
**效果**：可以看到模型具有 think 的能力，但由于模型较小，效果依然不是特别好。  

**总结**：Ollama 比较适合本地快速测试大模型。

## 方式 2：LM Studio

LM Studio 是一款桌面应用程序，用于在您的计算机上开发和试验 LLMs。

### LM Studio 的特点
- 运行本地 LLMs 的桌面应用程序  
- 熟悉的聊天界面  
- 搜索和下载功能（通过 Hugging Face 🤗）  
- 可以侦听类似 OpenAI 端点的本地服务器  
- 用于管理本地模型和配置的系统  
- LM Studio 是一个可视化的软件，基本上没有任何学习成本。

**总结**：LM Studio 最适合普通人使用，没有任何的使用成本，全部都是可视化操作。比如适合个人学习、内容创作、教育演示。以及需要隐私保护的本地对话场景。

## 方式 3：vLLM

vLLM 是加州大学伯克利分校开发的高性能大模型推理框架，专为生产环境优化，支持分布式部署和极低延迟的模型服务，适合企业级应用。这也是业界使用最多的推理框架之一。如果你需要稳定的、优化性能更强的、社区支持更好的推理框架，vLLM 是不二之选。

### vLLM 的特点
- **极致性能**：通过各种算法优化，显著提升吞吐量。  
- **生产级功能**：动态批处理（Continuous Batching）、分布式推理、多 GPU 并行。  
- **模型兼容性**：支持 HuggingFace Transformers 模型架构（如 Llama、DeepSeek 等）。  
- **开放生态**：与 OpenAI API 兼容，可无缝替换商业模型。  

通过 vLLM 对外提供一个服务，你可以一次部署，在多个地方使用（如家里、公司、甚至星巴克），也可以将服务提供给其他人。

### 安装和使用
通过以下命令安装 vLLM：
```bash
pip install vllm
```

一般有两种使用方式：

#### 方式 1：提供 OpenAI 的 API 接口的 Server

以下以 AIStackDC 平台演示（当然也可以本地使用）。AIStackDC 是一个云服务器平台，最大的特点是便宜好用。如果使用我的邀请链接：https://aistackdc.com/phone-register?invite_code=D872A9，可以额外获得 1 张 1 折（5 小时）和几张 5 折（36 小时）优惠券。

##### 服务器启动

1. 在容器管理中，点击创建实例，选择 4090 和 PyTorch 2.30 以及 Python 3.12 对应的镜像。  

2. 进入 Jupyter Lab（类似 Jupyter Notebook）：  

3. 选择一个 Terminal 进入命令行，安装 vLLM：
   ```bash
   pip install vllm
   ```
   如果执行失败，可使用阿里云镜像：
   ```bash
   pip install vllm --index-url https://mirrors.aliyun.com/pypi/simple/
   ```

4. 安装完毕后，启动 vLLM 服务（以 DeepSeek-R1 的蒸馏模型为例）：
   ```bash
   HF_ENDPOINT=https://hf-mirror.com vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```
   其中 `HF_ENDPOINT` 是为了设置代理，以便在国内下载更快。  
   后面的 `vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 才是真正的启动命令。

##### 服务器中使用

###### 方法 1：使用 curl 调用 generate 接口
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

###### 方法 2：代码实现
```python
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```

##### 本地电脑（Mac/Windows）中使用
由于 AIStackDC 平台提供对应的 IP，可以通过 SSH 登录
在本地电脑执行以下命令：
```bash
ssh -i <SSH私钥文件路径> -CNg -L <本地监听端口>:127.0.0.1:<容器内服务监听端口> root@221.178.84.158 -p <实例SSH连接端口>
```
- **SSH私钥文件路径**：本地存放 SSH 私钥文件的路径  
- **容器内服务监听端口**：容器实例中启动的服务监听的端口  
- **本地监听端口**：本地监听的转发端口（自定义，确保本地系统中该端口未被占用）  
- **实例SSH连接端口**：SSH 连接容器实例使用的端口  

**举例**：
```bash
ssh -i id_rsa -CNg -L 9000:127.0.0.1:8000 root@221.178.85.21 -p 34538
```

执行后，可在本地通过 `localhost:9000` 配置到 Open-WebUI 等聊天界面中。

#### 方式 2：批量离线推理
使用 Python API 在代码内部加载模型，进行定制化开发。如果你采用这种方式，说明你有足够的能力完成模型的部署和开发。

##### Example
```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```