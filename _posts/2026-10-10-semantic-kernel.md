---
layout: post
title: "Semantic Kernel: The Kernel That Ran Enterprise AI Agents - Inside microsoft/semantic-kernel"
description: "A source-code tour of Semantic Kernel, Microsoft's tri-language agent SDK: the Kernel and its four extension mixins, KernelFunction and the plugin registry, three prompt template engines, auto function invocation filters, ChatCompletionAgent, the magentic and handoff orchestrations, the kernel process framework with local and Dapr runtimes, thirteen vector stores, and the honest transition to Microsoft Agent Framework."
date: 2026-10-10
header-img: "img/post-bg.jpg"
permalink: /semantic-kernel/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/semantic-kernel/microsoft-semantic-kernel-overview-architecture.svg
tags: [AI Agents, Python, Open Source]
categories: [AI, Open Source]
keywords: "semantic kernel, microsoft, agent framework, python sdk, chat completion agent, plugins, prompt templates, magentic, process framework"
author: "PyShine"
---

Before "agent framework" was a product category, Microsoft shipped an SDK with a stranger and better idea: treat the LLM like an operating system kernel. Prompts became functions, your Python methods became plugins, and a single orchestrator scheduled them all. That SDK is Semantic Kernel, MIT-licensed at microsoft/semantic-kernel, and this post tours its source honestly, because honesty is required up front: the README now opens with a banner announcing that Semantic Kernel is succeeded by the Microsoft Agent Framework, an enterprise-ready follow-up that reached version 1.0 with stable APIs, multi-agent orchestration, and cross-runtime interoperability via A2A and MCP. The banner does not diminish the artifact. At Python version 1.45.0, spanning Python 3.10+, .NET 10.0+, and Java 17+ in one repository, Semantic Kernel remains one of the most complete teaching sources for how enterprise AI agents are actually assembled, and its ideas flow directly into the successor.

The repository layout makes the ambition visible. python, dotnet, and java sit side by side as first-class SDKs, with a shared prompt_template_samples folder at the root and a FEATURE_MATRIX that tracks parity across languages. The Python package alone is a dense forest worth walking: agents, connectors, contents, functions, filters, processes, memory, template_engine, prompt_template, services, schema, text, and utils. As always in this series, what follows is an educational tour of published source code, treating the transition banner as a fact to learn from rather than a rumor to repeat.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/semantic-kernel/microsoft-semantic-kernel-overview-architecture.svg" alt="Semantic Kernel overview architecture diagram" style="max-width:100%;"></div>
<p><em>Semantic Kernel at a glance: the Python, .NET, and Java SDKs expose a shared design where the Kernel orchestrates functions and plugins, renders three styles of prompt templates, wraps invocations in filters, hosts chat completion agents and multi-agent orchestrations, runs kernel processes on local or Dapr runtimes, and talks to model providers and vector stores through connectors.</em></p>

Reading the overview from left to right:

- The Python surface is the semantic_kernel package in [python/semantic_kernel/__init__.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/__init__.py), which exports the Kernel and pins the version.
- The .NET implementation lives under [dotnet/src](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src), with Java a sibling tree, both mirroring the same concepts you see here.
- The orchestrator is the Kernel class in [python/semantic_kernel/kernel.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/kernel.py), a Pydantic model composed from four extension mixins.
- The unit of work is KernelFunction in [python/semantic_kernel/functions/kernel_function.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/functions/kernel_function.py), held in registries shaped by [python/semantic_kernel/functions/kernel_plugin.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/functions/kernel_plugin.py).
- Prompt rendering supports three syntaxes through [python/semantic_kernel/prompt_template/kernel_prompt_template.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/prompt_template/kernel_prompt_template.py) and its Handlebars and Jinja2 siblings.
- Invocation hooks live in [python/semantic_kernel/filters/kernel_filters_extension.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/filters/kernel_filters_extension.py), including the auto function invocation machinery that lets a model call your tools.
- Agents are exemplified by ChatCompletionAgent in [python/semantic_kernel/agents/chat_completion/chat_completion_agent.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/agents/chat_completion/chat_completion_agent.py), coordinated by the orchestration module in [python/semantic_kernel/agents/orchestration/orchestration_base.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/agents/orchestration/orchestration_base.py) and the process framework in [python/semantic_kernel/processes/kernel_process/kernel_process.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/processes/kernel_process/kernel_process.py).
- Model and memory access run through the OpenAI connector family in [python/semantic_kernel/connectors/ai/open_ai/services/open_ai_chat_completion_base.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_chat_completion_base.py) and the vector stores in [python/semantic_kernel/connectors/memory_stores](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/connectors/memory_stores).

## Why You Need This

The first reason is the kernel abstraction itself, and kernel.py earns the metaphor. The class declares itself as class Kernel, inheriting KernelFilterExtension, KernelFunctionExtension, KernelServicesExtension, and KernelReliabilityExtension, four mixins that each own one concern: hooking invocations, finding functions, resolving AI services, and retry policy. The invoke methods accept either a function object or a plugin and function name pair, build KernelArguments from kwargs, and delegate to function.invoke with the kernel passed back in, so every function can reach services, plugins, and filters during execution. invoke_stream carries a subtle rule worth reading in the docstring: when multiple functions are provided they execute in order, and only the last one is streamed, the earlier ones running first with their outputs available as context. There is even clone, which deep-copies the plugin dictionary, and as_mcp_server, which turns the kernel's function collection into a Model Context Protocol server, a 2026-era exit ramp that shows how the design anticipated the tool-protocol world.

The second reason is the function system, which is the cleanest explanation you will find of how an LLM sees your code. The @kernel_function decorator in kernel_function_decorator.py inspects a Python method's signature and docstring and emits KernelParameterMetadata entries, which is exactly what gets serialized into the tool schema a model reasons over. KernelFunctionFromMethod wraps the callable, while KernelFunctionFromPrompt, at twenty thousand bytes, turns a prompt template itself into a first-class function with its own execution settings, so a prompt can be invoked, chained, and reused like code. KernelPlugin then groups functions into named registries that Kernel.add_plugin accepts. The punchline of the design is automatic tool use: after a model response contains function calls, invoke_function_call resolves each requested name against the plugin registry, executes it, and feeds results back, with the filter extension in filters/auto_function_invocation letting you intercept, modify, or terminate that loop.

The third reason is breadth made principled. The connectors/ai folder speaks to OpenAI and Azure OpenAI through a shared open_ai_chat_completion_base with a dedicated open_ai_handler request layer, plus Anthropic, Google AI and Vertex AI, MistralAI, HuggingFace, Ollama, ONNX Runtime, NVIDIA NIM, and AWS Bedrock. The memory_stores folder carries thirteen vector stores from Qdrant, Pinecone, Redis, Weaviate, Chroma, Milvus, and MongoDB Atlas to Azure AI Search, Cosmos DB, PostgreSQL, and AstraDB. The agents folder goes beyond the chat completion agent to OpenAI Assistants, Azure AI Foundry agents, Bedrock agents, Copilot Studio, and even an AutoGen bridge. And the orchestration module implements four distinct multi-agent patterns, sequential, concurrent, handoffs, and magentic, the last at thirty-six thousand bytes implementing the Magentic-One style coordination with its actor model. Very few repositories let you compare this many integration surfaces under one consistent contract.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/semantic-kernel/microsoft-semantic-kernel-architecture.svg" alt="Semantic Kernel detailed architecture diagram" style="max-width:100%;"></div>
<p><em>The detailed view: the three SDK surfaces, the Kernel with its mixins and arguments, the function and plugin machinery with template engines, the agent families and orchestration patterns, the kernel process framework with local and Dapr runtimes, the OpenAI connector stack and alternative providers, and the memory layer with its vector stores.</em></p>

The detailed diagram rewards a slow pass. In the function group, KernelPromptTemplate renders the native Semantic Kernel syntax, where variable_block references like a dollar-brace placeholder pull values from KernelArguments and code_block braces invoke plugin functions inline, with the tokenizer split between template_tokenizer and code_tokenizer and the whole block parser living in template_engine/blocks. Handlebars and Jinja2 templates plug into the same PromptTemplateConfig so teams can choose their syntax per prompt. In the agent group, ChatCompletionAgent extends the Agent base, holds a ChatHistoryAgentThread for state, and exposes get_response, invoke, and invoke_stream; AgentGroupChat layers selection and termination strategies on top, and the orchestration actors coordinate members through a broadcast queue.

The process group is the part most enterprise architects overlook and should not. A KernelProcess is a graph of KernelProcessStep types connected by typed edges and events; steps carry state that can be persisted and restored; the local_runtime executes the graph in-process, while the dapr_runtime version maps steps onto Dapr actors for distributed execution. This is durable workflow modeling, not chat plumbing, and reading kernel_process_step_state next to kernel_process_edge teaches more about event-driven design than many dedicated workflow engines. In the connector group, the realtime module at forty-five thousand bytes implements the WebSocket-based audio session clients for OpenAI and Azure, and the memory group pairs SemanticTextMemory with a VolatileMemoryStore for tests and the thirteen production stores for everything else.

## From Install to First Agent

The README path is short. Install the Python package and run the basic agent:

```bash
pip install semantic-kernel
export AZURE_OPENAI_API_KEY=...
```

```python
import asyncio
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

async def main():
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="SK-Assistant",
        instructions="You are a helpful assistant.",
    )
    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")
    print(response.content)

asyncio.run(main())
```

On .NET the same shape appears as Kernel.CreateBuilder, AddAzureOpenAIChatCompletion, and a ChatCompletionAgent with a Kernel attached, which is the whole point of the tri-language design: one mental model, three grammars. To give the agent tools, write a class, decorate methods with @kernel_function, and pass the instance to the agent or add it via add_plugin, and the model will discover the signatures automatically. Prompt-only functions, Handlebars or Jinja2 templates, OpenAPI-specified plugins, MCP servers as tool sources, and the process framework are all documented with runnable samples under prompt_template_samples in the repository root. For .NET, the packages are installed with dotnet add package Microsoft.SemanticKernel and the agents package alongside it.

The honest closing of this tour belongs to the banner again. If you are starting a new enterprise project in 2026, the Microsoft Agent Framework is the documented destination, and the migration guide maps each Semantic Kernel concept to its successor. But frameworks do not appear fully grown; they accrete, and this repository is the accretion record of four years of production agent engineering, from the kernel metaphor to plugins to filters to processes to multi-agent orchestration. Read it the way you would read a well-kept operating system codebase: the abstractions are the API, the connectors are the drivers, and the process framework is the scheduler. Whatever framework you use next will be clearer for having read where it came from.
