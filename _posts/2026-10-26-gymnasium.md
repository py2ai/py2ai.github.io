---
layout: post
title: "Gymnasium: A Source Tour of the Reinforcement Learning API That Replaced Gym"
description: "A guided tour through the Gymnasium source code: the reset and step contract, wrapper composition, vectorized environments, typed spaces, and how the Farama Foundation keeps a huge ecosystem coherent with a tiny core."
date: 2026-10-26
header-img: "img/post-bg.jpg"
permalink: /gymnasium/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/gymnasium/farama-foundation-gymnasium-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: gymnasium, reinforcement learning, env api, farama foundation, openai gym successor, vectorized environments, spaces, python, open source
author: "PyShine"
---

<p>Every reinforcement learning result you have ever read about, from an agent learning to balance a pole to a policy that lands a lunar module, rests on the same two method calls: <code>env.reset()</code> and <code>env.step(action)</code>. Gymnasium, maintained by the Farama Foundation, is the library that defines precisely what those calls mean, what they return, and how they must behave. It is the community successor to OpenAI Gym, adopted so thoroughly that essentially every modern reinforcement learning library, from training frameworks to evaluation harnesses, speaks its dialect. When Gym was archived in 2022, Gymnasium became the de facto standard: same ergonomics, stricter semantics, and a long-overdue cleanup of ambiguous return values.</p>

<p>What makes the source worth reading is its economy. The entire environment contract lives in one file, <code>gymnasium/core.py</code>, and it is barely seven hundred lines. Everything else in the library is composition on top of that contract: wrappers that layer behavior onto any environment, vector environments that step dozens of them in parallel, spaces that give observations and actions machine-checkable types, and a registration system that lets a string like <code>"CartPole-v1"</code> resolve into a running simulation. There is very little magic here, and that is exactly the lesson: a package used by thousands of research papers is held together by one small interface used with unusual discipline.</p>

<p>We have walked through the source of training-side systems before, like <a href="/pytorch/">PyTorch</a> internals, the compact <a href="/nanogpt/">nanoGPT</a> training script, and the RLHF framework in <a href="/openrlhf/">OpenRLHF</a>. But all of those consume experience data; Gymnasium is where that data is produced. Understanding the producer side completes the picture, and it also explains why certain quirks show up everywhere in RL code: why <code>step()</code> returns five values, why "truncated" and "terminated" are separate flags, and why seeding is an argument to <code>reset()</code> rather than a constructor call.</p>

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/gymnasium/farama-foundation-gymnasium-overview-architecture.svg" alt="Gymnasium architecture overview" style="max-width:100%;"></div>

<p><em>The whole library in one view: a single Env contract at the center, with factories, wrappers, vectorization, typed spaces, and reference environments arranged around it.</em></p>

## Reading the overview from left to right:

<ul>
<li><strong>The front door is one function.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/registration.py#L649"><code>make()</code></a> resolves a string id through a plain dictionary registry and returns a ready environment, while <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/registration.py#L844"><code>make_vec()</code></a> builds the batched variant. Third-party packages never modify the library; they call <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/registration.py#L572"><code>register()</code></a> to add their own ids. The registry is literally a module-level dict, which is refreshing to see in an era of over-engineered plugin systems.</li>
<li><strong><code>Env</code> is the entire contract.</strong> The base class at <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L30">core.py, line 30</a> defines <code>reset()</code>, <code>step()</code>, <code>render()</code>, and two typed attributes, <code>observation_space</code> and <code>action_space</code>. That is all an environment owes the world. The famous five-tuple return of <code>step()</code>, observation, reward, terminated, truncated, info, is documented right here, and the split between "the episode reached a natural end" and "we cut it off" is enforced in the type signature, not by convention.</li>
<li><strong>Wrapping is the extension mechanism.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L298"><code>Wrapper</code></a> is a transparent proxy: any attribute you do not override is forwarded to the inner environment, so a wrapper can override exactly one behavior and inherit everything else. Three convenience subclasses carve out the common cases, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L542"><code>ObservationWrapper</code></a>, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L586"><code>RewardWrapper</code></a>, and <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L621"><code>ActionWrapper</code></a>, each asking you to implement a single transformation function.</li>
<li><strong>The boring-but-critical guards live in one file.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/common.py#L44"><code>TimeLimit</code></a> converts infinite-horizon games into finite ones, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/common.py#L403"><code>OrderEnforcing</code></a> makes it an error to <code>step()</code> before <code>reset()</code>, and <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/common.py#L500"><code>RecordEpisodeStatistics</code></a> accumulates episode returns and lengths into the info dict. Notably, <code>make()</code> applies several of these automatically, which is why a raw class and a registered id are not quite the same object.</li>
<li><strong>Video capture is just another wrapper.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/rendering.py#L166"><code>RecordVideo</code></a> hooks into the render stream and writes mp4s of your episodes, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/rendering.py#L446"><code>HumanRendering</code></a> upgrades a headless environment into a windowed one, and <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/rendering.py#L34"><code>RenderCollection</code></a> simply accumulates frames in memory. Because these are wrappers rather than core features, an environment author can stay completely ignorant of how recording works.</li>
<li><strong>Vectorization is a class hierarchy, not a flag.</strong> The abstract <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py#L112"><code>VectorEnv</code></a> treats N environments as one, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py#L27"><code>SyncVectorEnv</code></a> steps them sequentially in a Python loop, and <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/async_vector_env.py#L56"><code>AsyncVectorEnv</code></a> runs each in its own process so stepping happens genuinely in parallel. This is the workhorse behind the "collect a rollout of 4096 transitions" idiom in modern algorithms.</li>
<li><strong>Spaces make shapes machine-checkable.</strong> The base class at <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/space.py#L18">space.py, line 18</a> promises <code>sample()</code>, <code>contains()</code>, and seeding. Concrete subclasses cover the universe: <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/box.py#L64"><code>Box</code></a> for continuous vectors, <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/discrete.py#L27"><code>Discrete</code></a> for integer actions, plus composites like <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/dict.py#L25"><code>Dict</code></a> for structured observations. Network input layers can be derived from spaces instead of hand-configured, which removes a whole class of shape bugs.</li>
<li><strong>Reference environments show the contract in action.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py#L21"><code>CartPoleEnv</code></a> is two hundred lines of physics plus rendering, an ideal template for writing your own environment, and <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py#L79"><code>LunarLander</code></a> shows the same contract applied on top of a real physics engine with fuel constraints and shaping rewards.</li>
<li><strong>A checker keeps the ecosystem honest.</strong> <a href="https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/utils/env_checker.py#L406"><code>check_env()</code></a> runs a battery of assertions against any environment you hand it: spaces must match what <code>reset()</code> and <code>step()</code> actually emit, determinism with a fixed seed must hold, and dtypes must not drift. Most "my env trains on Monday and crashes on Tuesday" bugs are caught here before training ever starts.</li>
</ul>

## The Detail View: One Step Deeper

<p>The overview compresses the story; the detail graph expands it into the classes you will actually touch when writing or debugging an environment. The left-to-right narrative now becomes a nesting narrative, because almost every arrow in the second diagram is a "wraps" or "contains" relationship rather than a call.</p>

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/gymnasium/farama-foundation-gymnasium-architecture.svg" alt="Gymnasium detailed architecture" style="max-width:100%;"></div>

<p><em>The detail graph: the wrapper family tree on the core, the two vectorization strategies, the space hierarchy with its composites, and the two-tier checker that validates environments both actively and passively.</em></p>

<p>Three relationships deserve your attention. First, the wrapper family tree: <code>TimeLimit</code>, <code>RecordEpisodeStatistics</code>, and friends all descend from the proxy in <code>core.py</code>, which is why they compose in any order and why the env wrapper chain grows at runtime without any central coordination. Second, the two vectorization strategies share one abstract parent, so an algorithm written against <code>VectorEnv</code> runs on either backend unchanged; the synchronous version is your debugging baseline and the asynchronous version is your speed lever, and swapping them is a one-line change. Third, the checker is two-tiered by design: <code>check_env()</code> performs loud, exhaustive validation when you ask for it, while the passive checker runs quietly inside every <code>make()</code> call, issuing warnings instead of exceptions so that a sloppy third-party environment degrades gracefully rather than breaking the library.</p>

## From Install to Your First Model

<p>Install is a single package name, and the canonical first script fits on a Post-it note:</p>

<pre><code>import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
</code></pre>

<p>Read it as a conversation with the contract. <code>reset()</code> returns the first observation plus an info dict; the seed argument makes the episode reproducible. <code>step()</code> returns the five-tuple, and the two booleans must be handled separately: <code>terminated</code> means the environment ended for a reason intrinsic to the task, while <code>truncated</code> means an external time limit fired. Conflating the two is a classic bootstrap-error bug, because bootstrapping from a state after a time-limit cut introduces bias that the value function then learns to propagate.</p>

<p>Wrappers attach like layers of clothing:</p>

<pre><code>env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
obs, info = env.reset()
</code></pre>

<p>Each wrapper returns an object that still looks and smells like an <code>Env</code>, so code written for the naked environment keeps working through any number of layers. When you are ready for speed, <code>gym.make_vec("CartPole-v1", num_envs=8)</code> hands you a batched environment whose <code>step()</code> takes an array of eight actions and returns arrays of eight observations, rewards, and flags. For a taste of where this goes, an environment loop is the part of an RLHF training loop that <a href="/openrlhf/">OpenRLHF</a> and its siblings delegate to exactly this API, while the model updates run in <a href="/pytorch/">PyTorch</a>; the two worlds meet at the five-tuple.</p>

<p>One more habit worth stealing from the source: define your environment's spaces with the same care the built-ins do. If your observation is a camera image plus a joystick state, declare a <code>Dict</code> space with a <code>Box</code> and a <code>Discrete</code> inside it rather than flattening everything into one vector and remembering the layout by comment. The same philosophy shows up in supervised tooling, where <a href="/scikit-learn/">scikit-learn</a> users get the same class of bug prevention from typed feature containers, and it costs nothing at import time.</p>

## Honest Limits

<p>Gymnasium is superb infrastructure, and it is worth knowing where the infrastructure stops. The classic control and Box2D environments are pedagogical toys; their dynamics are smooth, low-dimensional, and cheap, which makes them wonderful for validating an algorithm and useless for predicting how it will fare on a real robot or a web-scale recommendation task. The moment you need realistic dynamics you leave the built-in registry and depend on third-party packages, with all the version-drift risk that implies.</p>

<p>The asynchronous vector environment has sharp edges too. It pays for parallelism with process boundaries: every environment object must survive pickling, every observation crosses a pipe, and a crash in one worker surfaces as an opaque error on the next collective call. For small or cheap environments the synchronization overhead can exceed the benefit, and the honest benchmarking advice is to compare <code>SyncVectorEnv</code> and <code>AsyncVectorEnv</code> on your workload before assuming the async one is faster.</p>

<p>Finally, there is the migration tax. The rename from <code>gym</code> to <code>gymnasium</code> happened precisely because the old API's ambiguities, most notoriously the four-tuple step return and hidden seeding state, could not be fixed in place. Ten years of tutorials, blog posts, and GitHub issues still use the old names, and copy-pasting them yields confusing import errors rather than clean failures. The library also deliberately knows nothing about agents, replay buffers, or experiment tracking; if you want those, you compose a training framework on top, or accept that you are reading the wrong source tree. What you get in exchange is a dependency that fits in your head: one contract, wrappers you can reason about locally, and a checker that keeps the ecosystem's promises machine-verified rather than folklore-verified.</p>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "datePublished": "2026-10-26",
  "headline": "Gymnasium: A Source Tour of the Reinforcement Learning API That Replaced Gym",
  "description": "A guided source tour of Gymnasium: the Env contract, wrapper composition, vectorized environments, typed spaces, and the env checker.",
  "keywords": "gymnasium, reinforcement learning, env api, farama foundation, open source",
  "author": { "@type": "Person", "name": "PyShine" }
}
</script>
