---
layout: post
title: "X Algorithm: The Open-Source For You Feed Ranking System from xAI"
description: "The xai-org/x-algorithm repository contains the core code that determines which posts a viewer sees in the For You feed on X. The system combines in-network content from Thunder (recent posts from accounts the viewer follows) with out-of-network content discovered through Phoenix retrieval and SimClusters, filters posts through a chain of pre-scoring and post-selection filters, and ranks them with a transformer model called Phoenix that predicts a probability for each action the viewer might take. Ranking sets the order; a separate visibility filtering system decides whether a post can be shown at all. The architecture splits into two paths: a Request Path that runs per feed request through home-mixer's Post Pipeline and Blending Pipeline, and a Labeling Path that runs continuously to produce the content-understanding scores and labels that drive visibility filtering. Written in Rust, licensed under Apache 2.0, the repo includes 24 component services plus a transparency tool called Under the Hood. This post walks through the request path, the labeling path, the scoring and ranking model, and the component map."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /X-Algorithm-For-You-Feed-xAI/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - X Algorithm
  - For You Feed
  - Recommendation System
  - Ranking
  - Transformer
  - Rust
  - Open Source
  - xAI
  - Visibility Filtering
author: PyShine
---

## What is the X Algorithm

The [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm) repository contains the core code that determines which posts a viewer sees in the **For You** feed on X. It combines in-network content (from accounts the viewer follows) with out-of-network content (discovered through ML-based retrieval and other mechanisms), filters content based on a variety of inputs, and ranks posts using a transformer model called Phoenix. The code is written in Rust, licensed under the Apache License 2.0, and is organized into 24 component services that together implement the two paths the feed runs on: the Request Path and the Labeling Path.

## Overview

The For You feed is assembled per request. Posts come from two places:

1. **In-Network** - [thunder/](https://github.com/xai-org/x-algorithm/tree/main/thunder) keeps recent posts from the accounts a viewer follows in memory.
2. **Out-of-Network** - [phoenix/](https://github.com/xai-org/x-algorithm/tree/main/phoenix) retrieval and [simclusters/](https://github.com/xai-org/x-algorithm/tree/main/simclusters) find posts from accounts the viewer does not follow.

Both are ranked together by the same model. Phoenix reads the viewer's recent engagement history and predicts, for each post, how likely the viewer is to take each action on it. Those predictions are combined into one score using weights held in the code.

Two pipelines do the work. The **Post Pipeline** finds, ranks and filters posts. The **Blending Pipeline** wraps it and adds what the model does not rank: ads, Who to Follow recommendations, prompts.

Ranking sets the order. Whether a post can be shown at all is decided separately, by [visibility-filtering/](https://github.com/xai-org/x-algorithm/tree/main/visibility-filtering), from the viewer's own actions such as blocks and mutes and from labels that other systems attach to posts and accounts.

## Request Path

The Request Path runs per feed request through `home-mixer`, the service that builds the For You feed. It has two pipelines: the Post Pipeline, which finds, ranks, and filters posts, and the Blending Pipeline, which wraps the Post Pipeline and adds what the model does not rank.

![X For You feed request path](/assets/img/diagrams/x-algorithm/x-algorithm-request-path.svg)

The Post Pipeline has seven stages:

1. **Query Hydration** assembles the viewer's context: user action sequence (recent engagements, the main input to the model), following list, blocks and mutes, muted keywords, posts already seen and served, followed topics.
2. **Candidate Sources** are queried in parallel. Thunder returns recent posts from followed accounts; Phoenix retrieval embeds the viewer and each post as vectors and returns the nearest posts; SimClusters clusters accounts and posts by engagement and uses the clusters to find candidates.
3. **Candidate Hydration** loads post text and media, author details and account labels, quoted post, language, engagement counts, subscription status.
4. **Pre-Scoring Filters** remove duplicates across sources, posts older than 48 hours, the viewer's own posts, blocked and muted accounts, muted keywords, already-seen or served posts, subscriber-only posts the viewer cannot access, and more.
5. **Scoring** runs three scorers. PhoenixScorer predicts a probability for each action the viewer might take. RankingScorer computes a weighted sum, then applies repeated-author decay, an out-of-network discount, and a new-author boost. VMRanker calls a separate service that reorders the result with a determinantal point process over embeddings, trading a little score for less similarity between neighbors.
6. **Selection** sorts by final score and keeps the top K (TopKScoreSelector).
7. **Post-Selection Filters** run after the order is fixed. VFCandidateHydrator asks visibility-filtering per post and viewer; VFFilter removes the posts it said to drop; DedupConversationFilter collapses branches of one conversation.

The Blending Pipeline then interleaves the ranked posts with ads, Who to Follow recommendations, and prompts. The BlenderSelector handles interleaving; the default ads blender reorders posts for ad adjacency, and Who to Follow and prompts go at fixed positions. Side effects run after the response is sent: recording which posts were served, refreshing the post cache, logging ad and client events.

## Labeling Path

The Labeling Path runs continuously, off the request path, to produce the scores and labels that drive visibility filtering. It has five stages.

![X For You feed labeling path](/assets/img/diagrams/x-algorithm/x-algorithm-labeling-path.svg)

1. **Content Understanding** runs as posts are published and as accounts act. For posts and media, grox runs text and media classifiers, media-model-proxy serves image and video models (adult content, violence, hateful symbols, known-media matching), and clip trains the image-text embedding model. For accounts, agatha runs offline batch jobs that label an account from how others respond to its posts (blocks, reports, spam reports relative to favorites), bdsm reads the sequence of actions an account takes over time to identify inauthentic behavior, and user-cred-v2 runs PageRank over the follow graph and engagement edges to produce a per-account score.
2. **Labeling Rules** apply labels. Scarecrow reacts to events as they happen, embedding botmaker as its rule engine and loading rules from botmaker-rules. Abuse-enforcement-service reads model scores about an account and labels it or its posts, challenges it, or suspends it. Safety-label-user-agg labels an account for what its posts collected.
3. **Storage** writes labels to storage, read back on the request path.
4. **Visibility Filtering** determines, for each post and viewer, one of three answers: ALLOW (show normally), INTERSTITIAL (show behind a tappable interstitial, e.g. for adult or graphic media), or DROP (do not show). The rules read the labels above, plus whether the viewer blocks, mutes, or follows the author, whether the account is protected, suspended, or deactivated, subscriber-only status, and the viewer's settings and country. Some rules drop a post only when it is a recommendation from an account the viewer does not follow; the same post is allowed to a follower.
5. **Post-Selection Filters** apply the visibility filtering answers on the request path. VFFilter drops posts visibility-filtering said to drop. AncillaryVFFilter drops posts whose ancestor in the thread, quoted post, or reposted post was itself dropped. Posts marked interstitial stay in the feed; nothing in this repository draws the interstitial.

## Scoring and Ranking

Phoenix predicts a probability for each of many actions, not a single relevance score. Combining them into one number is a separate, explicit step.

![X For You feed scoring and ranking](/assets/img/diagrams/x-algorithm/x-algorithm-scoring.svg)

The actions predicted include:

- **Engagement:** favorite, reply, repost, quote, share, share via DM, share via copy link
- **Clicks:** post, profile, link, photo expand, video open, quoted post
- **Attention:** video quality view, dwell, dwell time, click dwell time, active seconds
- **Author:** follow author
- **Negative:** not interested, mute author, block author, report, not dwelled

`RankingScorer` combines them:

```
Final Score = Sum(weight_i * P(action_i))
```

Positive actions carry positive weights, negative actions negative ones. The weights are in [home-mixer/params/param.rs](https://github.com/xai-org/x-algorithm/tree/main/home-mixer/params/param.rs); the arithmetic is in [home-mixer/scorers/ranking_scorer.rs](https://github.com/xai-org/x-algorithm/tree/main/home-mixer/scorers/ranking_scorer.rs).

There is a common misconception about the weights: they scale the predicted probabilities (or predicted continuous values, e.g. dwell time), not the raw engagement counts. It would be incorrect to see that a report has a much higher weight than a like and conclude that one report cancels out hundreds of likes. The weights are a multiple on the viewer's own predicted probability of liking, reporting, etc, which is substantially driven by the viewer's own behavior.

Three adjustments follow:

- **Author Diversity:** each post after an author's first is multiplied by a decaying factor, down to a floor.
- **Out-of-Network Discount:** posts from accounts the viewer does not follow are multiplied by a factor below 1, as are replies and reposts from accounts the viewer does follow.
- **New-Author Boost:** posts from authors whose impressions are below a threshold are lifted toward a target position.

`VMRanker` then calls vm-ranker, a separate service that reorders the result with a determinantal point process over their embeddings, giving up a little score for less similarity between neighbors.

## Component Map

The repository is organized into 24 component services. The diagram below shows how they connect.

![X For You feed component map](/assets/img/diagrams/x-algorithm/x-algorithm-components.svg)

The component groups are:

- **Home Mixer and Candidate Pipeline:** home-mixer (builds the For You feed) and candidate-pipeline (the framework home-mixer is built on).
- **Candidate Sources:** thunder (in-network cache), phoenix (retrieval and ranking), simclusters (cluster similarity).
- **Retrieval Index:** phoenix-rankall (post index), phoenix-rankall-strato (event layer).
- **Ranking:** phoenix ranking (JAX training, Rust serving), vm-ranker (DPP reranking).
- **Content Understanding:** grox, media-model-proxy, clip, agatha, bdsm, user-cred-v2, adult-content, pnsfwmedia.
- **Visibility Filtering:** visibility-filtering, scarecrow, botmaker, botmaker-rules, abuse-enforcement-service, safety-label-user-agg, visibility-filtering-client, under-the-hood.

## Key Design Decisions

1. **Multi-Action Prediction:** rather than predicting a single relevance score, the model predicts probabilities for many actions. Combining them into one number is a separate, explicit step.
2. **Candidate Isolation in Ranking:** during transformer inference, candidates cannot attend to each other - only to the viewer context. This ensures the score for a post does not depend on which other posts are in the batch, making scores consistent and cacheable.
3. **Hash-Based Embeddings:** both retrieval and ranking use multiple hash functions for embedding lookup, so there is no vocabulary to maintain and a new post is representable immediately.
4. **Ranking and Visibility Are Separate:** ranking decides the order. Visibility filtering decides whether a post can be shown at all. Different services, different inputs, different rules.
5. **Composable Pipeline Architecture:** the candidate-pipeline crate provides a framework for building recommendation pipelines with separation of execution and monitoring from business logic, parallel execution of independent stages, and graceful error handling.

## Experiments and Configuration

As xAI works to improve the algorithm, experiments run regularly on a small percentage of timeline traffic. The aim is for experiments running at a notable share of traffic (e.g. 10% or more) to be visible in the repository. Many tunable values are read from a configuration system rather than written into the code. Cron scripts set the defaults in the repository's code to match production values, so the public can understand the production defaults.

A notable example is the Brazil 2026 Elections filter: in accordance with Brazilian electoral law, For You runs Brazil2026ElectionFilter, which removes posts from accounts reported to Brazil's Electoral Court unless the viewer explicitly follows the account. The code is in [home-mixer/filters/brazil_2026_election_filter.rs](https://github.com/xai-org/x-algorithm/tree/main/home-mixer/filters).

## Under the Hood Transparency Tool

xAI is piloting a transparency tool called Under the Hood that lets people see aggregate statistics about the visibility-impacting labels on their account and posts. The tool is available at [x.com/i/under_the_hood](https://x.com/i/under_the_hood). The jobs and serving code that build the report are in [under-the-hood/](https://github.com/xai-org/x-algorithm/tree/main/under-the-hood).

## What's Not in the Repo

To reduce the risk of gaming the system, a limited set of files is not currently published, including Grox prompts (the j2 files with the specific LLM prompts) and some botmaker rules. However, the public still has insight into these systems through the Under the Hood transparency tool, which shows the outcomes of these systems and whether they affect a viewer's own account.

## Conclusion

The X Algorithm repository is one of the most transparent recommendation systems ever open-sourced. By publishing the code that determines which posts a viewer sees in the For You feed, X and xAI let the public audit, critique, and even help improve the system. The two-path architecture (Request Path and Labeling Path), the multi-action Phoenix model, the separation of ranking from visibility filtering, and the Under the Hood transparency tool together give a complete picture of how a major social media feed is built. The source is on GitHub at [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm), licensed under Apache 2.0.
