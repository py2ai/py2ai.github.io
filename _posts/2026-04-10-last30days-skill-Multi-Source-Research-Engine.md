---
layout: post
title: "last30days-skill: AI-Powered Multi-Source Research Engine"
description: "Discover how last30days-skill researches topics across 12+ platforms and synthesizes grounded summaries scored by real engagement metrics."
date: 2026-04-10
header-img: "img/post-bg.jpg"
permalink: /last30days-skill-Multi-Source-Research-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Python
  - Open Source
  - Research
  - LLM
author: "PyShine"
---

# last30days-skill: AI-Powered Multi-Source Research Engine

In an era where information is scattered across dozens of platforms, finding what actually matters has become increasingly difficult. Google aggregates editorial content, but misses the real conversations happening on Reddit, X (Twitter), YouTube, and emerging platforms. ChatGPT has Reddit access but cannot search X or TikTok. Each platform is a walled garden with its own API, authentication, and data silos.

**last30days-skill** breaks down these walls. As the GitHub Trending #1 Repository of the Day with over 19,637 stars, this AI agent-led search engine researches topics across 12+ platforms simultaneously, scores results by real engagement metrics (upvotes, likes, views, and even real money from prediction markets), and synthesizes everything into one comprehensive brief.

## What Makes last30days-skill Different

Traditional search engines rank by SEO optimization and editorial authority. last30days-skill ranks by **social relevancy** - what real people actually engage with. A Reddit thread with 1,500 upvotes carries more weight than a blog post nobody read. A TikTok with 3.6 million views tells you more about cultural relevance than a press release. Polymarket odds backed by $66,000 in volume provide harder evidence than pundit predictions.

The engine searches Reddit, X, YouTube, TikTok, Instagram, Hacker News, Polymarket, GitHub, Threads, Pinterest, Bluesky, and Perplexity in parallel, then merges duplicate stories across platforms into unified clusters. The result: you get the complete picture, not fragmented pieces.

![Architecture Overview](/assets/img/diagrams/last30days-skill-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the complete data flow from user query to synthesized brief. Let's examine each component in detail:

**User Query Layer**
The journey begins when a user enters a topic - whether a person, company, product, technology, or comparison. The query can be as simple as "Kanye West" or as complex as "OpenClaw vs Hermes vs Paperclip." The system accepts natural language input without requiring specialized syntax or formatting.

**Intelligent Pre-Research Engine**
Before any search begins, the v3 engine performs intelligent entity resolution. This is the killer feature that separates last30days-skill from keyword-based search. When you type "OpenClaw," the engine resolves @steipete (Peter Steinberger, the creator), r/openclaw, r/ClaudeCode, relevant YouTube channels, and TikTok hashtags. This pre-research phase ensures searches target the right communities and voices from the start.

The entity resolution works bidirectionally: person to company, product to founder, name to GitHub profile. "Peter Steinberger" resolves to @steipete on X and steipete on GitHub. "Dave Morin" resolves to @davemorin plus @OpenClaw plus the TWiST podcast. This contextual understanding happens before a single API call fires.

**Multi-Source Parallel Search**
Once entities are resolved, the engine dispatches parallel queries to all configured sources. Reddit searches via public JSON APIs. X queries through browser cookie authentication. YouTube transcripts via yt-dlp. TikTok and Instagram through ScrapeCreators API. Each source returns results with engagement metrics attached.

**Engagement Scoring System**
Results flow through a sophisticated scoring pipeline that weighs multiple factors:
- Upvotes and likes (community validation)
- View counts and reach (audience size)
- Comment depth (conversation quality)
- Polymarket odds backed by real money (prediction market confidence)
- Recency within the 30-day window

The scoring algorithm prioritizes content that real people actually engaged with, not content optimized for search engines.

**Cross-Source Clustering**
When the same story appears across multiple platforms, v3 merges them into one cluster instead of showing duplicate items. Entity-based overlap detection catches matches even when titles use different words. A story about a product launch on Reddit, discussed on X, with pricing details on TikTok becomes one unified research item.

**AI Synthesis Engine**
The final judge synthesizes all scored and clustered results into one coherent brief. Grounded in specific data with citations by source, the synthesis ranks by what people actually engage with. The output is not "here's what I found" - it's "here's what matters."

**Session Memory Integration**
After one run, your Claude session knows everything the community knows. You can ask follow-up questions, have it write prompts, draft emails, plan trips, or architect systems - all grounded in current, real-world data.

## Data Sources: 12+ Platforms in Parallel

![Data Sources](/assets/img/diagrams/last30days-skill-sources.svg)

### Understanding the Data Sources

The data sources diagram organizes the 12+ platforms by category and authentication requirement. This comprehensive coverage ensures no critical perspective is missed:

**Social Media Platforms (X, Threads, Bluesky)**
These platforms capture real-time reactions, expert threads, and breaking news. X (Twitter) provides hot takes and expert commentary - often the first to know and first to argue. Threads offers the post-Twitter text layer with conversations from creators and brands. Bluesky brings the decentralized social layer via AT Protocol posts from the post-Twitter migration. Authentication for X requires only browser cookie login - no API key needed.

**Video Platforms (YouTube, TikTok, Instagram Reels)**
YouTube delivers the 45-minute deep dive format. Full transcripts are searched for the 5 quotable sentences that matter, extracting key insights from lengthy content. TikTok surfaces creator perspectives reaching millions with takes you will never find on Google. Instagram Reels adds influencer perspectives with spoken-word transcripts, capturing visual culture signals. These platforms require yt-dlp for YouTube (free) or ScrapeCreators API for TikTok and Instagram.

**Community Platforms (Reddit, Hacker News)**
Reddit provides the unfiltered take - top comments with upvote counts via free public JSON. The real opinions that Google buries surface here. Hacker News delivers the developer consensus with point counts and comment depth. Where technical people actually argue, you find signal over noise. Both platforms work immediately with zero configuration.

**Prediction Markets (Polymarket)**
Polymarket is unique: not opinions, but odds backed by real money. 96% confidence on album sales, 4% on an acquisition. These are not pundit guesses but market-priced probabilities. The signal quality is exceptional because people put money behind their predictions. Polymarket works immediately with no configuration required.

**Developer Platforms (GitHub)**
GitHub enables person-mode queries. When the topic is a person, the engine switches from keyword search to author-scoped queries. Instead of "who mentioned this name in an issue body," it answers: what are they shipping and where is it landing? PR velocity, top repos by stars, release notes, and README summaries provide the developer perspective.

**Discovery Platforms (Pinterest, Perplexity)**
Pinterest offers visual discovery with pins, saves, and comments on products and ideas. Perplexity provides grounded web search with citations via Sonar Pro. These platforms add editorial coverage and blog comparisons as one signal among many.

**Web Search (Brave Search)**
Traditional web search remains part of the mix, but as one signal of many, not the only one. Editorial coverage and blog comparisons supplement social signals.

**Authentication Requirements:**
- **Zero Config**: Reddit (with comments), Hacker News, Polymarket, GitHub work immediately
- **Browser Auth**: X/Twitter requires only logging into x.com in any browser
- **Free Tools**: YouTube requires `brew install yt-dlp`
- **API Keys**: Bluesky needs app password from bsky.app; Perplexity needs OpenRouter key; Brave Search offers 2,000 free queries/month
- **ScrapeCreators**: TikTok, Instagram, Threads, Pinterest, YouTube comments require ScrapeCreators key (10,000 free calls)

## The Six-Phase Pipeline

![Pipeline Flow](/assets/img/diagrams/last30days-skill-pipeline.svg)

### Understanding the Pipeline

The pipeline diagram shows the six-phase flow from raw query to synthesized brief. Each phase adds intelligence and filtering:

**Phase 1: Query Intake and Entity Resolution**
The pipeline begins with natural language query intake. The system accepts any topic format: person names, company names, product names, technologies, or comparisons like "X vs Y." The entity resolution engine then identifies relevant entities across platforms. For "Kanye West," it resolves r/hiphopheads, @kanyewest, and "bully review" on YouTube. For "OpenClaw," it resolves openclaw/openclaw on GitHub and fetches live star counts.

This phase represents the v3 intelligence breakthrough. The old engine searched keywords. The new engine understands your topic first, then searches the right people and communities. Bidirectional resolution means person-to-company and product-to-founder connections work both ways.

**Phase 2: Multi-Query Expansion**
Each resolved entity spawns multiple search queries optimized for each platform. The expansion engine understands platform-specific search syntax and optimal query construction. Reddit queries differ from X queries which differ from YouTube transcript searches. The expansion maximizes recall while maintaining precision.

**Phase 3: Parallel Source Fetching**
All expanded queries execute in parallel across configured sources. The fetching engine handles rate limits, timeouts, and authentication for each platform. Resilient Reddit implementation includes timeout budgets and runtime fallback - one slow thread does not kill the whole run. Results stream in as they arrive, not in sequential blocking order.

**Phase 4: Engagement Scoring and Relevance Ranking**
Raw results flow through the scoring pipeline. Each result receives multiple scores:
- Engagement score (upvotes, likes, views, comment depth)
- Relevance score (topic match quality)
- Freshness score (recency within 30-day window)
- Fun score (humor, wit, virality for "Best Takes" section)

The ranking algorithm weights these scores based on source reliability and user preferences. Per-author caps prevent any single voice from dominating - maximum 3 items per author ensures diversity.

**Phase 5: Cross-Source Clustering and Deduplication**
The clustering engine identifies when the same story appears across multiple platforms. Entity-based overlap detection catches matches even when titles use different words. A product announcement on Reddit, discussed on X, with pricing on TikTok becomes one unified cluster. This prevents the "three separate items" problem and provides comprehensive coverage.

**Phase 6: AI Synthesis and Brief Generation**
The final synthesis engine receives all scored, clustered results and generates one coherent brief. Grounded in specific data with citations by source, the synthesis tells you what matters, not just what was found. The output includes:
- Executive summary of key findings
- Detailed breakdown by topic cluster
- Best Takes section (cleverest one-liners, viral quotes)
- Source citations with engagement metrics
- Follow-up question suggestions

After synthesis, the session memory retains all context for follow-up questions. You can ask for deeper analysis, prompt generation, email drafts, or system architecture recommendations - all grounded in the research.

## Key Features and Innovations

![Key Features](/assets/img/diagrams/last30days-skill-features.svg)

### Understanding the Key Features

The features diagram highlights the four major innovations that distinguish last30days-skill from traditional search:

**Intelligent Pre-Research with Entity Resolution**
The v3 engine does not just search for your topic - it figures out where to search before the search begins. This contextual intelligence represents a fundamental shift from keyword matching to semantic understanding. The entity resolution system maintains mappings between:
- People and their X handles, GitHub profiles, and company affiliations
- Products and their subreddits, hashtags, and creator communities
- Companies and their founders, competitors, and industry discussions

This pre-research phase ensures every subsequent search targets the right communities. The result: v3 finds content v2 never could. "Paperclip" resolves to @dotta. "Dave Morin" resolves to @davemorin plus @OpenClaw plus the TWiST podcast. The right subreddits, the right handles, the right hashtags - resolved before a single API call fires.

**Engagement-First Ranking System**
Traditional search engines rank by SEO optimization and link authority. last30days-skill ranks by what real people actually engage with. The scoring system weighs:
- Reddit upvotes (community validation)
- X likes and retweets (real-time engagement)
- YouTube view counts and transcript relevance
- TikTok engagement metrics (cultural signal)
- Polymarket odds backed by real money (prediction market confidence)
- GitHub stars and PR velocity (developer adoption)

A Reddit thread with 1,500 upvotes ranks higher than a blog post nobody read. A TikTok with 3.6 million views tells you more about cultural relevance than a press release. This engagement-first approach surfaces what matters to real people, not what ranks well in search algorithms.

**Cross-Source Cluster Merging**
When the same story appears on Reddit, X, and YouTube, v3 merges them into one cluster instead of showing three separate items. Entity-based overlap detection catches matches even when titles use different words. This clustering provides several benefits:
- Comprehensive coverage without duplication
- Multiple perspectives on the same story
- Cross-platform verification of claims
- Unified context for synthesis

The clustering algorithm uses semantic similarity, entity extraction, and temporal proximity to identify related content across platforms. A story about a product launch appears once with perspectives from Reddit discussion, X reactions, and YouTube analysis all integrated.

**Zero Configuration Core Sources**
Reddit (with comments), Hacker News, Polymarket, and GitHub work immediately with no configuration. This zero-config approach removes friction for new users. Run `/last30days` once and the setup wizard unlocks additional sources in 30 seconds. The free tier provides substantial value before any API keys or authentication setup.

For power users, the "bring your own keys" model unlocks the full potential. X requires only browser login. YouTube needs yt-dlp. Bluesky needs an app password. ScrapeCreators enables TikTok, Instagram, Threads, and Pinterest. Each platform adds a unique perspective, and the synthesis engine integrates them all.

## Installation

### Claude Code
```
/plugin marketplace add mvanhorn/last30days-skill
```

### OpenClaw
```bash
clawhub install last30days-official
```

### Manual Installation
```bash
git clone https://github.com/mvanhorn/last30days-skill.git ~/.claude/skills/last30days
```

Reddit (with comments), Hacker News, Polymarket, and GitHub work immediately with zero configuration. Run `/last30days` once and the setup wizard unlocks more sources.

## Use Cases

### Before a Meeting
```
/last30days Peter Steinberger
```
Returns: joined OpenAI's Codex team, fighting Anthropic's ban on third-party agents, 23 PRs merged at 85% merge rate on GitHub, building LobsterOS for cross-device agent control. r/ClaudeCode discussions with 569 upvotes debating whether he's a hero or "insufferable." That's not on LinkedIn.

### When Something Drops
```
/last30days Kanye West
```
Returns: UK blocked his visa, Wireless Festival canceled, sponsors fled. But BULLY debuted #2 on Billboard. Fantano came back from his "Yay sabbatical" to review it (653K views). Polymarket: "Will Kanye tweet again?" 86% Yes. 23 Reddit threads, 17 YouTube videos, 86K upvotes.

### Tool Comparisons
```
/last30days OpenClaw vs Hermes vs Paperclip
```
Returns: "These aren't competitors, they're layers." OpenClaw is the executor (351K GitHub stars, live), Hermes is the self-improving brain (31K stars), Paperclip is the org chart (49K stars). Side-by-side table with architecture, memory, security, best-for recommendations.

### Understanding World Events
```
/last30days Iran vs USA
```
Returns: Day 38 of the war. Trump's Tuesday deadline for Iran to reopen the Strait of Hormuz. Two US warplanes downed. Oil at $126/barrel. IEA called it "the largest supply disruption in the history of the global oil market." Polymarket: ceasefire by Dec 31 at 74%. 27 X posts, 10 YouTube videos, 20 prediction markets.

### Trip Planning
```
/last30days Universal Epic Universe
```
Returns: Expansion already under construction. "Project 680" permit filed. Fireworks show confirmed by infrastructure but unannounced. Wait times: Mine-Cart Madness averaging 148 minutes. No annual pass yet, and locals are frustrated. Stardust Racers down for refurbishment through April 5.

### Learning Something Fast
```
/last30days Nano Banana Pro prompting
```
Returns: JSON-structured prompts are replacing tag soup. @pictsbyai's nested format prevents "concept bleeding." Edit-first workflow beats regeneration. Then it writes you a production prompt using exactly what the community said works.

## What People Are Saying

> "I found a Claude Code skill that researches any topic across Reddit, X, YouTube, and HN from the last 30 days. Then writes the prompts for you. I've been manually searching Reddit and X for research before every piece of content I write. Tab by tab. Thread by thread. That's the part that takes 90 minutes. This eliminates it." - @itsjasonai

> "This one skill replaced my entire research workflow. You give it a topic, it scrapes Reddit, X, and the web for what people are actually talking about. Not old blog posts. Real conversations from the last 30 days." - @itswilsoncharles

> "5 of the 10 trending repos on GitHub today are Claude tools. #1: mvanhorn/last30days-skill" - @yieldhunter95

## Technical Details

- **Language**: Python 3.12+
- **License**: MIT
- **Tests**: 1,012 passing
- **Dependencies**: yt-dlp, Node.js (vendored Bird client for X search), ScrapeCreators API
- **Architecture**: v3 engine architecture by @j-sperling

The system is built with resilience in mind. Timeout budgets and runtime fallback ensure one slow thread does not kill the whole run. Per-author caps prevent any single voice from dominating results. Entity disambiguation prevents false matches - no more Mallorca resorts winning over Washington athletic clubs.

## Conclusion

last30days-skill represents a paradigm shift in research. Instead of manually searching platform after platform, tab by tab, thread by thread, you get comprehensive coverage in a single query. The engagement-first ranking ensures you see what matters to real people, not what ranks well in search algorithms.

The v3 intelligent pre-research feature transforms keyword matching into semantic understanding. Cross-source clustering merges duplicate stories into unified perspectives. Zero-config core sources remove friction for new users while the bring-your-own-keys model unlocks full potential for power users.

Whether you are preparing for a meeting, researching a topic, comparing tools, or understanding world events, last30days-skill provides the complete picture. 12+ platforms searched in parallel, scored by engagement, synthesized into one brief. That is the unlock - not one better search engine, but a dozen disconnected platforms bridged by an AI agent.

**Repository**: [github.com/mvanhorn/last30days-skill](https://github.com/mvanhorn/last30days-skill)

**Stars**: 19,637+ (GitHub Trending #1 Repository of the Day)
