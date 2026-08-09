---
layout: post
title: "Embabel Agent Framework: Goal-Oriented AI Agents on the JVM"
description: "A deep dive into the Embabel Agent Framework, a JVM agentic AI framework created by Spring founder Rod Johnson. Learn how its non-LLM planning engine, OODA loop, and strongly typed domain model differentiate it from finite state machine approaches. Complete guide with architecture diagram, key concepts, installation, and quick start."
date: 2026-08-08
permalink: /Embabel-Agent-Framework-JVM-AI-Agent/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/embabel-agent/embabel-agent-architecture.svg
tags: [Embabel, AI agents, JVM, Kotlin, Java, Spring Boot, agentic AI, GOAP, planning, LLM, domain model, framework]
categories: [AI Agents, Java, Developer Tools]
keywords: "Embabel agent framework, JVM agentic AI, goal oriented action planning, OODA loop AI agents, Spring AI agents, Kotlin agent framework, Java LLM framework, Rod Johnson Embabel, non-LLM planning algorithm, typed domain model agents, embabel-agent-starter maven, how to build AI agents on JVM"
author: "PyShine"
---

# Embabel Agent Framework: Goal-Oriented AI Agents on the JVM

## Introduction

AI agent frameworks have proliferated over the last two years, but the overwhelming majority of them live in the Python ecosystem. Developers working on the JVM have largely been left to either call out to Python services or hand-roll orchestration logic on top of raw LLM client libraries. The Embabel Agent Framework (pronounced Em-BAY-bel) takes a different position: it brings first-class, goal-oriented agentic AI to the JVM, with a planning model that goes well beyond the finite state machines and linear chains that dominate the current landscape.

Created by Rod Johnson, the original creator of the Spring Framework, Embabel treats agentic flows as a collaboration between a classical AI planner and LLM-driven transforms. The planner decides which action runs next, the LLM handles the fuzzy transformations and judgments it excels at, and a strongly typed domain model keeps everything refactoring-friendly and testable. The result is a framework that feels native to Java and Kotlin developers while delivering the dynamic, adaptive behavior people expect from modern agents.

This post walks through what Embabel is, how its architecture works, the core concepts of actions, goals, and conditions, the planning engine that sets it apart, and how to install and run your first agent. The project is open source under the Apache 2.0 license and lives at [github.com/embabel/embabel-agent](https://github.com/embabel/embabel-agent), with documentation at [docs.embabel.com](https://docs.embabel.com) and a companion hub at [hub.embabel.com](https://hub.embabel.com).

## What is Embabel

Embabel is a framework for authoring agentic flows on the JVM that seamlessly mix LLM-prompted interactions with code and domain models. It is written in Kotlin but offers a natural usage model from Java, and it builds on Spring Boot, Spring AI, Apache Maven, JUnit 5, Jinja templates, and Docker. An Embabel application is, at its core, a Spring Boot application in which agents are discovered and registered as Spring beans.

The framework models agentic flows in terms of actions, goals, conditions, a domain model, and dynamically formulated plans. Rather than forcing the developer to hard-wire a state machine, Embabel infers a plan from the available actions and the types they consume and produce, then replans after each action executes. This is effectively an OODA loop (Observe, Orient, Decide, Act) running on every step.

Several differentiators distinguish Embabel from other agent frameworks. Its planning uses a non-LLM AI algorithm rather than asking a model to decide the next step, which makes planning deterministic, debuggable, and cheap. Its superior extensibility means new actions and goals can be added without editing existing finite state machine definitions. Its strong typing and object orientation mean prompts and manually authored code interact cleanly through a domain model, with no magic maps and full refactoring support. And because it is built on Spring and the JVM, it integrates naturally with existing enterprise infrastructure such as persistence, transactions, and observability.

## How It Works

![Embabel Agent Architecture](/assets/img/diagrams/embabel-agent/embabel-agent-architecture.svg)

### Understanding the Embabel Architecture

The architecture diagram above presents a layered view of the Embabel Agent Framework, showing how user intent flows through a planning-driven runtime and ultimately surfaces as typed domain objects.

The top tier, rendered in green, represents the inputs that an application developer supplies. The User Goal describes what the agent should accomplish, while the Domain Model is a set of strongly typed objects that underpin the entire flow.

These inputs are not passive configuration; they actively shape every subsequent decision the framework makes. The domain model informs both the planning step, by defining the types actions can consume and produce, and the conditions that gate action execution.

The middle tier, rendered in blue, is the Embabel framework itself and the heart of the system. The Goal Resolver takes the user's intent and identifies which registered goal best matches it.

The Planning Engine then uses a non-LLM AI algorithm to synthesize a sequence of actions that can reach that goal from the current world state. This is classical AI planning, not a prompt asking an LLM to decide what to do next.

The OODA Loop component orchestrates the observe-orient-decide-act cycle that drives execution. This ensures the agent never blindly follows a static script and instead adapts after every step.

Layer 2, also in blue, contains the action-level primitives that execute on every iteration. Check Conditions evaluates the current state before any action runs, and these conditions are reassessed after each action completes.

Execute Action runs the chosen action, which may invoke an LLM, call a tool, or execute plain JVM code. The ability to mix LLM calls with regular code in the same flow is one of Embabel's defining characteristics.

Replan hands control back to the planning engine so a fresh plan can be formulated in light of what the previous action produced. This is the mechanism that keeps the agent responsive to new information.

The bottom-left tier, rendered in orange, covers LLM integration. Prompt Rendering turns typed inputs and Jinja templates into well-structured prompts, so prompts remain typesafe and refactoring-friendly.

Tool Calls expose Spring AI `@Tool` methods, including Model Context Protocol tools, to the model. This lets the agent take real actions rather than only emit text.

The bottom-right tier, rendered in purple, represents the Domain Objects that flow through the system. The Typed Model gives full refactoring support and JSON schema generation for the LLM.

Behavior lets those objects expose safe methods to the LLM via `@Tool` annotations, while Validation ensures the data stays consistent as it moves between actions and back into the world state.

The labelled edges trace the OODA loop explicitly. The cycle begins with Observe, where the OODA Loop directs Check Conditions to assess the world state.

Orient follows, as the observed conditions flow back to the Goal Resolver to re-establish context and refine the understanding of the situation.

Decide is the Planning Engine formulating the next action, choosing from all available actions based on their preconditions and the effects they produce.

Act is Execute Action carrying out that decision. After each action, the Replan edge returns control to the loop, and the cycle repeats indefinitely until the goal is achieved.

This continuous replanning is what lets Embabel adapt to new information and combine known steps in novel orders, achieving tasks that were never explicitly programmed as a fixed sequence.

A critical point is that the planning step is not delegated to an LLM. By using a classical AI planning algorithm, Embabel keeps planning deterministic, debuggable, and decoupled from token cost.

The LLM is reserved for the transformations and judgments it is genuinely good at, while structural reasoning about which action should run next is handled by code.

This separation is what enables the framework to find novel paths to a goal, to parallelize work where appropriate, and to remain testable end to end with standard JUnit tooling.

## Key Concepts

Embabel models agentic flows using a small set of composable concepts. Application developers rarely have to deal with them directly, because most conditions result from data flow defined in code and the system can infer pre and post conditions. Understanding them, however, makes the framework's behavior much easier to reason about.

### Actions

Actions are the steps an agent takes. An action is simply a method, annotated with `@Action`, that accepts typed inputs and returns a typed output. Inputs are satisfied from the current world state, and the returned object is added back to the world state for subsequent actions to consume. An action can call an LLM, invoke a tool, hit a database, or execute pure code. The following Kotlin snippet shows a typical action that uses the default LLM to create a typed object from user input:

```kotlin
@Action
fun extractPerson(
    userInput: UserInput,
    ai: Ai,
): StarPerson =
    ai.withDefaultLlm()
        .createObject("Create a person from this user input, extracting their name and star sign: $userInput")
```

Because the return type is `StarPerson`, the planner knows that any other action requiring a `StarPerson` can now run. This type-driven flow is what lets the framework infer execution plans without explicit wiring.

### Goals

Goals describe what an agent is trying to achieve. An action annotated with `@AchievesGoal` signals that completing that action achieves a particular goal, so the agent run can complete. In open mode, the platform assesses the user's intent and searches across all known goals, building a custom agent from the start state to reach the chosen goal. Goals are independent of any particular planning algorithm, and an agent can declare multiple goals with different return types.

### Conditions

Conditions are predicates assessed before executing an action or determining that a goal has been achieved. Crucially, conditions are reassessed after each action executes, which is what makes the flow adaptive rather than static. Most conditions never need to be written by hand because they fall out of the types an action requires and returns. When explicit conditions are needed, a method annotated with `@Condition` can gate an action based on the current world state. This is also how the planner can choose between mutually exclusive actions that produce the same type.

## Planning Engine

The planning engine is where Embabel diverges most sharply from other agent frameworks. Instead of a finite state machine or a sequential pipeline with nesting, Embabel introduces a true planning step that uses a non-LLM AI algorithm to decide what to do next.

The default planning approach is Goal Oriented Action Planning (GOAP), a popular AI planning algorithm originally used in game AI. GOAP treats each action as having preconditions and effects, and it searches for a sequence of actions that transforms the current world state into one in which the goal is satisfied. Because the search is performed over a graph of typed actions rather than a hardcoded transition table, the planner can combine known steps in novel orders and even accomplish tasks that were never explicitly programmed as a single flow.

The planning step is pluggable. GOAP is the default, but Embabel also supports Utility AI out of the box, which runs the same actions but chooses them based on potentially dynamic utility scores rather than strict preconditions and postconditions. Utility AI is valuable for exploration and open-ended tasks where there is no single goal to achieve but the agent should maximize overall utility.

After each action executes, the planner replans. The world state has changed, conditions are reassessed, and a fresh plan is formulated from the new state. This OODA loop (Observe the world state, Orient toward the goal, Decide on the next action, Act on it) runs continuously until a goal is achieved or no further progress is possible. The benefit is adaptivity: if an action returns unexpected data, the planner routes around it rather than failing a brittle script.

A consequence of this design is superior extensibility and reuse. Adding a new action, goal, or domain object extends the capability of the system without editing any existing finite state machine definition. The planner automatically incorporates the new action into future plans. This is analogous to how adding a new Spring `@Controller` method extends a web application without modifying existing controllers.

## Installation

Since version 0.2.0, Embabel Agent Framework is available directly on Maven Central, so no custom repository configuration is needed for stable releases. Add the Embabel Spring Boot starter to your `pom.xml`:

```xml
<dependency>
    <groupId>com.embabel.agent</groupId>
    <artifactId>embabel-agent-starter</artifactId>
    <version>${embabel-agent.version}</version>
</dependency>
```

The Spring Milestones repository is also required because the Embabel BOM has transitive dependencies on experimental Spring components such as `mcp-bom`. Add it alongside Maven Central:

```xml
<repositories>
    <repository>
        <id>spring-milestones</id>
        <url>https://repo.spring.io/milestone</url>
        <snapshots>
            <enabled>false</enabled>
        </snapshots>
    </repository>
</repositories>
```

For Gradle Kotlin DSL, the equivalent configuration is:

```kotlin
repositories {
    mavenCentral()
    maven {
        name = "Spring Milestones"
        url = uri("https://repo.spring.io/milestone")
    }
}

dependencies {
    implementation("com.embabel.agent:embabel-agent-starter:${embabelAgentVersion}")
}
```

You will also need an API key for at least one LLM provider. Embabel prefers conventional environment variable names such as `OPENAI_API_KEY` over the Spring AI variants. Set the key in your environment or in a `.env` file:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

Enable Embabel by annotating your Spring Boot application class. This is a normal Spring Boot application, so you can add any other Spring Boot annotations you need:

```java
@SpringBootApplication
public class MyAgentApplication {
    public static void main(String[] args) {
        SpringApplication.run(MyAgentApplication.class, args);
    }
}
```

## Quick Start

The fastest way to get an agent running is to create a project from one of the official GitHub templates. The [Java template](https://github.com/embabel/java-agent-template) and [Kotlin template](https://github.com/embabel/kotlin-agent-template) give you a working Spring Boot application with the Embabel starter already configured. Alternatively, generate a custom project with the project creator:

```bash
uvx --from git+https://github.com/embabel/project-creator.git project-creator
```

A minimal agent is a Spring bean annotated with `@Agent` that exposes one or more `@Action` methods. The action below extracts a typed `StarPerson` from user input and then retrieves a horoscope, mixing an LLM call with a plain service invocation in the same flow:

```kotlin
@Agent(description = "Find news based on a person's star sign")
class StarNewsFinder(
    private val horoscopeService: HoroscopeService,
) {

    @Action
    fun extractPerson(userInput: UserInput, ai: Ai): StarPerson =
        ai.withDefaultLlm()
            .createObject("Create a person from this user input, extracting their name and star sign: $userInput")

    @Action
    fun retrieveHoroscope(starPerson: StarPerson): Horoscope =
        Horoscope(horoscopeService.dailyHoroscope(starPerson.sign))

    @AchievesGoal(description = "Write an amusing writeup for the target person")
    @Action
    fun writeup(person: StarPerson, horoscope: Horoscope, ai: Ai): Writeup =
        ai.withDefaultLlm().createObject(
            """
            ${person.name} is an astrology believer with the sign ${person.sign}.
            Their horoscope for today is: ${horoscope.summary}.
            Write up something amusing and format it as Markdown.
            """.trimIndent()
        )
}
```

The domain classes that flow through the agent are ordinary Kotlin data classes or Java records, annotated with Jackson descriptions so the LLM understands the schema:

```kotlin
@JsonClassDescription("Person with astrology details")
data class StarPerson(
    override val name: String,
    @get:JsonPropertyDescription("Star sign")
    val sign: String,
) : Person
```

If you added the shell starter, run the application and use the `execute` (or `x`) command to invoke an agent. The `-p` flag logs prompts and the `-r` flag logs LLM responses:

```bash
execute "Lynda is a Scorpio, find news for her" -p -r
```

Agents are unit testable just like any other Spring-managed bean. Construct them with mocks, call individual action methods, and assert on the prompts and hyperparameters that were sent to the LLM. This testability is a first-class design goal of the framework, inherited from the same philosophy that made Spring applications easy to test.

## Conclusion

The Embabel Agent Framework brings a genuinely different model to agentic AI on the JVM. By separating a deterministic, non-LLM planning engine from the LLM transformations themselves, it avoids the brittleness and cost of asking a model to orchestrate its own workflow. The OODA loop of observe, orient, decide, and act runs on every action, so agents adapt to new data rather than marching through a fixed script. Strong typing, a real domain model, and Spring integration mean the resulting code is refactoring-friendly, observable, and testable with the tools JVM developers already know.

The differentiators add up. Sophisticated planning via GOAP enables novel paths to goals that were never explicitly programmed. Superior extensibility means new actions and goals extend the system without editing existing finite state machine definitions. Goal orientation means agents find their own way to a result rather than following hand-wired transitions. For teams already invested in Java, Kotlin, and Spring, Embabel offers a path to production-grade agentic AI without leaving the platform they trust.

The project is young but ambitious, with a roadmap that includes budget-aware agents, agent federation, and ports to other platforms. It is open source under the Apache 2.0 license, created by Rod Johnson and a team of experienced engineers. To go deeper, explore the [source code on GitHub](https://github.com/embabel/embabel-agent), read the [reference documentation](https://docs.embabel.com), or try the interactive [Embabel hub](https://hub.embabel.com) where an Embabel agent answers questions about the framework in natural language.
