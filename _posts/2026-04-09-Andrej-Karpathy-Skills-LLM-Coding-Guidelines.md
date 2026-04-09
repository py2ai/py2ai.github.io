---
layout: post
title: "Andrej Karpathy Skills: LLM Coding Guidelines That Prevent Common Mistakes"
description: "Learn how to reduce LLM coding mistakes with four key principles from Andrej Karpathy's observations: Think Before Coding, Simplicity First, Surgical Changes, and Goal-Driven Execution."
date: 2026-04-09
header-img: "img/post-bg.jpg"
permalink: /Andrej-Karpathy-Skills-LLM-Coding-Guidelines/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - LLM
  - Coding Best Practices
  - Andrej Karpathy
author: "PyShine"
---

# Andrej Karpathy Skills: LLM Coding Guidelines That Prevent Common Mistakes

Large Language Models have revolutionized software development, enabling developers to write code faster than ever before. However, as Andrej Karpathy observed, LLMs come with their own set of behavioral pitfalls that can lead to bloated code, unnecessary changes, and hidden assumptions. The `forrestchang/andrej-karpathy-skills` repository addresses these issues head-on with four core principles designed to make LLM-assisted coding more reliable and maintainable.

## The Problem: LLM Coding Pitfalls

Andrej Karpathy, former Director of AI at Tesla and co-founder of OpenAI, shared critical observations about how LLMs behave when writing code:

> "The models make wrong assumptions on your behalf and just run along with them without checking. They don't manage their confusion, don't seek clarifications, don't surface inconsistencies, don't present tradeoffs, don't push back when they should."

> "They really like to overcomplicate code and APIs, bloat abstractions, don't clean up dead code... implement a bloated construction over 1000 lines when 100 would do."

> "They still sometimes change/remove comments and code they don't sufficiently understand as side effects, even if orthogonal to the task."

These observations highlight three fundamental problems with LLM coding behavior:

1. **Hidden Assumptions**: LLMs silently interpret ambiguous requests without surfacing their assumptions
2. **Overengineering**: They create complex abstractions and speculative features that weren't requested
3. **Collateral Changes**: They modify unrelated code while performing simple tasks

The `andrej-karpathy-skills` repository provides a solution: a single `CLAUDE.md` file containing four principles that directly address these issues.

## Four Principles Overview

![Four Principles Overview](/assets/img/diagrams/karpathy-principles-overview.svg)

### Understanding the Four Principles

The diagram above illustrates the four core principles that form the foundation of better LLM coding behavior. Each principle targets a specific category of LLM mistakes while providing actionable guidelines for improvement.

**Principle 1: Think Before Coding**

This principle addresses the tendency of LLMs to make silent assumptions. When faced with an ambiguous request, LLMs often pick an interpretation and proceed without clarification. This leads to solutions that may not match the user's actual intent.

The principle enforces explicit reasoning through several mechanisms:
- State assumptions explicitly before implementation
- Present multiple interpretations when ambiguity exists
- Push back when a simpler approach is available
- Stop and ask for clarification when confused

By forcing LLMs to surface their assumptions, developers can catch misunderstandings before code is written, saving time and reducing rework.

**Principle 2: Simplicity First**

This principle combats the overengineering tendency inherent in LLMs. When asked to implement a feature, LLMs often create elaborate abstractions, configuration systems, and error handling for scenarios that may never occur.

The principle establishes clear boundaries:
- No features beyond what was explicitly requested
- No abstractions for single-use code
- No speculative flexibility or configurability
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

The test is simple: would a senior engineer say this is overcomplicated? If yes, simplify.

**Principle 3: Surgical Changes**

This principle addresses the collateral damage problem. When LLMs edit existing code, they often "improve" adjacent code, reformat files, or refactor things that aren't broken.

The principle establishes strict boundaries for code modifications:
- Don't improve adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd write it differently
- Only mention unrelated dead code, don't delete it

The test: every changed line should trace directly to the user's request.

**Principle 4: Goal-Driven Execution**

This principle transforms how LLMs approach tasks. Instead of imperative instructions ("fix the bug"), it encourages declarative goals with verification ("write a test that reproduces the bug, then make it pass").

This approach leverages the LLM's ability to loop until success criteria are met:
- Transform "add validation" into "write tests for invalid inputs, then make them pass"
- Transform "fix the bug" into "write a test that reproduces it, then make it pass"
- Transform "refactor X" into "ensure tests pass before and after"

Strong success criteria allow LLMs to work independently. Weak criteria require constant clarification.

## Anti-Patterns vs Correct Approach

![Anti-Patterns Flowchart](/assets/img/diagrams/karpathy-antipatterns-flowchart.svg)

### Understanding LLM Anti-Patterns

The flowchart above demonstrates the contrast between common LLM anti-patterns and the correct approach advocated by the Karpathy principles. Understanding these patterns is essential for recognizing when an LLM is going astray.

**Anti-Pattern 1: Hidden Assumptions**

When a user requests "Add a feature to export user data," an LLM following anti-patterns might:
- Assume it should export ALL users without considering pagination or privacy
- Assume a file-based export when an API endpoint might be preferred
- Assume which fields to include without asking about sensitive data
- Implement the solution immediately without clarification

The correct approach surfaces these assumptions upfront:
- Ask about scope (all users or filtered subset?)
- Clarify format (download file, background job, or API endpoint?)
- Confirm which fields to include
- Understand volume requirements

This clarification phase takes seconds but saves hours of rework.

**Anti-Pattern 2: Over-Abstraction**

When asked to "Add a function to calculate discount," an LLM might create:
- An abstract `DiscountStrategy` base class
- Multiple implementations (PercentageDiscount, FixedDiscount)
- A `DiscountConfig` dataclass with validation
- A `DiscountCalculator` class with dependency injection
- 100+ lines of code for what should be a simple calculation

The correct approach starts simple:
- A single function: `calculate_discount(amount, percent)`
- Add complexity only when requirements demand it
- Refactor when multiple discount types become necessary

**Anti-Pattern 3: Drive-by Refactoring**

When asked to "Fix the bug where empty emails crash the validator," an LLM might:
- Fix the bug AND improve email validation
- Add username validation nobody asked for
- Reformat quotes from single to double
- Add type hints and docstrings
- Change the function signature

The correct approach is surgical:
- Only change lines that fix the reported issue
- Preserve existing style and formatting
- Leave unrelated code untouched

**Anti-Pattern 4: Vague Goals**

When asked to "Fix the authentication system," an LLM might:
- Make broad changes without clear success criteria
- Implement improvements without verification
- Create new features instead of fixing existing issues

The correct approach defines verifiable goals:
- Write a test that reproduces the specific issue
- Implement the fix
- Verify all tests pass
- Check for regressions

## Goal-Driven Execution Workflow

![Goal-Driven Execution Workflow](/assets/img/diagrams/karpathy-goal-driven-workflow.svg)

### Understanding Goal-Driven Execution

The workflow diagram above illustrates how goal-driven execution transforms the development process from vague imperative instructions into verifiable, incremental progress. This approach leverages the LLM's exceptional ability to loop until specific goals are met.

**The Problem with Imperative Instructions**

Traditional instructions like "add validation" or "fix the bug" are problematic because:
- They lack clear success criteria
- They don't define what "done" looks like
- They require constant back-and-forth for clarification
- They make it hard to measure progress

**The Goal-Driven Alternative**

Goal-driven execution transforms these instructions into verifiable goals:

| Instead of... | Transform to... |
|--------------|-----------------|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

**The Workflow Steps**

**Step 1: Define Success Criteria**
Before writing any code, clearly define what success looks like. For a rate limiting feature:
- Test: 100 requests to endpoint, first 10 succeed, rest get 429
- Manual verification: curl endpoint 11 times, see rate limit error

**Step 2: Write Failing Tests**
Create tests that fail because the feature doesn't exist. This proves you understand the requirement and provides a verification mechanism.

**Step 3: Implement Minimum Code**
Write the simplest code that makes tests pass. No speculative features, no over-engineering.

**Step 4: Verify Success**
Run tests to confirm the implementation works. If tests fail, loop back to step 3.

**Step 5: Check for Regressions**
Ensure existing functionality still works. This prevents collateral damage.

**Step 6: Increment or Complete**
If more work is needed, define the next verifiable goal. Otherwise, mark complete.

**Benefits of Goal-Driven Execution**

This approach provides several advantages:
- **Measurable Progress**: Each step has clear verification criteria
- **Reduced Rework**: Tests catch misunderstandings early
- **Independent Work**: LLMs can loop without constant human input
- **Regression Prevention**: Existing tests ensure no collateral damage
- **Documentation**: Tests serve as living documentation of requirements

**Multi-Step Task Planning**

For complex tasks, state a brief plan with verification at each step:

```
1. Add basic in-memory rate limiting (single endpoint)
   Verify: Test passes, manual check works

2. Extract to middleware (apply to all endpoints)
   Verify: Rate limits apply to multiple endpoints, existing tests pass

3. Add Redis backend (for multi-server)
   Verify: Rate limit persists across restarts, shared between instances

4. Add configuration (rates per endpoint)
   Verify: Different endpoints have different limits, config file parsed correctly
```

Each step is independently verifiable and deployable, reducing risk and enabling incremental delivery.

## Surgical Changes Decision Tree

![Surgical Changes Decision Tree](/assets/img/diagrams/karpathy-surgical-changes-tree.svg)

### Understanding Surgical Changes

The decision tree above provides a systematic approach to making surgical code changes. This principle is critical for maintaining code quality and preventing the "collateral damage" that LLMs often introduce when editing existing code.

**The Problem with Non-Surgical Changes**

When LLMs edit code, they often:
- Reformat code to match their preferred style
- Add type hints or docstrings to existing functions
- Refactor adjacent code that wasn't broken
- Delete comments or code they don't understand
- "Improve" variable names or function signatures

These changes may seem harmless, but they:
- Create noise in version control diffs
- Introduce bugs in previously working code
- Violate the principle of least surprise
- Make code review more difficult
- Break existing tests or integrations

**The Decision Tree Approach**

**Question 1: Is this line directly related to the user's request?**

If NO: Don't change it. Even if you see "better" ways to do things, leave it alone.

If YES: Proceed to the next question.

**Question 2: Is this the minimum change needed?**

If NO: Simplify. Find the smallest change that accomplishes the goal.

If YES: Proceed to the next question.

**Question 3: Does it match existing style?**

If NO: Adjust to match. Use the same quote style, indentation, naming conventions, and patterns as the surrounding code.

If YES: Proceed to the next question.

**Question 4: Does it preserve existing behavior?**

If NO: Reconsider. Unless the request explicitly asks for behavior change, preserve existing functionality.

If YES: Proceed with the change.

**Examples of Surgical vs Non-Surgical Changes**

**User Request**: "Fix the bug where empty emails crash the validator"

**Non-Surgical (Wrong)**:
```diff
- # Check email format
+ """Validate user data with comprehensive checks."""
  if not user_data.get('email'):
+     email = user_data.get('email', '').strip()
+     if not email:
          raise ValueError("Email required")
+     if '@' not in email or '.' not in email.split('@')[1]:
+         raise ValueError("Invalid email format")
  
- # Check username
+ # Validate username with length requirements
  if not user_data.get('username'):
      raise ValueError("Username required")
+ if len(username) < 3:
+     raise ValueError("Username too short")
```

**Surgical (Correct)**:
```diff
  if not user_data.get('email'):
+     email = user_data.get('email', '')
+     if not email or not email.strip():
          raise ValueError("Email required")
```

Only the lines that fix the empty email bug are changed. No additional validation, no refactoring, no style changes.

**The Orphan Code Rule**

When your changes make existing code unused:
- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked
- Mention unrelated dead code in comments or to the user, don't delete it

This ensures you clean up after yourself without overstepping.

## Simplicity First Comparison

![Simplicity First Comparison](/assets/img/diagrams/karpathy-simplicity-comparison.svg)

### Understanding Simplicity First

The comparison diagram above illustrates the stark contrast between overengineered solutions and simple solutions. This principle is perhaps the most counterintuitive for LLMs, which are trained on vast amounts of code that often includes sophisticated patterns and abstractions.

**Why LLMs Overengineer**

LLMs tend to create complex solutions because:
- They've seen many design patterns in training data
- They anticipate future requirements that may never materialize
- They follow "best practices" even when inappropriate
- They want to demonstrate comprehensive solutions
- They don't have the context to know what's truly needed

**The Overengineering Problem**

Consider a request to "Add a function to calculate discount":

**Overengineered Solution (100+ lines)**:
- Abstract base class for discount strategies
- Multiple strategy implementations
- Configuration dataclass with validation
- Factory pattern for strategy selection
- Dependency injection for testability
- Error handling for edge cases that may never occur

This solution follows design patterns and best practices, but it's fundamentally wrong for a simple requirement. It:
- Takes longer to write
- Is harder to understand
- Has more potential bugs
- Is harder to test
- Requires more maintenance

**Simple Solution (5 lines)**:
```python
def calculate_discount(amount: float, percent: float) -> float:
    """Calculate discount amount. percent should be 0-100."""
    return amount * (percent / 100)
```

This solution:
- Takes minutes to write
- Is immediately understandable
- Has minimal bug surface
- Is trivial to test
- Requires minimal maintenance

**When to Add Complexity**

The principle isn't "never add complexity" - it's "add complexity when needed, not before":

| Scenario | Approach |
|----------|----------|
| Single discount type | Simple function |
| Multiple discount types needed | Add strategy pattern |
| Configuration required | Add config system |
| Performance matters | Add caching |

**The Senior Engineer Test**

Ask yourself: "Would a senior engineer say this is overcomplicated?"

If yes, simplify. Senior engineers understand that:
- Complexity has costs beyond implementation time
- Simple code is easier to debug, test, and maintain
- Requirements change, making speculative features wasteful
- YAGNI (You Aren't Gonna Need It) is a valid principle

**Speculative Features to Avoid**

When implementing a feature, avoid adding:
- Configuration options nobody asked for
- Error handling for impossible scenarios
- Abstractions for single implementations
- Flexibility for future requirements
- Logging, metrics, or monitoring beyond requirements
- Documentation beyond what's necessary

Add these when they're actually needed, not when you imagine they might be.

**The Refactoring Mindset**

Simple doesn't mean "never refactor." It means:
- Start simple
- Refactor when complexity becomes necessary
- Don't pre-optimize for imagined future needs
- Trust that you can add complexity later

Good code solves today's problem simply, not tomorrow's problem prematurely.

## Installation and Usage

The `andrej-karpathy-skills` repository provides two installation options:

### Option A: Claude Code Plugin (Recommended)

From within Claude Code, add the marketplace and install:

```
/plugin marketplace add forrestchang/andrej-karpathy-skills
/plugin install andrej-karpathy-skills@karpathy-skills
```

This installs the guidelines as a Claude Code plugin, making the skill available across all your projects.

### Option B: CLAUDE.md (Per-Project)

For new projects:
```bash
curl -o CLAUDE.md https://raw.githubusercontent.com/forrestchang/andrej-karpathy-skills/main/CLAUDE.md
```

For existing projects (append):
```bash
echo "" >> CLAUDE.md
curl https://raw.githubusercontent.com/forrestchang/andrej-karpathy-skills/main/CLAUDE.md >> CLAUDE.md
```

## Practical Examples

The repository includes extensive examples demonstrating each principle in action. Here are key patterns:

### Think Before Coding

**Wrong**: Silently assume file format and implement export.

**Right**: Ask clarifying questions:
- What format? (JSON, CSV, XML)
- What fields? (some may be sensitive)
- What scope? (all users or filtered)
- What volume? (affects approach)

### Simplicity First

**Wrong**: Create strategy pattern for single discount type.

**Right**: Write a simple function. Add complexity when multiple discount types are actually needed.

### Surgical Changes

**Wrong**: Fix bug AND improve validation AND reformat code AND add type hints.

**Right**: Only change lines that fix the reported bug. Match existing style.

### Goal-Driven Execution

**Wrong**: "I'll review and improve the code."

**Right**: "Write test for bug X, make it pass, verify no regressions."

## How to Know It's Working

These guidelines are working if you see:

- **Fewer unnecessary changes in diffs** - Only requested changes appear
- **Fewer rewrites due to overcomplication** - Code is simple the first time
- **Clarifying questions come before implementation** - Not after mistakes
- **Clean, minimal PRs** - No drive-by refactoring or "improvements"

## Tradeoff Note

These guidelines bias toward **caution over speed**. For trivial tasks (simple typo fixes, obvious one-liners), use judgment - not every change needs the full rigor.

The goal is reducing costly mistakes on non-trivial work, not slowing down simple tasks.

## Conclusion

The `andrej-karpathy-skills` repository provides a practical solution to common LLM coding pitfalls. By following four principles - Think Before Coding, Simplicity First, Surgical Changes, and Goal-Driven Execution - developers can significantly improve the quality of LLM-assisted code.

These principles aren't about restricting LLM capabilities; they're about channeling those capabilities more effectively. When LLMs surface assumptions, write simple code, make targeted changes, and work toward verifiable goals, they become more reliable coding partners.

The repository is available at [https://github.com/forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) and can be installed as a Claude Code plugin or added to any project as a `CLAUDE.md` file.

## Related Posts

- [AgentSkillOS: Skill Orchestration System](/AgentSkillOS-Skill-Orchestration-System/)
- [MattPocock Skills: AI Agent Workflows](/MattPocock-Skills-AI-Agent-Workflows/)
- [DESIGN.md: AI-Powered Design Systems](/DESIGN-md-AI-Powered-Design-Systems/)