---
layout: post
title: "Catch2: The Natural C++ Testing Framework with Expression Decomposition"
description: "A deep dive into Catch2, the C++ unit testing framework that uses natural expression decomposition, self-registering test cases, sections for fixture-free setup sharing, and a multi-reporter architecture for CI integration."
date: 2026-09-01
header-img: "img/post-bg.jpg"
permalink: /Catch2-Natural-Cpp-Testing-Framework/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - C++
  - Testing
  - Unit Testing
  - BDD
  - BSL
author: "PyShine"
---
# Catch2: The Natural C++ Testing Framework with Expression Decomposition

Testing C++ code should not feel like writing boilerplate. Catch2, the second most popular C++ unit testing framework according to the 2022 JetBrains C++ ecosystem survey, is built on the principle that tests should be natural to write and easy to read. Test names are free-form strings, assertions use standard C++ boolean expressions (decomposed automatically for failure reporting), and sections provide a fixture-free way to share setup code. This post explores Catch2's architecture through its test execution pipeline, section tree model, expression decomposition mechanism, and multi-reporter system.

![Catch2 Test Execution Pipeline](/assets/img/diagrams/Catch2/1_pipeline.svg)

### Understanding the Test Execution Pipeline

The pipeline diagram above shows how Catch2 transforms self-registering test case macros into test results through a layered execution architecture.

**TEST_CASE Macros and Self-Registration**
Catch2 tests begin with the `TEST_CASE` macro, which takes a free-form test name (not a valid identifier) and optional tags in square brackets. The macro expands to a class with a static instance that registers itself with the `RegistryHub` during static initialization. This means you never need to manually register tests - just write `TEST_CASE("My test", "[tag]")` and Catch2 discovers it automatically. The same mechanism supports `SCENARIO` (BDD-style, prefixed with "Scenario: "), `TEMPLATE_TEST_CASE` (type-parametrized), `TEMPLATE_PRODUCT_TEST_CASE` (template-product parametrized), and `TEMPLATE_LIST_TEST_CASE` (type-list parametrized).

**Test Registry and RegistryHub**
The `RegistryHub` (in `catch_registry_hub.cpp`) is the central registry for all test cases, reporters, exception translators, and tag aliases. During static initialization, each `TEST_CASE` macro creates a `TestCaseRegistrar` that adds the test case to the registry. The registry stores `TestCaseInfo` objects containing the test name, tags, source location, and a pointer to the test function. This design allows the test binary to discover all tests at runtime without any external registration step.

**Session and Command-Line Parsing**
The `Session` class (in `catch_session.cpp`) is the entry point for test execution. It uses the Clara parser (a custom command-line argument parser in `catch_clara.cpp`) to handle arguments like `--reporter`, `--verbosity`, test name patterns, tag expressions, `--list-tests`, `--list-reporters`, and `--order` (for randomized test execution). The session configures the run context with the selected tests, reporter, and execution options before starting the test run.

**RunContext and Execution Loop**
The `RunContext` (in `catch_run_context.cpp`) manages the actual test execution loop. For each selected test case, it invokes the test function while tracking sections and generators. The RunContext coordinates with the `TestCaseTracker` for section tracking and the `AssertionHandler` for assertion processing. It also handles exception catching (for `REQUIRE_THROWS` and similar macros) and reports test case results to the active reporter.

**TestCaseTracker and Section Tracking**
The `TestCaseTracker` (in `catch_test_case_tracker.cpp`) implements Catch2's unique section tree model. As the test function executes, it tracks which sections have been entered and completed. The tracker walks the section tree in depth-first order, executing exactly one leaf section per run-through of the test case. This allows sections to share setup code without fixtures - the setup code runs fresh for each leaf section.

**AssertionHandler and Expression Decomposition**
The `AssertionHandler` (in `catch_assertion_handler.cpp`) processes each `REQUIRE`, `CHECK`, and related macro. It captures the source location, creates an `AssertionResult` with the decomposed expression, and reports the result to the active reporter. The expression decomposition mechanism (via `LazyExpr` in `catch_lazy_expr.cpp`) is what allows `REQUIRE(a == b + c)` to report both the original expression and the evaluated values on failure.

**Reporter Interface and Output**
The Reporter interface (defined in `catch_interfaces_reporter.hpp`) is the output abstraction. Catch2 supports multiple simultaneous reporters - for example, a Console reporter for human-readable output and a JUnit reporter for CI integration, both active in the same test run. The MultiReporter dispatches events to all active reporters and event listeners. Nine built-in reporters cover common output formats, and custom reporters can be registered for specialized needs.

## How to Build

Catch2 v3 is a proper compiled library (no longer single-header). It supports CMake, Bazel, and Meson.

### CMake (FetchContent)

```cmake
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.16.0
)
FetchContent_MakeAvailable(Catch2)

target_link_libraries(your_test_target PRIVATE Catch2::Catch2WithMain)
```

### CMake (System Install)

```bash
git clone --branch v3.16.0 --depth 1 https://github.com/catchorg/Catch2.git
cd Catch2

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local

cmake --build build --parallel $(nproc)
sudo cmake --install build
```

### Using catch_discover_tests

Catch2 provides a CMake helper to automatically register each `TEST_CASE` as a CTest test:

```cmake
include(Catch)
catch_discover_tests(your_test_target)
```

This calls the test binary with `--list-tests` and registers each test case individually with CTest, allowing fine-grained test selection and parallel execution in CI.

## Writing Tests

### Basic Test Cases

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

uint32_t factorial(uint32_t number) {
    return number <= 1 ? number : factorial(number - 1) * number;
}

TEST_CASE("Factorials are computed", "[factorial]") {
    REQUIRE(factorial(0) == 1);
    REQUIRE(factorial(1) == 1);
    REQUIRE(factorial(2) == 2);
    REQUIRE(factorial(3) == 6);
    REQUIRE(factorial(10) == 3'628'800);
}
```

If `factorial(0)` returns 0 instead of 1, Catch2 reports:

```
example.cpp:9: FAILED:
  REQUIRE( factorial(0) == 1 )
with expansion:
  0 == 1
```

Both the original expression and the expanded values are shown - this is expression decomposition in action.

## The Section Model

![Catch2 Section Tree Execution Model](/assets/img/diagrams/Catch2/2_sections.svg)

### Understanding Section Execution

The section diagram illustrates how Catch2's unique section mechanism eliminates the need for traditional test fixtures. Instead of creating a class with setup/teardown methods, you write setup code directly in the `TEST_CASE` body and use `SECTION` blocks to define individual test scenarios.

**Setup Code Sharing**
In the diagram, the `TEST_CASE` begins with setup code - creating a `std::vector<int> v(5)` and asserting its initial state. This setup code runs once for each leaf section. Unlike class-based fixtures where setup happens in a constructor, Catch2's setup is inline and visible, making it clear what state each section starts with.

**Section Tree and Depth-First Walk**
The `TestCaseTracker` maintains a tree of sections. Each `SECTION` block is a node in this tree. The tracker walks the tree in depth-first order, but crucially, only one leaf section executes per run-through of the `TEST_CASE`. This means the test case is re-entered from the beginning for each leaf section, ensuring fresh setup each time.

**Leaf-Only Execution**
The "Leaf Execution" node in the diagram emphasizes that only leaf sections (sections with no nested sections inside them) actually execute their assertions. Parent sections serve as grouping and setup-sharing mechanisms. This design prevents partial execution of nested sections and ensures each test scenario runs in isolation.

**Multiple Runs**
For a `TEST_CASE` with four leaf sections, the test case runs four times total - once per leaf. Each run starts fresh from the top of the `TEST_CASE`, executing the setup code and then entering the specific leaf section for that run. The section tracker remembers which leaves have already been executed and selects the next one.

**Section Nesting**
Sections can be nested arbitrarily deep. A nested section creates a sub-tree where the parent section is entered multiple times - once for each of its leaf descendants. The documentation recommends keeping nesting to 3 levels or fewer for readability. Nested sections are most useful when multiple tests share part of their setup - the outer section provides the common setup, and inner sections provide the specific scenarios.

**Comparison with Traditional Fixtures**
Traditional xUnit fixtures use class-based setup: each test method is a method on a fixture class, and setup/teardown happen in constructor/destructor. Catch2's sections invert this: setup is in the test case body, and sections define the variations. This is more natural for tests that share most setup but differ in specific operations, and it avoids the boilerplate of fixture classes.

### Using Sections

```cpp
#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("vectors can be sized and resized", "[vector]") {
    std::vector<int> v(5);

    REQUIRE(v.size() == 5);
    REQUIRE(v.capacity() >= 5);

    SECTION("resizing bigger changes size and capacity") {
        v.resize(10);
        REQUIRE(v.size() == 10);
        REQUIRE(v.capacity() >= 10);
    }

    SECTION("resizing smaller changes size but not capacity") {
        v.resize(0);
        REQUIRE(v.size() == 0);
        REQUIRE(v.capacity() >= 5);
    }

    SECTION("reserving bigger changes capacity but not size") {
        v.reserve(10);
        REQUIRE(v.size() == 5);
        REQUIRE(v.capacity() >= 10);
    }
}
```

### BDD-Style Testing

Catch2 provides BDD macros that map onto `TEST_CASE` and `SECTION`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <vector>
#include <string>

SCENARIO("vector can be sized and resized") {
    GIVEN("an empty vector") {
        auto v = std::vector<std::string>{};

        THEN("the size and capacity start at 0") {
            REQUIRE(v.size() == 0);
            REQUIRE(v.capacity() == 0);
        }

        WHEN("push_back() is called") {
            v.push_back("hello");

            THEN("the size changes") {
                REQUIRE(v.size() == 1);
                REQUIRE(v.capacity() >= 1);
            }
        }
    }
}
```

## Assertion Decomposition

![Catch2 Assertion Decomposition](/assets/img/diagrams/Catch2/3_assertions.svg)

### Understanding Expression Decomposition

The assertion diagram shows how Catch2 decomposes natural C++ expressions for detailed failure reporting. This is one of Catch2's most distinctive features - you write assertions as normal C++ boolean expressions, and Catch2 automatically breaks them down for reporting.

**Natural Expressions**
Unlike frameworks that require `ASSERT_EQUALS(a, b)` or `EXPECT_THAT(a, Eq(b))`, Catch2 uses standard C++ operators: `REQUIRE(a == b)`. This is more natural to write and read. The `REQUIRE` macro aborts the test case on failure, while `CHECK` continues execution (useful for seeing multiple failures in one test).

**Macro Expansion and Source Location**
The `REQUIRE` macro (defined in `catch_test_macros.hpp`) expands to create an `AssertionHandler` that captures the source location (file, line) using `__FILE__` and `__LINE__`. This information is included in failure reports, making it easy to locate failing assertions.

**Expression Decomposition via LazyExpr**
The key to decomposition is the `LazyExpr` template (in `catch_lazy_expr.cpp`). When you write `REQUIRE(a == b + c)`, the `==` operator is overloaded to return a `DecomposedExpression` object that stores references to the left and right operands and the operator, rather than immediately evaluating to a boolean. This allows Catch2 to stringify both the original expression and the individual operand values when a failure occurs.

**LHS and RHS Evaluation**
The decomposition captures the left-hand side (`a`) and right-hand side (`b + c`) separately. Each side is stringified using `Catch::StringMaker<T>`, which can be specialized for custom types. This means that when `REQUIRE(factorial(0) == 1)` fails, Catch2 can report `0 == 1` - showing both the actual return value and the expected value.

**AssertionResult**
The `AssertionResult` object bundles the pass/fail boolean, the original expression string, the decomposed values, and the source location. This object is passed to the active reporter, which formats it for output. The reporter receives complete information about each assertion, enabling rich failure messages.

**Failure Reporting**
On failure, Catch2 reports the original expression (e.g., `REQUIRE(a == b + c)`) and the expanded values (e.g., `3 == 1 + 2`). This dual representation is invaluable for debugging - you see both what you wrote and what the values actually were. The failure message also includes the file and line number for quick navigation.

**Limitations of Decomposition**
Expressions containing `&&` or `||` cannot be decomposed because overloading these operators would break short-circuit evaluation. To test such expressions, either enclose them in parentheses (which forces evaluation before decomposition) or split them into separate assertions. This is a deliberate trade-off - Catch2 prioritizes correct semantics over decomposition coverage.

## The Reporter System

![Catch2 Reporter System](/assets/img/diagrams/Catch2/4_reporters.svg)

### Understanding Reporters and Events

The reporter diagram shows how Catch2's output system works through a multi-reporter architecture with event-based dispatch.

**Test Events**
Catch2 defines a comprehensive set of test events that reporters can handle: `testRunStarting`, `testCaseStarting`, `sectionStarting`, `assertionStarting`, `assertionEnded`, `sectionEnded`, `testCaseEnded`, `testCasePartialStarting`, `testCasePartialEnded`, and `testRunEnded`. These events provide reporters with complete visibility into the test execution lifecycle.

**MultiReporter Dispatch**
The `MultiReporter` receives all events and dispatches them to every active reporter and event listener. This enables a key feature: multiple reporters can be active simultaneously. For example, you can have a Console reporter writing human-readable output to stdout while a JUnit reporter writes machine-readable XML to a file - both receiving the same events from a single test run.

**Nine Built-In Reporters**
Catch2 includes nine reporters covering common output formats:
- **Console** - Human-readable, colorized output (default)
- **Compact** - Minimal, one line per test
- **XML** - Machine-parseable structured XML
- **JUnit** - CI integration format for Jenkins/Bamboo
- **TAP** - Test Anything Protocol (Perl-style)
- **Automake** - Makefile integration format
- **SonarQube** - Code quality metrics import
- **TeamCity** - JetBrains CI service messages
- **JSON** - Structured JSON output

**Custom Reporters**
You can write custom reporters by deriving from `Catch::ReporterBase` (handling all events) or from utility bases like `StreamingReporterBase` or `CumulativeReporterBase` (handling only the events you need). Custom reporters are registered with `CATCH_REGISTER_REPORTER`. This extensibility allows integration with any CI system or custom tooling.

**Event Listeners**
Event listeners are lightweight hooks that receive events but do not produce output. They are registered with `CATCH_REGISTER_LISTENER` and are useful for side effects like logging to external systems, collecting metrics, or triggering actions on test failure. Listeners are always active (unlike reporters which are selected via command line).

**Multiple Reporter Configuration**
You can specify multiple reporters on the command line with different output destinations and options:
```bash
./tests --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
```
This runs the JUnit reporter writing to a file and the Console reporter writing to stdout with ANSI colors.

## Data Generators

Catch2's data generators (data-driven testing) allow you to run the same assertions across multiple input values:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

TEST_CASE("Generators") {
    auto i = GENERATE(1, 3, 5);
    REQUIRE(is_odd(i));
}

TEST_CASE("Range generators") {
    auto value = GENERATE(range(1, 100));
    REQUIRE(value > 0);
    REQUIRE(value <= 100);
}

TEST_CASE("Cartesian product") {
    auto i = GENERATE(1, 2);
    auto j = GENERATE(3, 4, 5);
    // Runs 6 times (2 * 3)
    REQUIRE(i != j);
}
```

Generators can be combined with sections - a `GENERATE` call acts as an implicit section from its point of use to the end of the scope.

## Microbenchmarking

Catch2 includes basic microbenchmarking support via the `BENCHMARK` macro:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

uint64_t fibonacci(uint64_t n) {
    return n < 2 ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

TEST_CASE("Benchmark Fibonacci", "[!benchmark]") {
    REQUIRE(fibonacci(5) == 5);

    BENCHMARK("fibonacci 20") {
        return fibonacci(20);
    };

    BENCHMARK("fibonacci 25") {
        return fibonacci(25);
    };
}
```

Benchmarks are tagged with `[!benchmark]` and are not run by default. The benchmarking infrastructure includes statistical analysis (mean, standard deviation, outlier classification) and supports multiple clocks for accurate measurement.

## Key Features

| Feature | Description |
|---------|-------------|
| Natural Expressions | REQUIRE/CHECK use standard C++ operators; expressions auto-decomposed for failure reporting |
| Self-Registering Tests | TEST_CASE macros auto-register via static initialization; no manual registration |
| Sections | Fixture-free setup sharing via depth-first section tree; only leaf sections execute |
| BDD Macros | SCENARIO/GIVEN/WHEN/THEN map onto TEST_CASE/SECTION with formatting |
| Template Tests | TEMPLATE_TEST_CASE, TEMPLATE_PRODUCT_TEST_CASE for type-parametrized testing |
| Data Generators | GENERATE macro for data-driven testing with cartesian product support |
| Expression Decomposition | LazyExpr template breaks expressions into LHS/op/RHS for rich failure messages |
| Multi-Reporter | 9 built-in reporters (Console, XML, JUnit, Compact, TAP, etc.); run multiple simultaneously |
| Custom Reporters | Derive from ReporterBase; register with CATCH_REGISTER_REPORTER |
| Event Listeners | Lightweight side-effect hooks via CATCH_REGISTER_LISTENER |
| Microbenchmarking | BENCHMARK macro with statistical analysis and multiple clock support |
| Matchers | Hamcrest-style matchers for complex assertions (Contains, Equals, Predicate, etc.) |
| Tag System | Free-form tags for test grouping; special tags like [!throws], [!mayfail], [!benchmark] |
| CMake Integration | catch_discover_tests auto-registers TEST_CASEs with CTest |
| Floating Point | Catch::Approx for tolerant comparisons; full set of floating-point matchers |
| Cross-Platform | Windows, Linux, macOS; C++14 required (v3.x); BSL-1.0 license |

## Troubleshooting

**Linker error: undefined reference to `Catch2Main`**
You need to link against `Catch2::Catch2WithMain` (which includes the default `main` function) or provide your own `main` and link against `Catch2::Catch2` only. If providing your own main, include `#define CATCH_CONFIG_MAIN` before `#include <catch2/catch_test_macros.hpp>` in your main file.

**v2 to v3 Migration**
Catch2 v3 is no longer single-header. The migration involves:
1. Replace `#include "catch.hpp"` with specific includes like `#include <catch2/catch_test_macros.hpp>`
2. Link against the Catch2 library (CMake target `Catch2::Catch2WithMain`)
3. Use `catch_amalgamated.hpp`/`.cpp` from the `extras/` directory for the two-file distribution
4. Consult `docs/migrate-v2-to-v3.md` for detailed migration instructions

**Expression Decomposition Fails with `&&` or `||`**
Expressions with `&&` or `||` cannot be decomposed. Either enclose in parentheses: `REQUIRE((a && b))`, or split into separate assertions: `REQUIRE(a); REQUIRE(b)`. This is by design - overloading `&&`/`||` would break short-circuit evaluation.

**Section Setup Runs Too Many Times**
Each leaf section causes a fresh run-through of the `TEST_CASE`, including setup code. If setup is expensive, consider using class-based fixtures instead, or restructure tests to reduce the number of leaf sections. The documentation recommends keeping section nesting to 3 levels or fewer.

**Tests Not Discovered by CTest**
Ensure `catch_discover_tests` is called after the test target is defined and linked. The discovery script calls the test binary with `--list-tests`, so the binary must be runnable at CMake configure time. If the binary has missing DLL dependencies, discovery will fail.

**Random Test Ordering Not Working**
Use `--order rand` (or `--order lex` for lexicographic, `--order decl` for declaration order). The random seed can be specified with `--rng-seed time` or `--rng-seed <value>` for reproducibility. The actual seed used is reported in the output.

## Conclusion

Catch2's design philosophy is that testing should be natural - test names should be readable, assertions should look like normal C++ code, and setup sharing should not require class boilerplate. The section tree model achieves fixture-free setup sharing by re-entering the test case for each leaf section, providing fresh state without the overhead of fixture classes. The expression decomposition mechanism, powered by the LazyExpr template, delivers rich failure messages from natural C++ expressions - you write `REQUIRE(a == b)` and get `3 == 1 + 2` on failure.

The multi-reporter architecture enables seamless CI integration: one test run produces both human-readable console output and machine-parseable JUnit XML, with no extra configuration. The nine built-in reporters cover the major CI and output formats, and the extensibility through custom reporters and event listeners ensures that Catch2 can integrate with any toolchain. Combined with data generators for data-driven testing, template test cases for type-parametrized testing, and microbenchmarking support, Catch2 provides a comprehensive testing toolkit that has made it the second most popular C++ testing framework in the ecosystem.

## Links

- [Catch2 GitHub Repository](https://github.com/catchorg/Catch2)
- [Catch2 Tutorial](https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md)
- [Why Catch2?](https://github.com/catchorg/Catch2/blob/devel/docs/why-catch.md)
- [Test Cases and Sections](https://github.com/catchorg/Catch2/blob/devel/docs/test-cases-and-sections.md)
- [Assertion Macros](https://github.com/catchorg/Catch2/blob/devel/docs/assertions.md)
- [Reporters Documentation](https://github.com/catchorg/Catch2/blob/devel/docs/reporters.md)
- [Catch2 v3.16.0 Release Notes](https://github.com/catchorg/Catch2/blob/devel/docs/release-notes.md)
- [Catch2 Discord](https://discord.gg/4CWS9zD)

## Related Posts

- [abseil-cpp: Google's Foundation C++ Libraries](/abseil-cpp-Google-Foundation-Cpp-Libraries/)
- [yaml-cpp: YAML 1.2 Parser and Emitter in C++](/yaml-cpp-YAML-Parser-Emitter-Cpp/)
- [meshoptimizer: Mesh Optimization for GPU Rendering](/meshoptimizer-Mesh-Optimization-GPU-Rendering/)
