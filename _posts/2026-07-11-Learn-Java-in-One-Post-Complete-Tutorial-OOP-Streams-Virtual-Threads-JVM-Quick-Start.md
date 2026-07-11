---
layout: post
title: "Learn Java in a Single Post: A Complete Java Tutorial from OOP and Streams to Virtual Threads and the JVM"
description: "A complete Java tutorial in one blog post. Covers the whole language in 5 stages: fundamentals (primitives, methods, control flow), OOP (classes, interfaces, inheritance, polymorphism, sealed), collections + exceptions + generics, modern Java (lambdas, streams, Optional, records, pattern matching), and concurrency + JVM (threads, ExecutorService, CompletableFuture, virtual threads, GC, classloader, modules). Five diagrams, runnable code snippets, and a quick-start roadmap."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Learn-Java-in-One-Post-Complete-Tutorial-OOP-Streams-Virtual-Threads-JVM-Quick-Start/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Java
  - Tutorial
  - Programming
  - JVM
  - Streams
  - Virtual Threads
author: "PyShine"
---

# Learn Java in a Single Post: A Complete Java Tutorial from OOP and Streams to Virtual Threads and the JVM

Java is the enterprise default — strongly typed, garbage-collected, runs on the JVM (write once, run anywhere), and backed by one of the largest ecosystems on earth. For decades it had a reputation for verbosity; modern Java (8 through 21+) has changed that with lambdas, streams, records, sealed types, pattern matching, and virtual threads.

This post teaches the whole language in five stages with runnable snippets. By the end you'll understand OOP, generics, streams, the concurrency models (including Project Loom's virtual threads), and what the JVM actually does when you run `java Main`.

We target **Java 21 LTS** with notes on **22+** features. Everything here compiles on a current JDK.

## The Roadmap

![Java Roadmap](/assets/img/diagrams/java-tutorial/java-roadmap.svg)

1. **Fundamentals** — `main`, primitives vs wrappers, `String`, control flow, arrays, methods, `static`
2. **OOP** — classes, constructors, `this`/`super`, inheritance, interfaces vs abstract classes, polymorphism
3. **Collections + Exceptions** — `List`/`Set`/`Map`, generics, wildcards, checked vs unchecked, try-with-resources
4. **Modern Java** — lambdas, method references, streams, `Optional`, records, sealed, pattern matching, `var`, text blocks
5. **Concurrency + JVM** — threads, `ExecutorService`, `CompletableFuture`, virtual threads, GC, classloader, modules

## Stage 1 — Fundamentals

### A program

```java
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
}
```

Every Java program is a class with a `public static void main(String[] args)` entry point. Java is class-centric — everything lives in a class. Run it:

```bash
javac Main.java     # compile -> Main.class (bytecode)
java Main           # run on the JVM
```

### Primitives vs wrappers

```java
int n = 10;              // primitive - value type, fast, stack
Integer boxed = 10;      // wrapper - object, can be null, slower
double x = 3.14;
boolean ok = true;
char c = 'A';
long big = 9_000_000_000L;   // _ as digit separator
var inferred = 5;             // var infers type (local only, Java 10+)
final int MAX = 100;          // const

// 8 primitives: byte short int long float double char boolean
// wrappers: Byte Short Integer Long Float Double Character Boolean
// autoboxing: Integer i = 5;  int n = i;
```

Primitives are value types stored on the stack; wrappers are objects on the heap, nullable, and needed for generics (`List<int>` is illegal; `List<Integer>` works). Watch for autoboxing overhead in hot loops and `NullPointerException` on unboxing null.

### Strings

```java
String s = "Hello, " + "Java";    // + concatenates
String name = "Ada";
String greeting = String.format("Hi %s, %d", name, 42);  // printf-style
String f = "repeat".repeat(3);
String block = """
    text block
    multi-line
    """;                            // Java 15+

// String is IMMUTABLE - methods return new strings
s.length(); s.charAt(0); s.substring(1, 3); s.toUpperCase();
s.strip(); s.lines(); s.split(",");

// Mutable strings for building
StringBuilder sb = new StringBuilder();
sb.append("a").append("b").append("c");   // efficient concat in loops
```

`String` is immutable — every "modification" creates a new `String`. Use `StringBuilder` for repeated concatenation in loops (the `+` operator in a loop creates a new object each iteration).

### Control flow

```java
if (x > 0) { } else if (x == 0) { } else { }
switch (day) {                       // classic switch (fallthrough)
    case "MON": System.out.println("M"); break;
    default: System.out.println("?");
}
// switch expression (14+, stable 17+) - no fallthrough, returns value
String s = switch (day) {
    case "MON", "TUE" -> "weekday";
    case "SAT", "SUN" -> "weekend";
    default -> "?";
};
for (int i = 0; i < 10; i++) { }
for (String s : list) { }            // enhanced for / for-each
while (cond) { }
```

### Arrays and methods

```java
int[] nums = {1, 2, 3};
int[] arr = new int[5];              // zero-filled
nums[0] = 10;
int len = nums.length;               // not a method - field
var matrix = new int[3][3];          // 2D array

// Methods
public static int add(int a, int b) { return a + b; }
public static String greet(String name) { return "Hi " + name; }

// Varargs
public static int sum(int... nums) {
    int s = 0; for (int n : nums) s += n; return s;
}
sum(1, 2, 3);                       // 6
```

## Stage 2 — OOP

![Java OOP](/assets/img/diagrams/java-tutorial/java-oop.svg)

### Classes and constructors

```java
public class Counter {
    private int count;                          // field (instance state)
    public static final int MAX = 100;          // class constant

    public Counter() { this.count = 0; }        // no-arg constructor
    public Counter(int start) { this.count = start; }  // overloaded; this = current instance

    public void inc() { if (count < MAX) count++; }
    public int get() { return count; }            // accessor

    @Override
    public String toString() { return "Counter(" + count + ")"; }  // override Object
}

Counter c = new Counter(5);
c.inc();
System.out.println(c.get());   // 6
```

Access modifiers: `public` (everyone), `protected` (package + subclasses), package-private (default, package only), `private` (class only). **Make fields `private`; expose via methods** — encapsulation.

### Inheritance, super, override

```java
class Animal {
    protected String name;
    public Animal(String name) { this.name = name; }
    public String sound() { return "..."; }
}

class Dog extends Animal {                        // single inheritance
    public Dog(String name) { super(name); }       // call parent constructor
    @Override
    public String sound() { return "woof"; }       // override
}

Animal a = new Dog("Rex");
a.sound();   // "woof"  - dynamic dispatch (polymorphism)
```

Java has **single class inheritance** (`extends` one class) but **multiple interface implementation** (`implements` many). `@Override` is optional but catches typos (a method that doesn't actually override fails to compile).

### Interface vs abstract class

```java
interface Comparable<T> {
    int compareTo(T other);                       // abstract method
    default String describe() { return "?"; }      // default method (8+)
    static int compare(Comparable<?> a, Comparable<?> b) { return 0; }  // static
}

abstract class Shape {
    abstract double area();                       // must be implemented
    public String describe() { return "Shape area=" + area(); }  // concrete
}
class Circle extends Shape {
    private final double r;
    public Circle(double r) { this.r = r; }
    @Override double area() { return Math.PI * r * r; }
}

// Multiple interface implementation
class Duck implements Comparable<Duck>, Swimmer { ... }
```

**Use interfaces for capabilities** ("can be compared", "can swim") — a class can implement many. **Use abstract classes for shared implementation** with a partial skeleton — a class can extend only one. Prefer interfaces by default; reach for abstract class when you need shared state or constructors.

### Sealed types (17+)

```java
public sealed interface Shape permits Circle, Square, Triangle {}
public final class Circle implements Shape { ... }
public final class Square implements Shape { ... }
public final class Triangle implements Shape { ... }

// Exhaustive pattern matching - compiler knows all implementations
double area(Shape s) {
    return switch (s) {
        case Circle c -> Math.PI * c.r * c.r;
        case Square s -> s.s * s.s;
        case Triangle t -> 0.5 * t.b * t.h;
    };     // no default needed - exhaustive over permits
}
```

`sealed` + `permits` creates **closed hierarchies**: the compiler knows every subtype, enabling exhaustive pattern matching without a default case. This is the modern way to do algebraic data types in Java.

## Stage 3 — Collections, Generics, Exceptions

### Collections

```java
import java.util.*;

List<String> list = new ArrayList<>();   // dynamic array, O(1) random access
list.add("a"); list.add("b"); list.get(0);
List<Integer> immutable = List.of(1, 2, 3);  // unmodifiable (9+)
List<Integer> copy = List.copyOf(list);     // unmodifiable copy

Set<Integer> set = new HashSet<>();         // hash set, O(1) contains
Map<String, Integer> map = new HashMap<>();
map.put("a", 1); map.get("a");             // returns null if absent
map.getOrDefault("missing", 0);

Queue<Integer> q = new ArrayDeque<>();     // FIFO/LIFO
Deque<Integer> stack = new ArrayDeque<>(); // use as stack with push/pop

// Iterate
for (String s : list) { }
list.forEach(System.out::println);
map.forEach((k, v) -> System.out.println(k + "=" + v));
```

Choose: **`ArrayList`** for indexed access, **`LinkedList`** rarely (cache-unfriendly), **`HashSet`/`HashMap`** for O(1) lookup, **`TreeSet`/`TreeMap`** for ordered, **`ArrayDeque`** for stack/queue (don't use `Stack`/`Vector` — legacy and synchronized).

### Generics and wildcards

```java
// Generic method
public static <T> T first(List<T> xs) { return xs.get(0); }

// Generic class
public class Box<T> { private final T value; public Box(T v) { value = v; } public T get() { return value; } }
Box<String> b = new Box<>("hi");

// Bounded type parameter
public static <T extends Comparable<T>> T max(T a, T b) { return a.compareTo(b) >= 0 ? a : b; }

// Wildcards
void printAll(List<?> xs) { for (Object o : xs) System.out.println(o); }   // ? = any type (consumer)
void addAll(List<? super Integer> xs) { xs.add(1); }                        // ? super T = lower bound (write)
Number n = first(List<? extends Number> ...);                              // ? extends T = upper bound (read)
```

The PECS rule: **Producer `extends`, Consumer `super`**. If you read from a collection, `<? extends T>`; if you write, `<? super T>`. Generics are erased at runtime (`List<String>` and `List<Integer>` are both `List` at runtime) — you can't do `new T()` or `instanceof List<String>`.

### Exceptions

```java
// Checked (must catch or declare) - for recoverable conditions
class ConfigError extends Exception { public ConfigError(String m) { super(m); } }
// Unchecked (runtime) - for programmer errors
class BadInput extends RuntimeException { public BadInput(String m) { super(m); } }

void load() throws ConfigError {              // declared with throws
    try (var reader = Files.newBufferedReader(path)) {   // try-with-resources (AutoCloseable)
        String line = reader.readLine();
        if (line == null) throw new ConfigError("empty");
    } catch (IOException e) {
        throw new ConfigError("read failed: " + e.getMessage());  // wrap
    } finally {
        // always runs (avoid - prefer try-with-resources)
    }
}

// Multi-catch
try { ... } catch (IOException | SQLException e) { ... }
```

**Checked exceptions** (`extends Exception`) force callers to handle or declare — use for recoverable conditions (file not found, network down). **Unchecked** (`extends RuntimeException`) propagate silently — use for programmer errors (null, bad index, invalid state). The debate over checked exceptions is eternal; in practice, prefer unchecked for application code and reserve checked for genuinely recoverable library conditions.

## Stage 4 — Modern Java (8–21+)

![Modern Java Features](/assets/img/diagrams/java-tutorial/java-features.svg)

### Lambdas and method references

```java
// Functional interface (single abstract method)
Runnable r = () -> System.out.println("hi");
Comparator<String> byLen = (a, b) -> a.length() - b.length();
Function<Integer, Integer> sq = x -> x * x;
Predicate<String> nonEmpty = s -> !s.isEmpty();
Supplier<List<String>> factory = ArrayList::new;    // method reference / constructor

// Method references
list.forEach(System.out::println);                  // instance method
String::toUpperCase;                                 // on arbitrary instance
Integer::parseInt;                                    // static
```

A lambda is a concise instance of a **functional interface** (one abstract method). The common ones live in `java.util.function`: `Function<T,R>`, `Predicate<T>`, `Supplier<T>`, `Consumer<T>`, `BiFunction<T,U,R>`.

### Streams

```java
import java.util.stream.*;

List<Integer> nums = List.of(1, 2, 3, 4, 5);

int sum = nums.stream()
    .filter(n -> n % 2 == 0)
    .mapToInt(Integer::intValue)
    .sum();                                            // 6

List<String> names = users.stream()
    .map(User::name)
    .sorted()
    .collect(Collectors.toList());                    // terminal op

Map<String, List<User>> byCity = users.stream()
    .collect(Collectors.groupingBy(User::city));     // grouping

// Reduce
int product = nums.stream().reduce(1, (a, b) -> a * b);

// Parallel (use sparingly - has overhead)
nums.parallelStream().mapToInt(Integer::intValue).sum();
```

Streams are **lazy**: intermediate ops (`map`, `filter`, `sorted`) build a pipeline; nothing runs until a **terminal** op (`collect`, `forEach`, `reduce`, `count`). This lets you express "filter then transform then collect" as one readable chain. Don't abuse `parallelStream` — it has overhead and only helps CPU-bound, large, order-independent pipelines.

### Optional — null safety

```java
Optional<String> find(int id) { ... }   // explicit "may be absent"

find(1).ifPresent(System.out::println);
String name = find(1).orElse("default");
String n = find(1).orElseThrow(() -> new RuntimeException("not found"));
find(1).map(String::toUpperCase).orElse("NONE");

// Avoid: Optional fields, Optional as parameter. Use it for return types.
```

`Optional<T>` is a box that forces you to handle absence explicitly. Use it for **return types** of methods that might not find a value; don't use it for fields or parameters (it adds allocation for no benefit there).

### Records, pattern matching, var

```java
// Records (16+) - immutable data carriers; auto ctor/eq/hash/toString
public record Point(int x, int y) {}
Point p = new Point(1, 2);
p.x(); p.y();                   // accessor methods

// instanceof pattern matching (16+)
if (obj instanceof String s) { s.toUpperCase(); }    // s bound and typed

// switch pattern matching (21+) - with sealed types, exhaustive
String describe(Shape s) {
    return switch (s) {
        case Circle c -> "circle r=" + c.r();
        case Square sq -> "square s=" + sq.s();
    };
}

var nums = List.of(1, 2, 3);    // var - local type inference (10+), not for fields/params

String json = """
    {
      "name": "Ada",
      "age": 30
    }
    """;                          // text block (15+) - preserves indentation
```

Records are the modern way to write data classes — one line replaces a constructor, accessors, `equals`, `hashCode`, `toString`. Pattern matching (especially on sealed types) replaces verbose `instanceof` casts with concise, exhaustive switches.

## Stage 5 — Concurrency and the JVM

![Java Concurrency](/assets/img/diagrams/java-tutorial/java-concurrency.svg)

### Threads and ExecutorService

```java
// Classic threads (1:1 OS threads)
Thread t = new Thread(() -> System.out.println("running"));
t.start(); t.join();

// ExecutorService - pooled threads (java.util.concurrent)
ExecutorService pool = Executors.newFixedThreadPool(8);
Future<Integer> f = pool.submit(() -> compute());
Integer result = f.get();          // blocks
pool.shutdown();

// Thread-safe collections and atomics
var counters = new java.util.concurrent.ConcurrentHashMap<String, Integer>();
AtomicInteger n = new AtomicInteger(0);
n.incrementAndGet();
```

### CompletableFuture — async pipelines

```java
CompletableFuture<Integer> f = CompletableFuture
    .supplyAsync(() -> fetch("url"))           // async, on ForkJoinPool
    .thenApply(String::length)                  // transform
    .thenCompose(len -> CompletableFuture.supplyAsync(() -> compute(len)))  // chain
    .exceptionally(e -> -1);                    // handle error

int r = f.join();                               // block for result
CompletableFuture<Void> all = CompletableFuture.allOf(f1, f2, f3);
CompletableFuture<Object> first = CompletableFuture.anyOf(f1, f2);
```

`CompletableFuture` is Java's composable async primitive — `thenApply` (map), `thenCompose` (flatMap), `thenCombine` (zip), `allOf`/`anyOf` (fan-in). It's the `Promise` equivalent, more verbose than JS but with richer composition.

### Virtual threads (21+) — Project Loom

```java
// OS threads are expensive (~1MB stack). Virtual threads are cheap (~KB).
// One JVM can run millions of virtual threads.

try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
    IntStream.range(0, 100_000).forEach(i ->
        exec.submit(() -> {
            Thread.sleep(Duration.ofSeconds(1));
            return i;
        }));
}  // runs 100k tasks concurrently with tiny memory

// Or directly
Thread.startVirtualThread(() -> handleConnection(socket));
```

Virtual threads are the big Java 21 feature: cheap, user-mode threads for I/O-bound work. You write blocking code (the readable kind) and the runtime parks the virtual thread when it blocks on I/O, freeing the carrier OS thread. This makes "thread-per-request" servers scale like async frameworks without the async code complexity. For CPU-bound work, keep using `ForkJoinPool`/`parallelStream`.

### The JVM

![The JVM](/assets/img/diagrams/java-tutorial/java-jvm.svg)

```bash
javac Main.java           # .java -> .class bytecode (platform-independent)
java Main                 # JVM loads class, interprets/JITs bytecode
java -jar app.jar         # packaged
java --module-path mods -m my.module/main.Main   # modules
```

When you run a class:

1. **`javac`** compiles `.java` to `.class` bytecode (stack-based, platform-neutral).
2. **Classloader** loads bytecode: bootstrap (JDK core) → platform (extension) → application (your code). Java 9+ modules (JPMS) make dependencies explicit in `module-info.java`.
3. **JVM** verifies bytecode (type safety), links, initializes, then interprets it. The **JIT compiler** (C1 for fast startup, C2 for peak throughput; GraalVM as alternative) compiles hot methods to native code at runtime.
4. **Memory**: heap (objects, split into young/old generations), stack (frames per method call), metaspace (class metadata).
5. **Garbage collector**: G1 (default, balanced), ZGC / Shenandoah (low-pause, sub-ms), Serial (small heaps). Modern GCs are concurrent and pause times are tiny.

The result: "write once, run anywhere" — the same `.class` runs on any OS with a JVM. The JVM's maturity (decades of GC, JIT, profiling tooling like JFR, async-profiler) is a major reason Java dominates enterprise backends.

## The Toolchain

```bash
# Build tools
javac Main.java                   # direct compile
mvn package                       # Maven (XML, dependency mgmt)
gradle build                       # Gradle (Groovy/Kotlin DSL)
# Modern alternatives: Maven with mvnd (daemon), Gradle, or Bazel for monorepos

# Run
java -jar target/app.jar
java --enable-preview -jar app.jar # preview features

# Test
mvn test                           # JUnit 5
gradle test

# Package
jar cf app.jar -C classes .        # raw jar
mvn package                        # fat jar (shade plugin)

# Inspect
jdeps --list-deps app.jar          # module/dep analysis
jcmd <pid> Thread.print             # thread dump
jcmd <pid> GC.heap_info             # heap info
java -XX:+PrintFlagsFinal -version  # JVM tuning flags
```

A minimal `pom.xml` (Maven):

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>myapp</artifactId>
  <version>1.0.0</version>
  <properties><maven.compiler.release>21</maven.compiler.release></properties>
  <dependencies>
    <dependency><groupId>org.junit.jupiter</groupId><artifactId>junit-jupiter</artifactId><version>5.10.0</version><scope>test</scope></dependency>
  </dependencies>
</project>
```

**Essential tooling:**

- **Maven / Gradle** — build, dependency management, packaging. Maven is XML-verbose but ubiquitous; Gradle is scriptable.
- **JUnit 5** — the test framework; AssertJ for fluent assertions.
- **JFR (Java Flight Recorder) + async-profiler** — low-overhead profiling built into the JVM.
- **`jcmd` / `jstack` / `jmap`** — runtime inspection (thread dumps, heap).
- **Lombok** (controversial) — generates boilerplate; records often replace it now.
- **Spring Boot** — the dominant application framework (web, DI, data).

## A Quick-Start Checklist

1. **Install JDK 21 LTS** (or 22+ for newest features).
2. **Use Maven or Gradle** for any non-trivial project; never hand-manage classpath.
3. **Make fields `private`; expose via methods or records.**
4. **Prefer interfaces over abstract classes; use records for data carriers.**
5. **Use streams for collection transforms** — but keep them readable; break to a loop if it wraps.
6. **Return `Optional`** for "may be absent" results; never return null for absent.
7. **Use `try-with-resources`** for anything `AutoCloseable` (files, connections).
8. **Virtual threads for I/O-bound concurrency** (21+); `ForkJoinPool`/`parallelStream` for CPU.
9. **Run JFR in production** for profiling; enable GC logging.

## Common Pitfalls

- **`==` on objects** — compares references, not contents. Use `.equals()`. (`"a" == "a"` may work due to string interning, but don't rely on it.)
- **Autoboxing in loops** — `Integer` arithmetic allocates. Use primitives (`int`) in hot loops.
- **Mutating a collection while iterating** — `ConcurrentModificationException`. Use iterator's `remove()` or a stream to build a new list.
- **Checked exceptions in lambdas** — streams/`CompletableFuture` only accept unchecked. Wrap with a runtime exception.
- **`HashMap` with a mutable key** — mutating a key after insertion breaks lookup. Use immutable keys.
- **`finalize()` is deprecated** — use `Cleaner` or try-with-resources; finalization has no timeliness guarantees.
- **Shared mutable state** — the classic concurrency bug. Prefer immutability, `var`/`final`, thread-safe collections, or virtual threads with no sharing.
- **`ThreadLocal` leaks** — remove in `finally` or they accumulate in thread pools.
- **Date/time: avoid `Date`/`Calendar`** — use `java.time` (`Instant`, `LocalDate`, `ZonedDateTime`, `Duration`).

## What to Learn Next

- **The Java Tutorials** — [docs.oracle.com/javase/tutorial](https://docs.oracle.com/javase/tutorial/) official, thorough.
- **Effective Java** by Joshua Bloch — the canonical "write Java well" book; every Java dev should read it.
- **Modern Java in Action** — streams, lambdas, modern features in depth.
- **Java Concurrency in Practice** by Brian Goetz — the bible on concurrency (pre-virtual-threads but the principles hold).
- **Project Loom docs** — [openjdk.org/projects/loom](https://openjdk.org/projects/loom/) for virtual threads.
- **JEP index** — [openjdk.org/jeps](https://openjdk.org/jeps/) every enhancement proposal, the source of truth for new features.
- **Spring Boot guides** — [spring.io/guides](https://spring.io/guides) for the dominant app framework.
- **Baeldung** — [baeldung.com](https://www.baeldung.com/) practical Java/Spring recipes.

Java's verbosity is real but shrinking — records, `var`, pattern matching, and virtual threads have made modern Java genuinely pleasant. The JVM's maturity (GC, JIT, profiling, the library ecosystem) is the real moat: once your code runs on it, decades of optimization work for free.

Good luck — and read *Effective Java*.

**Resources:**

- Oracle docs: [https://docs.oracle.com/javase/](https://docs.oracle.com/javase/)
- JEP index: [https://openjdk.org/jeps/](https://openjdk.org/jeps/)
- Maven: [https://maven.apache.org/](https://maven.apache.org/)
- Spring: [https://spring.io/](https://spring.io/)
- JUnit: [https://junit.org/](https://junit.org/)