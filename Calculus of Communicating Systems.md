_Concurrent System_: A set of processes, _executing_ in _parallel_, that _interact_ between each other.

### Inter-Process Communication
There are multiple communication paradigms such as:

1. _Ether_: ^6a313f
	1. _send_ is always possible if the medium is unbounded, or possible only if there is space (bounded).
	2. _recv_ is possible only if there is a message available, and the read operation is **destructive** (i.e. removes the read message from the medium).
	3. There are _no order guarantees_.
2. _Buffer_:
	1. _send_ is identical to [[#^6a313f]]
	2. _recv_ is identical to [[#^6a313f]]
	3. Order of messages is _preserved_
3. _Shared Memory_:
	1. _send_ corresponds to writing to the memory area. Compared to before writes are **destructive**. Writes are always possible.
	2. _recv_ reads the corresponding memory area. Reads are not destructive and can be always done.
	3. There are _no order guarantees_.

These different mechanisms could lead to different paradigms for concurrent processes, but we _DO NOT_ want this, since we want to build a unifying theory for this.

_Idea_: **everything is a PROCESS** $\implies$ both active entities (agents) and passive (communication mediums) are modeled as _processes_, that communicate via _synchronous interaction_ (handshake, rendezvous, ...).

### Syntax of CCS Programs

In CCS a process is represented as a block with a name and a set of ports that consitutes its interface.

![[images/ccs-process-exa.png]]

In this example we see the modelization of a Computer Scientist process, that has three ports: coin to pay the coffee, pub to send publications and coffee to receive it.

#### CCS Constructs

^77dc9f

There are different CCS constructs needed to represent the behaviour of a program, which include its structure and the interactions between components.

- **Inaction**: $0$, the process is unable to proceed or do anything (deadlocked)
- **Action prefixing**: Given a channel $\alpha$ and a process $P$, we can construct a new process by prefixing an operation over the channel and the process: $\alpha.P$ (receive from $\alpha$ and do $P$), or $\bar{\alpha}.P$ (send to $\alpha$ and do $P$).
- **Process constants**: We can "label" pieces of code for instance: $\text{Break} \equiv \overline{\text{coin}}.\text{coffee}.0$. This is very powerful since it is the only way to express (unbounded) recursion: $\text{Clock} \equiv \overline{tick}.\text{Clock}$.
- **Non-deterministic choice**: Given processes $P$ and $Q$, one can construct $P + Q$, which signifies "choose one branch arbitrarily from the processes _ready to proceed_". $$
\begin{align}
&\text{CTM} \equiv \text{coin}.(\overline{\text{coffee}}.\text{CTM} + \overline{\text{tea}}.\text{CTM}) \\
&\text{CTM}' \equiv \text{coin}.\overline{\text{coffee}}.\text{CTM}' + \text{coin}.\overline{\text{tea}}.\text{CTM}'\\
\end{align}$$ Are NOT equivalent, because in the former the selection is made _after_ getting the coin, while the latter makes the decision _before_ it takes the coin. 
![[ccs-nondet-exa.png|center|300]]
The _ready to proceed_ requirement is important: let's suppose we want to represent a broken clock that surely emits one tick and then could stop at any step: $$
\begin{align}
&\text{BC} = \overline{\text{tick}}.(\text{BC} + 0) \\
&\text{BC} = \overline{\text{tick}}.\text{BC} + \overline{\text{tick}}.0
\end{align}
$$ The first specification is wrong, since the operator + only picks processes ready to proceed, but 0, by definition, is the stuck program; so it would actually behave as a normal clock (i.e. for any process $P, P \sim (P + 0)$).
- **Parallel Composition**: Given two processes $P, Q$, then $P | Q$ represents the notion of processes running _concurrently_ (possibly in parallel). Notice that communication channels are _public_, which means there may be multiple processes still communicating with the specified ones _outside_ of the definition, as shown in the image. ![[ccs-parallel-comp-exa.png]]
- **Restriction**: to solve the problem above, we introduce this construct that makes the restricted channels local to the composition, so they are not accessible from the outside, but only by the composed processes. ![[ccs-restriction-exa.png]]
- **Relabeling**: a common thing useful to perform is a way to generalize and specialize processes. For instance: $$
\begin{align}
&\text{VM} = \text{coin}.\overline{\text{item}}.\text{VM} \\
&\text{CHOC} = \text{VM}[\text{choc} | \text{item}]\\
&\text{CHIPS} = \text{VM}[\text{chips} | \text{item}]\\
\end{align}
$$ We have defined a generic vending machine and specialized it. In practice this means redirecting the item channel to the choc or chips channel, leaving the others untouched. 

![[ccs-relabeling-exa.png|center|400]]

#### The behaviour of processes

The idea behind the semantics of CCS is that processes will evolve over time, performing state transitions, that are determined by communications over channels. Notice that there is no distinction between states and processes: when a process undergoes a state transition it becomes a "new" process with a "new" behaviour, state is embedded in the process.

The following example shows the operation of a Computer Scientist process with a Coffee Machine, composed in parallel:
![[ccs-system-exa.png]]

The problem that now arises is: how do we define and describe the evolution of a system composed by multiple processes to an external observer? As shown below, some transitions happen in response to  interactions between processes _internally_ to the composed system, thus not observable externally. In this case we denote these transactions as _invisible_ or _silent_. ![[ccs-invisible-exa.png]]

Let's now consider a "semi-closed" system of an office, composed by a CS and a CM, the coin and coffee channels are restricted to the inside, the only publicly visible is the pub channel:$$\text{Office} = (\text{CS}|\text{CM}) \setminus \{ \text{coin}, \text{coffee} \}$$


So the only way for an external observer to evaluate its behaviour is to look at the pub channel.

This is crucially important as, in order to evaluate the _correctness_ of a system, the designer has to write a specification, and prove its equivalence to the system under test: ![[ccs-equiv-exa.png]]

#### Formal Syntax

##### Channels and Actions
Let's first formally define ports (channels): assume a _countably infinite_ collection of _channel names_
$$
\mathcal{A} = \{ a: a \text{ is a channel name} \}
$$

Then we can define the set of _complimentary channel names_. 
$$
\bar{\mathcal{A}} = \{ \bar{a} : a \in \mathcal{A} \}
$$

Recall that we use channel names for input actions and complimentary names for output actions.

We let the set of _labels_ $\mathcal{L}$ as the union of all channels name in both directions:
$$
\mathcal{L} = \mathcal{A} \cup \bar{\mathcal{A}}
$$

Then we can define the set of all possible actions as the union of all named channels and the _hidden_ actions:
$$
Act = \mathcal{L} \cup \{ \tau \}
$$

##### Processes

Let's define the set of process constants, which can be intuitively thought as the names for the processes. Assume it to be _countably infinite_ so we never run out of names.
$$
\mathcal{K} = \{ K : K \text{ is a process name} \} $$

After all these definitions we can formally define the grammar for a CCS expression:

$$
P,Q ::= K\quad|\quad \alpha.P\quad|\quad \sum_{i \in I}{P_{i}}\quad|\quad P | Q\quad |\quad P[f]\quad | P \setminus L
$$
where:
- $K$ is a _process name_ in $\mathcal{K}$
- $\alpha$ is an _action_ in $Act$
- $I$ is a possibly infinite _index set_
- $f: Act \to Act$ is a _relabelling function_ satisfying the constraints:
$$
\begin{align}
&f(\tau) = \tau \\
&f(a) = b \implies f(\bar{a}) = \bar{b} \text{ (it has to be bi-directional)}
\end{align}
$$
- $L$ is a subset of labels $L \subseteq \mathcal{L}$

Let's now show some typical patterns that allows us to implement operations discussed informally [[#^77dc9f|here]] using formal constructs:

- **Inaction**:$$
0 \sim \sum_{i \in \emptyset}P_{i} $$
- **Non-deterministic choice**:$$
P_{1} + P_{2} \sim P_{2} + P_{1} \sim \sum_{i \in \{ 1, 2 \}}{P_{i}}$$
- **Relabelling**:$$
\begin{array}{c c c}
a_{1} & \dots & a_{n} \\
\downarrow & \dots & \downarrow \\
b_{1} & \dots & b_{n} \\
\end{array}
\implies P\left[ \frac{b_{1}}{a_{1}}, \dots, \frac{b_{n}}{a_{n}} \right] \sim P[f] \text{ where } f(action) := 
\begin{cases}
f(\tau) = \tau \\
f(a_{i}) = b_{i} \\
f(x) = x\ \forall x \neq a_{i}, i = 1,\dots,n
\end{cases}$$
- **Restriction**:$$
P \setminus \{ a \} \sim P \setminus a \text{ (for conciseness)}$$
- **Priority of operations** (highest to lowest):
	1. $\setminus$ (restriction)
	2. $[f]$ (relabeling)
	3. $\alpha$ (action)
	4. $|$ (parallel composition)
	5. $+$ (non-deterministic choice)

#### Operational Behaviour

Now we want to be able to express formally the evolution of a system of processes over the execution of actions via _syntax-driven rules_. For this we need some rules to capture the semantics of the language. These rules will be used to automatically generate a _transition system_ whose states are CCS expressions (i.e. processes).

Note in the following notation the expressions above the bar represent the preconditions, while the expression below represents the actual transition.

##### ACT
The ACT rule is the simplest of all, it represents the _axiom_ that any process in the form $\alpha.P$ _affords_ the transition $\alpha.P \xrightarrow{\alpha} P$. Being axiomatic, there are no _subgoals_ to prove. The syntax of the ACT rule is:
$$
\frac{}{\alpha.P \xrightarrow{\alpha} P}
$$
Notice there are no pre-assumptions above the horizontal bar.

##### SUM
The SUM rule is applied when we encounter a non-deterministic choice between multiple branches. We pre-determine the branch to follow and then express the transition that would happen if that branch would be chosen, repeating the operation for all the possible branches.
$$
\frac{P_{j} \xrightarrow{\alpha} P'_{j}}{\sum_{i\in I}P_{i} \xrightarrow{\alpha}P'_{j} } \text{ for some } j \in I
$$

For example the binary non-deterministic choice would become:
$$
\begin{align}
\frac{P_{1} \xrightarrow{\alpha} P'_{1}}{P_{1} + P_{2} \xrightarrow{\alpha}P'_{1} } &\quad \text{for }j = 1 \\
\frac{P_{2} \xrightarrow{\alpha} P'_{2}}{P_{1} + P_{2} \xrightarrow{\alpha}P'_{2} } &\quad \text{for } j = 2 \\
\end{align}
$$

##### COM

Given the parallel composition of two processes $P, Q$ we can obtain the following transitions, depending on who executes its action first:
$$
\begin{align}
\frac{P \xrightarrow{\alpha} P'}{P | Q \xrightarrow{\alpha}P'|Q} &\quad \text{if } P \text{ transitions first} \\
\frac{Q \xrightarrow{\alpha} Q'}{P | Q \xrightarrow{\alpha}P|Q'} &\quad \text{if } Q \text{ transitions first} \\
\end{align}
$$

Additionally we can express the synchronization of $P$ and $Q$ over a channel $a$:
$$
\frac{P \xrightarrow{a} P' \quad Q \xrightarrow{\bar{a}} Q'}{P|Q \xrightarrow{\tau}P'|Q'}
$$

##### RES

Given a restriction over some actions we can express transactions happening over the non-restricted channels:
$$
\frac{P \xrightarrow{\alpha} P'}{P \setminus L \xrightarrow{\alpha} P'\setminus L} \text{where } \alpha, \bar{\alpha} \not\in L, L \subseteq \mathcal{L}
$$
Note that the restriction is still in place after the transition.

##### REL

We can express the redirection of operations, given a redirection function over a process:
$$
\frac{P \xrightarrow{\alpha} P'}{P[f] \xrightarrow{f(\alpha)} P'[f]}
$$

##### CON

By extending the ACT notion we can define transitions over named processes, given they are defined as the process subject to the transition:
$$
\frac{P \xrightarrow{\alpha} P'}{K \xrightarrow{\alpha} P'} \text{where } K := P
$$

All these rules will allow us to write an **interpreter** for CCS systems by pattern matching with the rules and decomposing them recursively until the axioms are obtained. The result of the interpreter will be the complete _transition system_.

### Value Passing CCS

Up until now we have only seen pure (synchronization) CCS, which means that processes can interact and synchronize only by pure message passing.

For a greater convenience we can also define a theory in which processes can have internal variables and messages can carry data. This is what is actually implemented in practice in programming languages.

This theory is called _value passing_ CCS, we will take a brief look at it for its practical implementations. However, since it is not more expressive then normal CCS (i.e. it cannot represent anything more that normal CCS), it is not actually used in theory, because it brings more complexity without actual theoretical use.

**Example**:

Consider this incrementing buffer example, that takes a value $n \in \mathbb{N}$ in input and outputs the value stored incremented by one:

With value passing we can express it as:
$$
\begin{align}
&B := in(x).B'(x) \\
&B'(x) := \overline{out}(x+1).B
\end{align}
$$
a much more convenient notation.

Notice that this theory only allows us to work with _closed_ programs, that is programs where variables in the system are bound to some value or binding structure (a function parameter or expression) and there are not free variables. This means that program execution is completely self-contained and deterministic and it doesn't depend on external state.
We also will assume variables are in $\mathbb{N}$.

#### Syntax

Let:
- $x, y, \dots \in Var$ be the variables for values
- $a, b, c \in \mathcal{A}$ be the channels
- $k(x_{1}, \dots, x_{n}) \in \mathcal{K}$ be the process constants
- $e \in Expr$ be the set of arithmetic or boolean expressions:
$$
\begin{align}
&e := k\ |\ e + e\ |\ e \cdot e\ |\ \dots \\
&b := e = e\ |\ e \leq e\ |\ \neg b\ |\ b \land b\ | \dots
\end{align}
$$

Then the formal syntax for CCS processes is:
$$
P, Q := k(e_{1}, \dots, e_{n})\ |\ a(x).P\ |\ \bar{a}(x).P\ |\ \tau.P\ |\ \text{if } b \text{ then } P\ |\ \sum_{i \in I}P_{i} \ |\ P|Q\ |\ P\setminus L\ |\ P[f]
$$
with many contructs following normal CCS.

#### Behaviour

##### ACT
We now have to distinguish between input and output for action prefixing:
- **Input**:
$$
\frac{}{a(x).P \xrightarrow{{a}(n)} P\left[ \frac{n}{x} \right]} \text{ for } n\geq 0
$$
This means the process is willing to accept an actual value $n$ on the $a$ channel and bind it to its $x$ variable (i.e. every free occurrence of $x$ is replaced by $n$).
- **Output**
$$
\frac{}{\bar{a}(e).P \xrightarrow{\bar{a}(n)} P} \text{ where } n \text{ is the result of evaluating }e
$$
This means the process has an internal expression $e$ bound to its internal variable, and it is willing to send the value of the expression evaluated.

- **Silent** (synchronization) interaction remains the same

##### CON
Let's generalize the definition for parametrizing process constants by value variables:
$$
\frac{P\left[ \frac{v_{1}}{x_{1}},\dots, \frac{v}{x_{n}} \right] \xrightarrow{\alpha} P'}{A(e_{1}, \dots, e_{n}) \xrightarrow{\alpha} P'} \text{ assuming } A(e_{1}, \dots, e_{n}) := P \text{ and each } e_{i} \text{ has value } v_{i}
$$

In simpler terms: To figure out what a parameterized process can do, plug in the actual values of its parameters, see what actions the resulting process can take, and those are the actions the parameterized process can take.

This rule is crucial because it connects the abstract definition of a parameterized process with its concrete behavior when specific values are provided.

##### COND

Since processes in value-passing CCS can manipulate data, it is natural to add an 'if then else' construct to the language. Formally:
$$
\begin{cases}
\frac{P \xrightarrow{\alpha} P'}{\textbf{if} \text{ bexp } \textbf{then } P  \textbf{ else } Q \xrightarrow{\alpha} P'} & \text{ if bexp evaluates to true} \\
\frac{Q \xrightarrow{\alpha} Q'}{\textbf{if} \text{ bexp } \textbf{then } P  \textbf{ else } Q \xrightarrow{\alpha} P'} & \text{ if bexp evaluates to false} \\
\end{cases}
$$

Notice that the 'if then else' construct is equivalent to the non-deterministic choice between two 'if then' constructs:
$$
\text{if } b \text{ then } P \text{ else } Q \equiv \text{if } b \text{ then } P + \text{ if } \neg b \text{ then } Q
$$
It may appear that parallel composition would be semantically equivalent, but parallel composition would spawn a "garbage" process that would never be used and stay there forever.

### Encoding Value-Passing CCS into Pure CCS

Define an encoding function $[[\quad]]: \text{CCS-VP} \to \text{CCS}$. It is possible to implement it by following the conversion rules:

- $[[a(x).P]] \to \sum_{n \in \mathbb{N}}{a_{n}.\left[ \left[ P\{\frac{n}{x}\} \right] \right]}$
- $[[\bar{a}(e).P]] \to \bar{a}_{n}.[[P]]$ if $e$ evaluates to $n$.
- $[[\tau.P]] \to \tau.[[P]]$
- $\left[ \left[ \sum_{i \in \mathbb{N}}{P_{i}}\right] \right] \to \sum_{i\in \mathbb{N}}{[[P_{i}]]}$
- $[[P|Q]] \to [[P]] | [[Q]]$
- $[[P\setminus L]] \to [[P]] \setminus \{ a_{n}:a \in L \}$
- $[[P[f]]] \to [[P]][f']$ where $f'(a_{n}) = f(a)_{n}$
- $[[\text{if } b \text{ then } P ]] \mapsto \begin{cases}
	[[P]] & b = True \\
	0 & b = False
\end{cases}$
- $[[K(e_{1}, \dots, e_{n})]] \to K_{n_{1}, \dots, n_{n}}$

**Theorem**:
> Let $[[\quad]]: \text{CCS-VP} \to \text{CCS}$ as above.
> Then, for all CCS-VP programs P
> 1. if $P \xrightarrow{\alpha} P'$ then $[[P]] \xrightarrow{\hat{\alpha}} [[Q]]$
> 2. if $[[P]] \xrightarrow{\hat{\alpha}} Q$ then there is $P'$ such that $P \xrightarrow{\hat{\alpha}} P'$ and $[[P']] = Q$
> Where:
> $$
\hat{\alpha} = \begin{cases}
a_{n} & \text{if } \alpha = a(n) \\
\overline{a_{n}} & \text{if } \alpha = \bar{a}(n) \\
\tau & \text{if } \alpha = \tau
\end{cases}$$

This theorem shows that properties are mantained by the translations, that is CCS-VP and pure CCS are semantically equivalent.

### Introduction to Behavioral Equivalence

In the Calculus of Communicating Systems (CCS), behavioral equivalence allows us to determine when two processes behave in the same way. This is crucial for verifying that an implementation meets its specification, or for substituting one component with another.

Consider the following example: $$ \begin{align} CS &= \overline{pub}.\overline{coin}.coffee.CS \\ CM &= coin.coffee.CM \\ Office &= (CS \mid CM) \setminus {coin, coffee} \end{align} $$

And the specification: $$Spec = \overline{pub}.Spec$$

We want to verify: $Office \sim Spec$

Where $\sim$ represents some notion of behavioral equivalence.

#### Properties of Behavioral Equivalence

An equivalence relation $\sim$ should satisfy these properties:

1. **Reflexive**: $P \sim P$
2. **Symmetric**: If $P \sim Q$ then $Q \sim P$
3. **Transitive**: If $P \sim Q$ and $Q \sim R$ then $P \sim R$

Additionally, for system composition, we need:

1. **Congruence/Compositionality**: If $P \sim Q$, then for every context $C[\cdot]$, $C[P] \sim C[Q]$

This last property is essential because it allows us to replace equivalent components within a larger system.

For example, if: $$ \begin{align} Spec &= Spec_1 \mid Spec_2  \\
Sys_1 &\sim Spec_1  \\
 Sys_2 &\sim Spec_2 \end{align} $$

Then compositionality ensures: $Sys_1 \mid Sys_2 \sim Spec_1 \mid Spec_2$

### Referential Transparency

Referential transparency means that replacing an expression with its value doesn't alter the behavior of the program:

$$P = \ldots exp \ldots$$

Replacing $exp$ by its value should not change $P$'s behavior.

For example, in mathematics: $$ \begin{align} x + 0y &= 2 \\ x + x &= 1 \end{align} $$

We can deduce $x = \frac{2}{3}$ and $y = -1$, and substituting these values preserves the equations.

### Observational Equivalence

Behavioral equivalence depends solely on observable behavior. Different notions of equivalence depend on what we consider observable:

- Messages
- Time
- Cost
- etc.

###  Same Transitions (Basic Equivalence)

This is the simplest form, where processes with the same transitions are considered equivalent.

Example: $$ \begin{align} A &= a.0 \\ B &= (a.0) + b \end{align} $$

When we only observe transitions, $A \sim B$ because both can perform an 'a' action.

However, this equivalence is not compositional. In the context $C[\cdot] = A + [\cdot]$: $$ \begin{align} C[A] &= A + A \\ C[B] &= A + B \end{align} $$

$C[A]$ and $C[B]$ no longer have the same transitions, as $C[B]$ can also perform a 'b' action.

### Trace Equivalence

Trace equivalence considers the sequences of actions a process can perform:

$$Tr(P) = {\alpha_1\ldots\alpha_n \mid P \stackrel{\alpha_1}{\longrightarrow} P_1 \stackrel{\alpha_2}{\longrightarrow} P_2 \ldots \stackrel{\alpha_n}{\longrightarrow} P_n}$$

$P \sim_T Q$ if $Tr(P) = Tr(Q)$

Trace equivalence is:

- An equivalence relation
- Based on observable behavior
- Compositional

Example: $$ \begin{align} CTM &= coin.(coffee.CTM + tea.CTM)\\CTM' &= coin.coffee.CTM' + coin.tea.CTM' \end{align} $$

These are trace equivalent: $Tr(CTM) = Tr(CTM') = (coin \cdot (coffee + tea))^* \cdot coin$

In the context of the Office system: $$ \begin{align} Office &= (CS \mid CTM) \setminus {coin, coffee, tea}\\Office' &= (CS \mid CTM') \setminus {coin, coffee, tea} \end{align} $$

Trace equivalence guarantees: $Office \sim_T Office'$

### Completed Trace Equivalence

Trace equivalence doesn't distinguish between deadlock states. Completed trace equivalence addresses this:

$$CTr(P) = {\alpha_1\ldots\alpha_n \mid P \stackrel{\alpha_1}{\longrightarrow} P_1 \stackrel{\alpha_2}{\longrightarrow} P_2 \ldots \stackrel{\alpha_n}{\longrightarrow} P_n \not\rightarrow}$$

Where $P_n \not\rightarrow$ means $P_n$ cannot perform any action (deadlock).

$P \sim_{CT} Q$ if $P \sim_T Q$ and $CTr(P) = CTr(Q)$

For our example: $$CTr(Office) \neq CTr(Office')$$

Therefore: $Office \not\sim_{CT} Office'$

Importantly, completed trace equivalence is **not compositional**.

### Bisimilarity

Bisimilarity is a stronger equivalence that considers intermediate states:

**Definition**: A binary relation $\mathrm{Re}l \subseteq Proc \times Proc$ is a bisimulation if for all $P,Q \in Proc$ with $P\ \mathrm{Re}l\  Q$:

- For all $P \stackrel{\alpha}{\longrightarrow} P'$, there exists $Q \stackrel{\alpha}{\longrightarrow} Q'$ and $P'\ \mathrm{Re}l\  Q'$
- For all $Q \stackrel{\alpha}{\longrightarrow} Q'$, there exists $P \stackrel{\alpha}{\longrightarrow} P'$ and $P' \ \mathrm{Re}l\ Q'$

$P \sim Q$ if there exists a bisimulation $R$ such that $P \ \mathrm{Re}l\ Q$

Example with infinite states: $$ \begin{align} A &= a.A\\B &= (a.B) + b \end{align} $$

There exists a bisimulation $\mathrm{Re}l = {(A, ((B \cdot b)^k \cdot b) \mid k \in \mathbb{N})}$ showing that $A \sim B$.

For our coffee machine example: $$ \begin{align} CTM &= coin.(coffee.CTM + tea.CTM)\\CTM' &= coin.coffee.CTM' + coin.tea.CTM' \end{align} $$

We can prove $CTM \not\sim CTM'$ by contradiction:

- Assume $CTM \sim CTM'$
- Then there's a bisimulation $R$ with $CTM \mathrm{Re}l CTM'$
- After $CTM \stackrel{coin}{\longrightarrow} CTM_1$ where $CTM_1 = coffee.CTM + tea.CTM$
- There should be $CTM' \stackrel{coin}{\longrightarrow} CTM_1'$ with $CTM_1 , R , CTM_1'$
- But $CTM_1'$ could be either $coffee.CTM'$ or $tea.CTM'$
- Neither can match both the coffee and tea transitions of $CTM_1$
- Contradiction: $CTM \not\sim CTM'$

## Summary of Equivalence Relations

From weakest to strongest:

1. **Same Transitions**: Simple but not compositional
2. **Trace Equivalence**: Compositional but ignores branching structure
3. **Completed Trace Equivalence**: Considers deadlocks but not compositional
4. **Bisimilarity**: Strongest, considers branching structure and is compositional

Bisimilarity is generally preferred because it preserves the most behavioral properties while ensuring compositionality.

## Relationship to Specification Verification

When we have: $$ \begin{align} Office &= (CS \mid CM) \setminus {coin, coffee}\\Spec &= \overline{pub}.Spec \end{align} $$

We want to verify $Office \sim Spec$ using an appropriate equivalence relation. Bisimilarity provides the strongest guarantees, ensuring that the implementation not only produces the same sequences of observable actions but also preserves the branching structure of choices.