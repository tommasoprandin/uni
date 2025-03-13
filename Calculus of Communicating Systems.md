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