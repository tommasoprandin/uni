**Apache Spark** is an open-source _framework_ for developing and running applications working on very big datasets, both in _local machines_ (local mode) or in _clustered_ environments.

Spark's main features are:
- _In-memory caching_, very important for efficient execution of _multi-round_ algorithms, avoiding the excessive usage of slow storage.
- _Fault tolerance_
- _Optimized scheduling_ of data transformations operating on distributed datasets

### Components

Spark provides:
- _Spark Core_, which is the computing engine of the framework, handling _task scheduling_, _optimization_ and _data abstractions_
- _Programming APIs_ usable with various languages such as _Scala_, _Java_, _Python_ and _R_.
- _Data Analysis APIs_ oriented to different use-cases:
	- _Spark SQL_ for structured data (e.g. relational)
	- _MLib_ for machine learning applications
	- _GraphX_ for graph analytics
	- _Spark Streaming_ for real-time data streams

### Why Spark?

A framework like Spark is necessary to process Big Data _efficiently_ and at _moderate costs_.
It is possible to achieve these goals by:
1. _Pooling_ together resources of several (relatively inexpensive) machines into one computing and storage distributed platform.
2. _Managing and coordinating_ the execution of tasks across the distributed platform.

### Spark Application Architecture

The basic architecture of a distributed application running on the Spark framework is composed by:

- A _Driver_ (master) that manages the application by:
	1. Creating the _Spark Context_, an object that represents a channel to access and manipulate Spark functionalities.
	2. _Distributes tasks_ to the executors (can be over multiple machines or local).
	3. _Monitors the status_ of the executors.
- Multiple _Executors_ (workers) that _execute_ the assigned tasks and _report_ their status to the driver.
- A _Cluster Manager_: when the application is run on a _distributed platform_, the cluster manager controls the physical machines and allocates resources to the application. There are many possible managers available for use such as Spark's Standaloge, YARN or Mesos.

![[mr-spark-arch.png]]

At run-time, when the application is run on a single machine both the driver and the executors run on that machine as _separate threads_.
Instead on a distributed platform they can run on different machines.

### Resilient Distributed Dataset (RDD)

The RDD is the fundamental data abstraction in Spark. It represents a _collection of homogeneous elements_ (i.e. of the same type), _partitioned_ and _distributed_ across several machines (if the application runs on a cluster).

#### Properties

- RDDs can be created from _data in stable storage_ (possibly distributed), or from _transformations_ over other RDDs.
- RDDs are _immutable_
- RDDs are _lazily evaluated_, i.e. they are not instantiated until they are actually needed. This behaviour has to be taken in consideration when evaluating performances, since Spark will not produce the RDD until they are really needed.
- The _lineage_ of any RDD, i.e. the sequence of transformations generating it, is stored by Spark. Thus it can restore the RDD in case of failure, by applying again the transformations starting from the original data in stable storage.

#### Partitioning

As mentioned in the introduction any RDD is partitioned into chunks, which are then distributed across the available machines.
The programmer can manually specify the number of partitions, or let Spark automatically handle it.

Partitions are created by default, using a _partitioner_ specified by the programmer. They typically are in the range of 2/3 times the number of cores to help with _load balancing_.

Partitioning is _exploited to enhance performance_, since Spark allocates map tasks such that every executor apply the map function on data present on the local partition (if possible).
Additionally the programmer can explicitly access RDD partitions to implement algorithms that require explicit partitioning (e.g. _mapPartition_).
Some ready-make Spark aggregation primitives automatically leverage RDD partitions (e.g. _reduceByKey_).

#### Operations

There are two main operations available over RDDs:

1. **Transformations**: a _transformation_ generates a new RDD $B$, starting from data in $A$. We distinguish between:
	- _Narrow_ transformations, where each partition of $A$ contributes to _at most one_ partition of $B$, stored in the same machine, thus no _shuffling_ across machines is needed. This allows for maximum parallelism (for instance during _map_ operations).
	- _Wide_ transformations, where each partition of $A$ can contribute to many partitions of $B$, hence _shuffling_ of data across machines may be required (for instance grouping before the reduce phase).
	![[mr-rdd-transform.png]]
2. **Actions**: an _action_ is a computation over the elements of $A$, which return a value to the application (e.g. the _count_ method). Recall, as mentioned above, that RDDs are materialized only when they are needed so _only_ when actions on it are performed. This must be considered carefully when measuring running times.

	It is possible to decide where to store the RDD data after the subsequent action using the following methods:
	- _cache()_: data is stored in RAM (if it fits). Data not fitting is recomputed when eventually needed (lazily).
	- _perisist()_: data is stored as specified by the provided parameter (e.g. MEMORY_AND_DISK stores the result in RAM, with the "overflowing" data spilled to the disk)