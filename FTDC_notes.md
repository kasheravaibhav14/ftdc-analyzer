# Notes on FTDC Metrics

## Working:
1. Decode the FTDC files based on timestamps, each log file is made up of chunks of 5 minutes of data and is named as per the first timestamp in the file. Taking all those files whose timestamps are at a delta of 6 hours absolute from our requested timestamp. `FTDC_capture.py` takes care of this
2. `FTDC_decoder.py` decodes each timestamp file and passes them as a dictionary of json object(each object containing 300 timepoints), keys of the dictionary are the timestamps in sequence. It creates an analysis object of `FTDC_analysis` to which it passes the dictionary, queryTimestamp, output path and duration. 
3. `FTDC_analysis.py` takes care of the analysis. It first creates a list of metrics to monitor, deciding whether they are point in time or cumulative in nature. It opens `FTDC_metrics.json` which contains a list of chosen and discarded metrics segregated as required. Then the function first decodes the dictionary to form a dictionary in which (key,value) is metric name and its timeseries as a list. 
4. Then time bounds are calculated based on queryTimestamp(qT). We check in the interval (qT-2*bucket_duration,qT+bucket_duration) and the point t0 is determined when either avaialable read or write tickets go below 50. Then we check for 12 intervals(1 hour if interval duration is 5 mins) to create the upper and lower limit for analysis.
5. For each metric, we check for anomaly in this bounds. We first calculate interval wise mean and 99th percentile score. We use InterQuartileRange on mean and IsolationForest on 99 percentile to check for anomalies. If the number of anomalous metrics are more than 60, we run IQR on 99th percentile score as well to perform higher thresholding. Some metrics are included based on their absolute values like cache ratio, dirty cache ratio, history store score etc. [can rewrite cache ratio and dirty cache ratio interval finding - discuss]
6. After that we use a baseline prompt modified with the type of ticket drop(read/write) and other information. We add the list of anomalies and their mean,99 percentile and average values of both of them across intervals as information to be passed to chatgpt. The prompt is then passed to a function which uses `ChatCompletion` on openAI to return a response. The dataframe and response message is passed to `FTDC_plot` class.
7. The `FTDC_plot` class takes care of dynamically plotting the graphs(svg) with their details, min and max values. It uses reportlab and svglib to implement it. At the end of all graphs, it appends the summary received from openAI API which is our final output.

## serverStatus

- `asserts`: Counts number of user assertions, not much useful

- `connections`[serverStatus.connections]: `available`, `current`, `totalCreated`, `active`
    - `exhaustHello`(v5.0) and `exhaustIsMaster`(v4.4.2) is a special command used to check the status and capabilities of a MongoDB server. When a client sends a "hello" or "isMaster" request to a MongoDB server, the server responds with information about its role, version, replica set configuration (if applicable), and various other details.
    - `awaitingTopologyChanges`: The number of clients currently waiting in a hello or isMaster request for a topology change. The awaitingTopologyChanges role allows the user to monitor the MongoDB cluster's topology changes. Users assigned the awaitingTopologyChanges role have read-only access to the cluster's configuration and monitoring information. They can view the current state of the cluster and receive notifications when changes occur, but they do not have write or administrative privileges.



- `electionMetrics`[serverStatus.electionMetrics]: 
    - `stepUpCmd`: number of elections when primary stepped down
    - `priorityTakeover`

- `opLatencies`[serverStatus.opLatencies]:
    - latency and ops field for `reads`, `writes`, `commands`, `transactions`

- `opCounters`[serverStatus.opcounters]:
    - `insert`, `query`, `update`, `delete`, `getmore`, `command`
    - These reflect recieved/requested operations, not necessary successful ones.
    - Bulk insert or multi update are counted as a single operation

- `opCountersRepl`[serverStatus.opcountersRepl]:
    - `insert`, `query`, `update`, `delete`, `getmore`, `command`
    - A document that reports on database replication operations by type since the mongod instance last started.These values only appear when the current host is a member of a replica set.These values will differ from the opcounters values because of how MongoDB serializes operations during replication. 

- `mem`[serverStatus.mem]: A document that reports on the system architecture of the mongod and current memory use.
    - `bits`(32/64 depending on sys arch) 
    - `resident`(used RAM in mebibyte ($2^{20}$ bytes)), `virtual`(used virtual memory in MiB), `supported`(indicates whether the underlying system supports extended memory information)

- `locks`[serverStatus.locks]: <detail more>
    - different types of locks[https://github.com/mongodb/mongo/blob/master/src/mongo/db/repl/README.md]: 
        - `ParallelBatchWriterMode`
        - `ReplicationStateTransition`
        - `Global`
        - `Database`
        - `Mutex`
        - `oplog`
    - different functions: 
        - `acquireCount`
        - `acquireWaitCount`
        - `timeAcquiringMicros`
        - `deadlockCount`
    - different modes of locking:
        - 'R' : Represents Shared (S) lock.
        - 'W' : Represents Exclusive (X) lock. 
        - 'r' : Represents Intent Shared (IS) lock.
        - 'w' :Represents Intent Exclusive (IX) lock.


- `metrics`[serverStatus.metrics]:
    - `aggStageCounters`: use of aggregation pipeline stages; reports the number of times a particular stage has been executed
        - `indexStats`
    - `commands`: for each different command, it includes counters for failed and total
    - `document`: A document that reflects document access and modification patterns. Compare these values to the data in the  opcounters document, which track total number of operations. [referring to only opCounters in serverStatus for now]
        - `deleted`, `inserted`, `returned`, `updated` {upsert counted as inserted}
    - `operation`:
        - `scanAndOrder`: The total number of queries that return sorted numbers that cannot perform the sort operation using an index. (implication is in memory sort which is slow)
        - `writeConflicts` : The total number of queries that encountered write conflicts.


    - `cursor`:
        - `timedOut`: The total number of cursors that have timed out since the server process started. If this number is large or growing at a regular rate, this may indicate an application error.

        - `open`:
            - `total`: The number of cursors that MongoDB is maintaining for clients. Because MongoDB exhausts unused cursors, typically this value small or zero. However, if there is a queue, or stale tailable cursors, or a large number of operations this value may increase.
            - `noTimeout`: The number of open cursors with the option DBQuery.Option.noTimeout set to prevent timeout after a period of inactivity.
            - `pinned`: The number of "pinned" open cursors.
    - `metrics.queryExecutor.collectionScans`: A document that reports on the number of queries that performed a collection scan.
    - `metrics.queryExecutor.collectionScans.total` :The total number queries that performed a collection scan. The total consists of queries that did and did not use a tailable cursor.



- `wiredTiger`[serverStatus.wiredTiger]:
    - `concurrentTransactions`: 
        - A document that returns information on the number of concurrent of read and write transactions allowed into the WiredTiger storage engine. 
        - categorized as `out`, `available` and `totalTickets` for both read and write
    /////wiredTiger.async:  A document that returns statistics related to the asynchronous operations API. This is unused by MongoDB. (written like this in documentation, but is present on a server ftdc file, values are 0 though)

    - `async.current work queue length`: same but current instead of max
    - `async.maximum work queue length`: measures the length or size of the work queue used for asynchronous operations. highest number of operations that were simultaneously waiting or in progress in the work queue. It indicates the peak workload or level of concurrency for the asynchronous operations.

    - `async.number of flush calls`: represents the count of explicit flush operations performed by WiredTiger to persist data from memory to disk. If high, means lot of transfers to disk, affecting performance and disk I/O.


    - `cache`:
        - In the context of MongoDB's WiredTiger storage engine, cache overflow occurs when the in-memory cache, known as the cache storage or cache cache, becomes full. The cache is used to store frequently accessed data, providing fast and efficient access without the need to access data from disk.

        When the cache storage reaches its capacity, WiredTiger employs a mechanism called cache overflow to handle the excess data that cannot fit in the cache. This overflowed data is temporarily stored on disk, in a separate data structure called cache overflow tables or cache eviction tables.

        Cache overflow tables are disk-based structures used to hold data that has been overflowed from the in-memory cache. These tables allow the system to continue operating even when the cache is full, ensuring that data is not lost due to cache limitations.
        - stats denoting insufficient cache in the system:
            - `cache overflow score`
            - `cache overflow table entries`
            - `bytes written from cache`: number of bytes written from the cache back to disk. It captures the amount of data that has been modified in the cache and subsequently flushed to disk to ensure durability and persistence.


        - `maximum bytes configured`
        - `bytes currently in the cache`
        - `bytes dirty in the cache cumulative`
        - `bytes allocated for updates`
        - `tracked dirty bytes in the cache`
        - `pages queued for urgent eviction from history store due to high dirty content`
        - `tracked dirty pages in the cache`
        - `percentage overhead`: total usage of cache until now
        - `pages evicted`: This parameter represents the number of cache pages that have been evicted.
        - `pages selected for eviction unable to be evicted`: This parameter indicates the number of cache pages that were selected for eviction but could not be evicted due to various reasons, such as being locked or holding uncommitted changes.
    - stats denoting wait time/ capacity:
        - `threshold to call fsync`
        - `time waiting due to total capacity (usecs)`
        - `time waiting during checkpoint (usecs)`
    - `perf`:
        - File system read and write latency histogram
        - Operation read and write latency histogram

    - `transaction`:
        - `transaction checkpoint max time (msecs)`: This parameter indicates the maximum amount of time, in milliseconds, that a transaction checkpoint operation is allowed to take. If a checkpoint exceeds this limit, it may impact the performance of the system.
        - `transaction checkpoint min time (msecs)`: This parameter represents the minimum amount of time, in milliseconds, that a transaction checkpoint operation is expected to take. If a checkpoint finishes too quickly, it may indicate that not enough data was written to disk.
        - `transaction begins`
        - `transactions committed` 
        - `transactions rolled back`
        - `update conflicts`
        
    - `connection`: 
        - `files currently open`
        - `memory allocations`, `memory frees`, `memory re-allocations`
        - `pthread mutex condition wait calls`:  This parameter represents the number of times WiredTiger has used a pthread mutex condition wait during its operations. It indicates the usage of mutex locks and wait conditions for synchronization. They provide information about the memory usage patterns.
        - `total fsync I/Os`, `total read I/Os`, `total write I/Os`: These parameters represent the total number of fsync I/O operations, read I/O operations, and write I/O operations performed by WiredTiger. They provide information about the storage I/O activity and workload patterns.

    - `thread-state`:
       - `active filesystem fsync calls`
       - `active filesystem read calls`
       - `active filesystem write calls`

## systemMetrics

- `cpu`:


## 10th June, 2023

`flowControl`: A document that returns statistics on the Flow Control. With flow control enabled, as the majority commit point lag grows close to the flowControlTargetLagSeconds, writes on the primary must obtain tickets before taking locks. As such, the metrics returned are meaningful when run on the primary.


- `flowControl.timeAcquiringMicros` : When run on the primary, the total time write operations have waited to acquire a ticket.

- `flowControl.locksPerOp` : When run on the primary, an approximation of the number of locks taken per operation.

    `flowControlWarnThresholdSeconds`: default 10 secs; The amount of time to wait to log a warning once the flow control mechanism detects the majority commit point has not moved.

/*
To read `opReadConcernCounters` and `opWriteConcernCounters` again
Sharding and replicaSets have their own separate stats. Sharding helps to distribute data, replicaset is used to create redundant data to ensure high availability, the secondary can be used to read as well
*/

/*
Journaling and log writing are impactful in a lot of cases, cannot ignore them as well.
*/

/*
No sharding needed, Replication sets pe hi kaam karna hai
*/

Notes on tcmalloc:
TCMalloc (Thread-Caching Malloc) is a memory allocator that reduces lock contention for multi-threaded programs, making it suitable for performance-critical applications like MongoDB. The `serverStatus.tcmalloc` metrics provide information about TCMalloc's internal behavior, which can help diagnose performance issues related to memory allocation.

1. `serverStatus.tcmalloc.generic.current_allocated_bytes`: This counter shows the current number of bytes allocated by the application. It is not an accumulating stat but gives you a snapshot of the current memory usage. High memory usage could cause performance issues.

2. `serverStatus.tcmalloc.generic.heap_size`: This counter shows the current total heap size. This includes all malloced memory, memory in freelists, and memory in unmapped pages. It is not an accumulating stat. A significant difference between `heap_size` and `current_allocated_bytes` may suggest memory fragmentation.

3. `serverStatus.tcmalloc.tcmalloc.pageheap_free_bytes`: This counter shows the amount of free memory within the page heap. It is not an accumulating stat. A low value may suggest memory pressure.

4. `serverStatus.tcmalloc.tcmalloc.pageheap_unmapped_bytes`: This counter shows the amount of memory returned to the system (i.e., unmapped), but still reserved by the page heap. It is not an accumulating stat. A high value here may suggest your application has released significant memory recently.

5. `serverStatus.tcmalloc.tcmalloc.max_total_thread_cache_bytes`: This counter shows the maximum total size of thread caches. It is not an accumulating stat.

6. `serverStatus.tcmalloc.tcmalloc.current_total_thread_cache_bytes`: This counter shows the current total size of thread caches. It is not an accumulating stat. High thread cache usage could suggest heavy multithreaded activity.

7. `serverStatus.tcmalloc.tcmalloc.total_free_bytes`: This counter shows the total free bytes, which may be touched by application threads without any system calls. It is not an accumulating stat. 

8. `serverStatus.tcmalloc.tcmalloc.central_cache_free_bytes`: This counter shows the total bytes in free, mapped central-cache pages. It is not an accumulating stat. 

9. `serverStatus.tcmalloc.tcmalloc.transfer_cache_free_bytes`: This counter shows the total bytes in free, mapped transfer-cache pages. It is not an accumulating stat.

10. `serverStatus.tcmalloc.tcmalloc.thread_cache_free_bytes`: This counter shows the total bytes in free, mapped thread-cache pages. It is not an accumulating stat.

11. `serverStatus.tcmalloc.tcmalloc.aggressive_memory_decommit`: This counter shows whether TCMalloc is returning unused memory to the system aggressively. It is not an accumulating stat.

12. `serverStatus.tcmalloc.tcmalloc.pageheap_committed_bytes`: This counter shows the amount of memory committed to the page heap. It is not an accumulating stat.

13. `serverStatus.tcmalloc.tcmalloc.pageheap_scavenge_count`: This counter shows the number of times the scavenger has been invoked. This is an accumulating stat.

14. `serverStatus.tcmalloc.tcmalloc.pageheap_commit_count`: This counter shows the number of times a commit has been performed. This is an accumulating stat.

15. `serverStatus.tcmalloc.tcmalloc.pageheap_total_commit_bytes`: This counter shows the total bytes committed. This is an accumulating stat.

16. `serverStatus.tcmalloc.tcmalloc.pageheap_decommit_count`: This counter shows the number of times a decommit has been performed. This is an accumulating stat.

17. `serverStatus.tcmalloc.tcmalloc.pageheap_total_decommit_bytes`: This counter shows the

 total bytes decommitted. This is an accumulating stat.

18. `serverStatus.tcmalloc.tcmalloc.pageheap_reserve_count`: This counter shows the number of times a reserve has been performed. This is an accumulating stat.

19. `serverStatus.tcmalloc.tcmalloc.pageheap_total_reserve_bytes`: This counter shows the total bytes reserved. This is an accumulating stat.

20. `serverStatus.tcmalloc.tcmalloc.spinlock_total_delay_ns`: This counter shows the total delay caused by spinlocks. This is an accumulating stat and could indicate contention in memory allocation.

21. `serverStatus.tcmalloc.tcmalloc.release_rate`: This counter shows the rate at which the page heap releases memory. It is not an accumulating stat. A high release rate may suggest memory churn.

These metrics can be important for diagnosing memory-related performance issues. If memory becomes a bottleneck, both read and write operations could be affected, leading to a potential decrease in available read/write tickets. However, these stats do not directly measure the availability of read/write tickets.

## logical session cache 

- In MongoDB, a session refers to a series of operations performed by a client application. These operations could include reads, writes, or transactions. Sessions are used in MongoDB to handle and manage these sequences of operations from clients. When a client initiates operations on a MongoDB server, a session is created for that client. This session helps MongoDB manage and maintain the state of the operations. Each session is associated with a unique session ID.

- The Logical Session Cache in MongoDB is a component that caches session information in memory. This cache is used to manage the sessions established between the client applications and the MongoDB server. The Logical Session Cache tracks active sessions and is responsible for maintaining session metadata.

- The cache is crucial for multi-document transactions and causally consistent sessions. It stores session information, like the session ID and last use time, and also maps operations to their respective sessions. The cache periodically updates the sessions collection with this information. This means it keeps track of the current active sessions, and allows MongoDB to efficiently manage resources associated with these sessions.

Here are the descriptions of the provided MongoDB WiredTiger metrics:

1. `serverStatus.wiredTiger.cache."application threads page read from disk to cache count"`: This metric represents the total number of pages read into the cache from the disk by the application threads. High values might mean the working set (data MongoDB is actively using) exceeds your available RAM. Accumulative in nature.

2. `serverStatus.wiredTiger.cache."application threads page read from disk to cache time (usecs)"`: This shows the total time spent by the application threads reading pages into the cache from the disk. This is measured in microseconds. A high value may indicate I/O latency. Accumulative in nature.

3. `serverStatus.wiredTiger.cache."application threads page write from cache to disk count"`: This is the total number of pages written from the cache to the disk by the application threads. A high value may suggest high write operations. Accumulative in nature.

4. `serverStatus.wiredTiger.cache."application threads page write from cache to disk time (usecs)"`: This indicates the total time spent by application threads writing pages from the cache to the disk, in microseconds. A high value can indicate I/O latency. Accumulative in nature.

5. `serverStatus.wiredTiger.cache."bytes allocated for updates"`: This shows the number of bytes allocated for modifications in the cache. A high value might imply a lot of write operations happening.

6. `serverStatus.wiredTiger.cache."bytes belonging to page images in the cache"`: This measures the number of bytes in the cache belonging to the internal page images. A high value can mean that MongoDB is trying to optimize for read I/Os by saving more page images into the cache.

7. `serverStatus.wiredTiger.cache."bytes belonging to the history store table in the cache"`: This metric indicates the amount of memory (in bytes) in the cache that is being used by the history store table. A high value may indicate a high number of updates or deletes, which requires keeping track of older versions of documents.

8. `serverStatus.wiredTiger.cache."bytes currently in the cache"`: This tells you the total amount of data currently held in the cache. Comparing this value to the cache size can help diagnose if the cache is too small for your working set.

9. `serverStatus.wiredTiger.cache."bytes dirty in the cache cumulative"`: This refers to the total size of all dirty (modified but not yet written to disk) pages in the cache. This is cumulative and may increase with more write operations.

10. `serverStatus.wiredTiger.cache."bytes not belonging to page images in the cache"`: This shows the total amount of memory in the cache that isn't associated with internal page images. A high value can indicate more memory being used for other MongoDB operations.

11. `serverStatus.wiredTiger.cache."bytes read into cache"`: This represents the total amount of data read into the cache. A high value may indicate a high amount of read operations. Accumulative in nature.

12. `serverStatus.wiredTiger.cache."bytes written from cache"`: This shows the total amount of data written from the cache to the disk. A high value can mean a high number of write operations. Accumulative in nature.

13. `serverStatus.wiredTiger.cache."cache overflow score"`: This is a score (between 0 and 100) indicating the extent to which the cache is overflowing. A high score (close to 100) means the cache is frequently full, suggesting the working set exceeds your available RAM.

These metrics can provide important insights into the performance of your MongoDB instance. Keep in mind that many factors, including the specific workload and hardware, can impact these metrics, and high or low values aren't necessarily bad—it depends on the context.

serverStatus.wiredTiger.block-manager.bytes read/written via memory map API: block manager bytes using main memory
serverStatus.wiredTiger.block-manager.bytes read/written via system call API: block manager bytes using disks

serverStatus.wiredTiger.cache.checkpoint of history store file blocked non-history store page eviction: Number of times a checkpoint operation on a history store file blocks the eviction of a non history file

serverStatus.wiredTiger.cache.checkpoint blocked page eviction: number of times a checkpoint operation blocks page eviction. Can cause increase in dirty cache.


serverStatus.wiredTiger.cache.history store score: In general, the history store score is a measure of how much data the history store is contributing to the overall cache. If the history store score is consistently high, it may suggest that many updates are occurring on your database, resulting in numerous historical versions of documents being stored in the cache. This could lead to increased memory usage and potentially affect the performance of your database.

Memory operations such as allocations, frees, and re-allocations are fundamental to the functioning of any application, including MongoDB. However, the behavior of these operations can also provide insight into the application's performance and possible issues. 

1. **Memory Allocations:** This count increases when MongoDB requests additional memory to handle its operations. If this count is increasing rapidly, it might be a sign that your application is dealing with larger data sets or more complex operations than usual. High memory allocation can lead to increased memory usage and might result in memory pressure. If the system's memory is exhausted, MongoDB might start swapping memory pages to disk, which can significantly degrade performance.

2. **Memory Frees:** This count increases when MongoDB releases memory that it no longer needs. If the count of memory frees is significantly lower than the count of memory allocations over time, it may indicate a memory leak, where memory that is no longer needed is not being appropriately freed.

3. **Memory Re-allocations:** This count increases when MongoDB needs to change the amount of memory allocated to a particular operation or data structure. High levels of memory re-allocation can be a performance concern because re-allocating memory is generally more costly in terms of CPU time than either allocating or freeing memory. Rapid increases in re-allocations could indicate that data structures are frequently changing in size, which might be a symptom of inefficient data handling or highly variable workloads.

While these metrics provide valuable insight, it's important to remember that they are just one piece of the puzzle when monitoring application performance. They should be interpreted in the context of other performance metrics and the specific workload your MongoDB instance is handling. If any of these metrics are trending in an unexpected direction, it might be worth investigating whether changes in your application or workload could be causing increased memory operations.

