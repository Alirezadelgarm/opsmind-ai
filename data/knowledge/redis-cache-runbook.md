# Redis Cache Saturation Runbook

Redis latency and cache failures often appear when memory is exhausted, eviction increases, clients create too many connections, or expensive commands block the event loop. Inspect used memory, maxmemory policy, evicted keys, connected clients, slowlog entries, command latency, and persistence activity.

If memory pressure is the cause, first confirm whether the working set unexpectedly grew or the configured memory limit is too small. A sudden increase in evictions can reduce cache hit rate and shift load toward databases. Avoid flushing the entire cache unless the operational impact is understood.

Possible mitigations include removing a pathological workload, changing key TTL behavior, scaling the cache tier, correcting connection pooling, or replacing expensive commands with bounded alternatives.
