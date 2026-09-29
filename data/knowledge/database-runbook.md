# Database Timeout Runbook

Database requests can time out when connection pools are exhausted, queries are blocked, indexes are missing, or the database server is under CPU or I/O pressure. Start by checking active connections, slow queries, lock waits, pool saturation, database latency, and recent schema or index changes.

For PostgreSQL, inspect pg_stat_activity and long-running queries. Verify that application connection limits align with the database max_connections setting. A saturated pool often causes requests to wait before a query even reaches the database.

Mitigation can include terminating pathological queries, increasing pool capacity only when the server has headroom, adding or correcting indexes, reducing transaction duration, and rolling back a recent database change. Avoid blindly increasing timeouts because that can hide an overloaded database and increase resource contention.

After recovery, capture the slow-query evidence, connection-pool metrics, lock information, and the change that triggered the incident.
