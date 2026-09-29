# Application Crash Runbook

Application processes may crash because of uncaught exceptions, memory exhaustion, dependency failures, invalid configuration, or bad deployments. Check the application logs around the first failure, process exit codes, memory and CPU use, health checks, deployment history, and dependency status.

When the incident follows a release, compare the failing version with the previous healthy version. Repeated restarts can hide the original exception, so preserve the first stack trace and relevant request or job context before restarting services repeatedly.

Mitigation can include rolling back a faulty deployment, correcting configuration, restoring an unavailable dependency, or increasing resources only when metrics show genuine capacity pressure. Follow up with a regression test or alert that would detect the same failure earlier.
