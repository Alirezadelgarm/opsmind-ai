# Kubernetes CrashLoopBackOff Runbook

A pod in CrashLoopBackOff is repeatedly starting and failing. Inspect pod events, container logs including the previous container instance, readiness and liveness probes, resource limits, mounted configuration, secrets, and the image or deployment revision.

Common causes include an application exception during startup, a missing environment variable, an unreachable dependency, an incorrect command, or an out-of-memory kill. Use kubectl describe pod and kubectl logs --previous to capture evidence from the failed container.

Mitigation may be a rollback, corrected configuration, repaired dependency, adjusted probe, or resource change. Confirm that the pod remains healthy and that traffic is successfully served before closing the incident.
