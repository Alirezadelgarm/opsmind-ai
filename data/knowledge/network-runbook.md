# Network Packet Loss Runbook

Packet loss, intermittent connectivity, and high latency can originate from overloaded interfaces, routing changes, firewall rules, DNS failures, or unhealthy network devices. Begin by checking interface errors, packet drops, latency between affected nodes, DNS resolution, routes, and recent firewall or security-group changes.

Compare connectivity from multiple hosts to determine whether the issue is isolated to one machine, one subnet, or a wider path. Tools such as ping, traceroute or tracert, and connection tests to the target service port can help narrow the failing hop.

Mitigation depends on the cause: revert an incorrect route or firewall change, move traffic away from a degraded link, repair a failing interface, or restore DNS service. Record the affected network path and evidence before and after remediation.
