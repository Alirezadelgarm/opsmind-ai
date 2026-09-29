# TLS Certificate Failure Runbook

TLS failures can happen because a certificate expired, the hostname does not match, an intermediate certificate is missing, trust stores are stale, or client and server protocol settings are incompatible. Inspect the presented certificate chain, expiration date, subject alternative names, issuer, and the hostname clients use.

Check whether certificate rotation or secret deployment recently changed. A service may have a valid certificate on disk while an old process or load balancer still serves the previous certificate. Validate from the client side because that reveals the chain actually presented over the network.

Mitigation can include deploying the renewed certificate, fixing the full chain, correcting the hostname, or updating trust configuration. After recovery, verify automated renewal and alerting before the next expiration.
