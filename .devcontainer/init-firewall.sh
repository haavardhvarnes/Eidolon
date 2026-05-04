#!/bin/bash
# Egress firewall for the Eidolon devcontainer.
# Default: deny all outbound. Allow only DNS, the Anthropic API, package
# registries, and GitHub. Adjust ALLOWED_DOMAINS for your needs.
#
# Run via postCreateCommand under sudo. Idempotent — safe to re-run.

set -euo pipefail
IFS=$'\n\t'

ALLOWED_DOMAINS=(
    "api.anthropic.com"
    "registry.npmjs.org"
    "pkg.julialang.org"
    "julialang-s3.julialang.org"
    "github.com"
    "api.github.com"
    "objects.githubusercontent.com"
    "codeload.github.com"
    "raw.githubusercontent.com"
)

echo "[firewall] flushing existing rules"
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
ipset destroy allowed-domains 2>/dev/null || true

echo "[firewall] allowing loopback"
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

echo "[firewall] allowing DNS (UDP/TCP 53) so we can resolve allowlisted hosts"
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT
iptables -A INPUT  -p tcp --sport 53 -j ACCEPT

echo "[firewall] allowing established/related"
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

echo "[firewall] allowing host-side gateway (so VS Code <-> server can talk)"
HOST_IP=$(ip route show | awk '/default/ {print $3; exit}')
if [[ -n "${HOST_IP:-}" ]]; then
    HOST_NET=$(echo "$HOST_IP" | sed 's/\.[0-9]*$/.0\/24/')
    iptables -A INPUT  -s "$HOST_NET" -j ACCEPT
    iptables -A OUTPUT -d "$HOST_NET" -j ACCEPT
fi

echo "[firewall] resolving and allowlisting:"
ipset create allowed-domains hash:net
for domain in "${ALLOWED_DOMAINS[@]}"; do
    echo "  - $domain"
    ips=$(dig +short "$domain" A | grep -E '^[0-9.]+$' || true)
    if [[ -z "$ips" ]]; then
        echo "    (warning: no A records resolved)"
        continue
    fi
    while IFS= read -r ip; do
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done <<< "$ips"
done

iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT

echo "[firewall] default deny on remaining outbound"
iptables -P INPUT   DROP
iptables -P FORWARD DROP
iptables -P OUTPUT  DROP

echo "[firewall] verification:"
if curl -sS --connect-timeout 4 https://api.anthropic.com/ > /dev/null; then
    echo "  api.anthropic.com reachable"
else
    echo "  WARNING: api.anthropic.com NOT reachable — check DNS allowlist"
fi
if curl -sS --connect-timeout 4 https://example.com/ > /dev/null 2>&1; then
    echo "  WARNING: example.com reachable — firewall is NOT effective"
    exit 1
else
    echo "  example.com correctly blocked"
fi

echo "[firewall] done"
