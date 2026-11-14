#!/bin/bash
# macOS dummynet shaping of the split runtime's TCP port on loopback (VALIDATION of the application-level
# emulation in netem.py; needs sudo). Each direction gets its own pipe: rate and one-way delay = RTT / 2.
#
#   sudo ./dummynet.sh up <wifi|4g|poor> <port>        e.g.  sudo ./dummynet.sh up 4g 5555
#   sudo ./dummynet.sh up <rate_mbit> <rtt_ms> <port>
#   sudo ./dummynet.sh status
#   sudo ./dummynet.sh down                             (removes the pipes, restores /etc/pf.conf)
#
# While "up", the main pf ruleset is replaced by a minimal one holding only the "usplit" anchors
# (Apple's default /etc/pf.conf anchors are restored by "down"). Then, without emulation in the runtime:
#   python check_netem.py --mode dummynet --port <port> --scenarios <scenario>
set -euo pipefail
TOKEN_FILE=/tmp/usplit_pf_token

case "${1:-}" in
  up)
    case "${2:-}" in
      wifi) RATE=50; RTT=20; PORT=${3:?port} ;;
      4g)   RATE=10; RTT=60; PORT=${3:?port} ;;
      poor) RATE=1;  RTT=250; PORT=${3:?port} ;;
      *)    RATE=${2:?rate_mbit}; RTT=${3:?rtt_ms}; PORT=${4:?port} ;;
    esac
    OW=$((RTT / 2))                                  # dnctl delay is in whole ms
    dnctl -q flush
    dnctl pipe 1 config bw ${RATE}Mbit/s delay ${OW}
    dnctl pipe 2 config bw ${RATE}Mbit/s delay ${OW}
    printf 'dummynet-anchor "usplit"\nanchor "usplit"\n' | pfctl -q -f -
    printf 'dummynet out quick on lo0 proto tcp from any to any port %s pipe 1\ndummynet out quick on lo0 proto tcp from any port %s to any pipe 2\n' \
      "$PORT" "$PORT" | pfctl -q -a usplit -f -
    pfctl -E 2>&1 | sed -n 's/^Token : //p' > "$TOKEN_FILE"
    echo "dummynet up: ${RATE} Mbit/s each way, RTT ${RTT} ms (one-way ${OW} ms), TCP port ${PORT} on lo0"
    ;;
  status)
    dnctl list || true
    pfctl -a usplit -s dummynet 2>/dev/null || true
    ;;
  down)
    pfctl -q -a usplit -F all 2>/dev/null || true
    dnctl -q flush || true
    pfctl -q -f /etc/pf.conf || true
    if [ -s "$TOKEN_FILE" ]; then pfctl -X "$(cat "$TOKEN_FILE")" 2>/dev/null || true; rm -f "$TOKEN_FILE"; fi
    echo "dummynet down; /etc/pf.conf restored"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
