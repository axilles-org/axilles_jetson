#!/usr/bin/env bash
# Runs the safety tests where the motor cannot be reached: unshare gives them an
# empty network namespace, so can0 does not exist for anything started from here.
set -euo pipefail
cd "$(dirname "$0")/.."
exec unshare -rn python3 -m pytest motor_safety_tests -q -p no:cacheprovider "$@"
