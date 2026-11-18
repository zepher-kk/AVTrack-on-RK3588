#!/usr/bin/env bash
# 板端 SSH/SCP 小工具：带重试，密码走 SSHPASS 环境变量（不落盘）。
# 用法： source ssh_lib.sh && bssh "ls -l"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
-o ConnectTimeout=15 -o PreferredAuthentications=password -o PubkeyAuthentication=no \
-o NumberOfPasswordPrompts=1 -o LogLevel=ERROR"

__need_pass() {
  if [ -z "${SSHPASS:-}" ]; then
    echo "需要先导出 SSHPASS（板端密码），例如：SSHPASS='****' bash $0" >&2
    exit 2
  fi
  command -v sshpass >/dev/null 2>&1 || { echo "缺少 sshpass：apt install sshpass" >&2; exit 2; }
}

bssh() {
  __need_pass
  local n=0
  until sshpass -e ssh $SSHOPT "$BOARD" "$@" 2>&1; do
    n=$((n + 1)); [ $n -ge 4 ] && return 1; sleep 3
  done
}

bscp() {
  __need_pass
  local n=0
  until sshpass -e scp $SSHOPT -r "$@" 2>&1; do
    n=$((n + 1)); [ $n -ge 4 ] && return 1; sleep 3
  done
}
