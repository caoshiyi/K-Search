#!/usr/bin/env bash
# 安装 ascendc-dev-knowledge skill 的 references 参考库 (约 88M, 不入 git)。
#
# 该参考库是 AscendC API/架构文档检索库, 体积过大不纳入仓库。
# 首次使用或在新机器克隆仓库后, 运行本脚本从源仓库复制到位。
# materializer 物化候选 worktree 时, 会把此 references 以符号链接方式共享, 不重复占用磁盘。
#
# 用法:
#   scripts/install_dev_knowledge.sh [SRC_REFERENCES_DIR]
#
# 默认源路径可用环境变量 KSEARCH_DEV_KNOWLEDGE_SRC 覆盖。

set -euo pipefail

DEFAULT_SRC="/home/c00958677/cv_agent_adv/agent_workdir/.claude/skills/ascendc-dev-knowledge/references"
SRC="${1:-${KSEARCH_DEV_KNOWLEDGE_SRC:-$DEFAULT_SRC}}"

# 定位仓库根 (本脚本在 <repo>/scripts/ 下)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DST="$REPO_ROOT/k_search/kernel_generators/claude_assets/skills/ascendc-dev-knowledge/references"

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] 源 references 目录不存在: $SRC" >&2
  echo "        请将 cv_agent_adv 的 ascendc-dev-knowledge/references 路径作为参数传入," >&2
  echo "        或设置 KSEARCH_DEV_KNOWLEDGE_SRC 环境变量。" >&2
  exit 1
fi

echo "[install_dev_knowledge] 源: $SRC"
echo "[install_dev_knowledge] 目标: $DST"

mkdir -p "$DST"
# -a 保留属性, --delete 保证目标与源一致 (优先 rsync, 回退 cp)
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$SRC/" "$DST/"
else
  rm -rf "$DST"
  mkdir -p "$DST"
  cp -a "$SRC/." "$DST/"
fi

echo "[install_dev_knowledge] 完成: $(du -sh "$DST" | cut -f1)"
