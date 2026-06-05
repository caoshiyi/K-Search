#!/usr/bin/env bash
# 安装 ascendc-dev-knowledge skill 的 references 参考库 (约 88M, 不入 git)。
#
# 该参考库是 AscendC API/架构文档检索库, 体积过大不纳入仓库。
# 首次使用或在新机器克隆仓库后, 运行本脚本从源仓库复制到位。
# materializer 物化候选 worktree 时会复制此 references, 保证候选本地可读且不回写仓库资产。
#
# 用法:
#   scripts/install_dev_knowledge.sh SRC_REFERENCES_DIR
#
# 也可通过环境变量 KSEARCH_DEV_KNOWLEDGE_SRC 提供源路径。

set -euo pipefail

SRC="${1:-${KSEARCH_DEV_KNOWLEDGE_SRC:-}}"

# 定位仓库根 (本脚本在 <repo>/scripts/ 下)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DST="$REPO_ROOT/k_search/kernel_generators/claude_assets/skills/ascendc-dev-knowledge/references"

if [[ -z "$SRC" ]]; then
  echo "[ERROR] 缺少源 references 目录。" >&2
  echo "        用法: scripts/install_dev_knowledge.sh SRC_REFERENCES_DIR" >&2
  echo "        或设置 KSEARCH_DEV_KNOWLEDGE_SRC 环境变量。" >&2
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] 源 references 目录不存在: $SRC" >&2
  echo "        请传入 ascendc-dev-knowledge/references 路径，或设置 KSEARCH_DEV_KNOWLEDGE_SRC。" >&2
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
