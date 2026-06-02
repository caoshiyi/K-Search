# log_reference 检索指南

**关联指南**: [troubleshooting.md](troubleshooting.md)（故障排查时常需配合日志）

## 文档概览

日志参考文档，覆盖日志级别配置、plog 框架使用和常见问题。

## 何时使用

- 配置日志级别（"怎么开 DEBUG 日志？"）
- plog 日志框架使用
- 日志相关 FAQ

## 文档结构

```
./references/log_reference_docs/
├── INDEX.md
└── FAQ/                                 # 日志常见问题
```

## 极速检索示例

```bash
# 看索引（文档量少，直接通读最快）
cat "./references/log_reference_docs/INDEX.md"

# 列出所有文档
find "./references/log_reference_docs/" -name "*.md" -not -name "INDEX.md"

# 搜索日志级别
grep -rl "日志级别\|LOG_LEVEL\|plog" "./references/log_reference_docs/"

# 搜索日志配置
grep -r "ASCEND_GLOBAL_LOG_LEVEL\|ASCEND_SLOG" "./references/log_reference_docs/"

# 直接读所有文档（只有 8 页）
find "./references/log_reference_docs/" -name "*.md" -not -name "INDEX.md" -exec cat {} \;
```
