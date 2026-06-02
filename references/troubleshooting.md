# troubleshooting 检索指南

**关联指南**: [api-reference.md](api-reference.md)（API 约束条件常导致的错误）

## 文档概览

故障处理文档，覆盖各模块错误码详解、典型故障专题（AI Core Error、OOM、进程卡住）和故障定位工具使用。

## 何时使用

- 遇到错误码需要查含义（"EZ9999 是什么？" "EE0001 怎么处理？"）
- AI Core Error 排查
- 内存 OOM 问题定位
- 进程卡死/中断问题
- 使用 asys 工具

## 文档结构

```
./references/troubleshooting_docs/
├── INDEX.md
├── 错误码参考/
│   ├── GE_Errors/                       # Graph Engine 错误
│   ├── RTS_Errors/                      # Runtime 错误
│   ├── HCCL_Errors/                     # 集合通信错误
│   ├── AI_CPU_Errors/                   # AI CPU 错误
│   ├── FE_Errors/                       # 前端错误
│   ├── Driver_Errors/                   # 驱动错误
│   ├── Auto_Tune_Errors/               # 自动调优错误
│   ├── TBE_Pass_Compiler_Errors/       # TBE 编译错误
│   └── TEFusion_Errors/                # 融合错误
├── 典型故障专题/
│   ├── AI_Core_Error问题定位专题/       # AI Core 错误现象/定位/处理
│   ├── 内存OOM问题定位专题/             # OOM 分析/定位/处理
│   ├── 进程中断问题定位专题/            # 进程异常退出
│   └── 进程卡住问题定位专题/            # 死锁/卡死
└── 故障定位工具/
    └── asys工具使用指导/                # asys 采集/分析
```

## 极速检索示例

### find — 按文件名定位

```bash
# 列出所有错误码模块
find "./references/troubleshooting_docs/错误码参考/" -maxdepth 1 -type d

# 列出 GE 错误码
find "./references/troubleshooting_docs/错误码参考/GE_Errors/" -name "*.md"

# 列出典型故障专题
find "./references/troubleshooting_docs/典型故障专题/" -name "*.md"

# 找 asys 工具文档
find "./references/troubleshooting_docs/故障定位工具/" -name "*.md"
```

### grep — 按内容搜索

```bash
# 按错误码查
grep -rl "EZ9999" "./references/troubleshooting_docs/"
grep -rl "EE0001" "./references/troubleshooting_docs/错误码参考/"
grep -rl "E10001" "./references/troubleshooting_docs/"

# 按错误前缀查某模块
grep -rl "^EG" "./references/troubleshooting_docs/错误码参考/GE_Errors/"

# 按故障类型查
grep -rl "AI Core Error\|aicore error" "./references/troubleshooting_docs/典型故障专题/"
grep -rl "OOM\|Out of Memory\|内存不足" "./references/troubleshooting_docs/"
grep -rl "卡死\|卡住\|hang\|deadlock" "./references/troubleshooting_docs/"

# 查特定错误信息
grep -rl "EZ9999" "./references/troubleshooting_docs/" | xargs grep -l "处理建议\|解决"
```

### cat — 直接阅读

```bash
# 看索引
cat "./references/troubleshooting_docs/INDEX.md"

# 读 AI Core Error 专题
cat "./references/troubleshooting_docs/典型故障专题/AI_Core_Error问题定位专题/"*.md

# 读 OOM 专题
cat "./references/troubleshooting_docs/典型故障专题/内存OOM问题定位专题/"*.md

# 读 asys 使用指导
cat "./references/troubleshooting_docs/故障定位工具/asys工具使用指导/"*.md
```

## 排查流程

1. **有错误码** → `grep -rl "EXXXXX" "./references/troubleshooting_docs/错误码参考/"` → 读错误码详情
2. **AI Core Error** → 读 `典型故障专题/AI_Core_Error问题定位专题/` 全部文档
3. **OOM** → 读 `典型故障专题/内存OOM问题定位专题/` 全部文档
4. **进程异常** → 根据"卡住"或"中断"选对应专题
5. **需要采集信息** → 读 `故障定位工具/asys工具使用指导/`
