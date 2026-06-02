# basic_knowledge 检索指南

**关联指南**: [api-reference.md](api-reference.md)（API 接口详情）、[troubleshooting.md](troubleshooting.md)（故障排查）

## 文档概览

编程指南 + 算子实践参考 + 入门教程，涵盖 AscendC 编程概念、编程范式、算子开发实例和性能优化方法论。

## 何时使用

- 理解编程概念（"什么是 PIPE？" "AI Core 架构？"）
- 学习编程范式（"CopyIn → Compute → CopyOut 怎么写？"）
- 查算子开发实例（"矢量算子怎么写？" "矩阵算子怎么写？"）
- 了解性能优化方法（"bank 冲突怎么避免？" "Tiling 怎么切？"）
- 工程化开发（"CMake 怎么配？" "算子怎么入图？"）

## 文档结构

```
./references/basic_knowledge_docs/
├── INDEX.md
├── 编程指南/
│   ├── 概念原理和术语/
│   │   ├── 神经网络和算子/              # 算子基础概念
│   │   ├── 内存访问原理/                # UB/L1/L2/GM 层次
│   │   └── 性能优化技术原理/            # 双缓冲/流水线
│   ├── 硬件实现/
│   │   └── 架构规格/                    # AI Core 架构
│   ├── 编程模型/
│   │   └── 编程范式/
│   │       └── AI_Core_SIMD编程/        # 典型范式/TPipe/TQue
│   ├── 语言扩展层/                      # AscendC 语言特性
│   ├── C++类库API/
│   │   ├── 基础API/
│   │   │   ├── 接口分类说明/            # API 分类总览
│   │   │   └── 常用操作速查指导/        # 快速查表
│   │   └── 高阶API/
│   │       └── 常用操作速查指导/
│   ├── 编译与运行/
│   │   └── AI_Core算子编译/             # 编译选项/流程
│   ├── 调试调优/
│   │   └── 功能调试/                    # 调试方法
│   └── 附录/
│       ├── FAQ/                          # 常见问题
│       ├── 常用操作/                     # 工具命令
│       ├── 工程化算子开发/
│       │   ├── Host侧Tiling实现/        # Tiling 开发指南
│       │   └── 算子包编译/              # 编译打包
│       ├── AI框架算子适配/
│       │   └── ONNX框架/               # ONNX 适配
│       └── 算子入图（GE图）开发/        # GE 图集成
│
├── 算子实践参考/
│   ├── SIMD算子实现/
│   │   ├── 矢量编程/
│   │   │   ├── 基础矢量算子.md          # Add 等基础示例
│   │   │   ├── DoubleBuffer场景.md      # 双缓冲
│   │   │   ├── TBuf的使用.md            # TBuf 管理
│   │   │   ├── Broadcast场景.md         # 广播
│   │   │   ├── 非对齐场景.md            # 非对齐处理
│   │   │   └── 多核&Tiling切分/         # 尾核/尾块/多核Tiling
│   │   ├── 矩阵编程（基础API）/         # 耦合/分离模式
│   │   ├── 矩阵编程（高阶API）/         # Matmul 高阶/特性场景
│   │   └── 融合算子编程/                # 通算融合/CV融合
│   ├── SIMD算子性能优化/
│   │   ├── Tiling策略/                  # Tiling 优化方法
│   │   ├── 内存访问/                    # bank冲突/对齐
│   │   ├── 流水编排/                    # 双缓冲/pipeline
│   │   ├── 矢量计算/                    # 计算优化
│   │   ├── 矩阵计算/                    # 矩阵优化
│   │   └── 头尾开销优化/                # 启动开销
│   ├── 优秀实践/                        # Matmul调优案例
│   ├── 功能调试/                        # 调试技巧
│   └── 性能分析/                        # 分析方法
│
└── 入门教程/
    ├── 什么是Ascend_C.md
    ├── 环境准备.md
    └── 快速入门/                        # HelloWorld/Add算子
```

## 极速检索示例

### find — 按文件名定位

```bash
# 找编程范式文档
find "./references/basic_knowledge_docs/编程指南/编程模型/编程范式/" -name "*.md"

# 找 Tiling 相关
find "./references/basic_knowledge_docs" -name "*Tiling*" -o -name "*tiling*"

# 找性能优化文档
find "./references/basic_knowledge_docs/算子实践参考/SIMD算子性能优化/" -name "*.md"

# 找矢量编程实例
find "./references/basic_knowledge_docs/算子实践参考/SIMD算子实现/矢量编程/" -name "*.md"

# 找入门教程
find "./references/basic_knowledge_docs/入门教程/" -name "*.md"

# 找工程化开发
find "./references/basic_knowledge_docs/编程指南/附录/工程化算子开发/" -name "*.md"
```

### grep — 按内容搜索

```bash
# 查编程概念
grep -rl "抽象硬件架构\|AI Core" "./references/basic_knowledge_docs/编程指南/"
grep -rl "编程范式\|CopyIn.*Compute.*CopyOut" "./references/basic_knowledge_docs/"
grep -rl "TPipe\|TQue" "./references/basic_knowledge_docs/编程指南/"

# 查性能优化技术
grep -rl "DoubleBuffer\|双缓冲" "./references/basic_knowledge_docs/"
grep -rl "bank冲突\|bank conflict" "./references/basic_knowledge_docs/算子实践参考/"
grep -rl "流水编排\|pipeline" "./references/basic_knowledge_docs/算子实践参考/"

# 查 Tiling 方法
grep -rl "尾核\|尾块\|Tail" "./references/basic_knowledge_docs/算子实践参考/SIMD算子实现/矢量编程/"

# 查内存层次
grep -rl "Unified Buffer\|UB\|L1 Buffer\|Global Memory" "./references/basic_knowledge_docs/编程指南/概念原理和术语/"

# 查编译相关
grep -rl "CMakeLists\|cmake\|编译" "./references/basic_knowledge_docs/编程指南/编译与运行/"
```

### cat — 直接阅读

```bash
# 看索引
cat "./references/basic_knowledge_docs/INDEX.md"

# 读核心编程概念
cat "./references/basic_knowledge_docs/编程指南/编程模型/编程范式/AI_Core_SIMD编程/"*.md | head -100

# 读矢量编程实例
cat "./references/basic_knowledge_docs/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md"

# 读 DoubleBuffer
cat "./references/basic_knowledge_docs/算子实践参考/SIMD算子实现/矢量编程/DoubleBuffer场景.md"

# 读 Tiling 策略
cat "./references/basic_knowledge_docs/算子实践参考/SIMD算子性能优化/Tiling策略/"*.md
```

## 常用概念速查

| 概念 | 检索路径 |
|------|---------|
| AI Core 架构 | `编程指南/硬件实现/架构规格/` |
| 编程范式 | `编程指南/编程模型/编程范式/AI_Core_SIMD编程/` |
| TPipe/TQue | `编程指南/编程模型/` + `编程指南/C++类库API/` |
| Tiling 切分 | `算子实践参考/SIMD算子实现/矢量编程/多核&Tiling切分/` |
| 性能优化 | `算子实践参考/SIMD算子性能优化/` |
| bank 冲突 | `算子实践参考/SIMD算子性能优化/内存访问/` |
| 融合算子 | `算子实践参考/SIMD算子实现/融合算子编程/` |
| 工程化开发 | `编程指南/附录/工程化算子开发/` |
