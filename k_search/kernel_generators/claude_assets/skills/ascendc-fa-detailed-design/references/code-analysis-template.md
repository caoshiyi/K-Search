# Code Analysis Template

AscendC 算子代码梳理文档标准模板。基于实际生产算子（如 KvQuantSparseFlashAttentionPioneer）的梳理实践总结。

## Document Structure

```markdown
# {OpName} 算子代码梳理

## 1. 算子概述

### 1.1 功能定位
[一段话描述算子的核心功能、目标场景、技术特点]

### 1.2 计算公式
$$
[核心计算公式，LaTeX 格式]
$$
- [公式中各符号的说明]

### 1.3 支持平台
| SoC | 架构目录 | 数据类型特点 |
|-----|---------|------------|
| ascend910b | arch32 | ... |
| ascend950 | arch35 | ... |

---

## 2. 目录结构

[带注释的完整文件树，每个文件标注职责]

```
{op_name}/
├── CMakeLists.txt                    # 顶层构建入口
├── README.md                         # API 文档与调用示例
│
├── op_host/                          # ═══ Host 侧（CPU 端）═══
│   ├── CMakeLists.txt                # Host 构建配置
│   ├── *_def.cpp                     # OpDef：算子接口定义
│   ├── *_infershape.cpp              # InferShape：输出 shape 推导
│   ├── *_tiling.h                    # Tiling 头文件：数据结构、类声明
│   ├── *_tiling.cpp                  # Tiling 实现：切分策略
│   └── arch*/                        # 架构特化 Tiling
│
└── op_kernel/                        # ═══ Kernel 侧（NPU 端）═══
    ├── *.cpp                         # Kernel 入口：__global__ 函数
    ├── *_kernel*.h                   # 主 Kernel 类：初始化 + 主循环
    ├── *_common.h                    # 公共常量、枚举、对齐工具
    ├── *_service_cube*.h             # Cube 服务：AIC 核矩阵乘
    ├── *_service_vector*.h           # Vector 服务：AIV 核向量运算
    ├── *_kvcache.h                   # KV Cache 参数计算
    ├── *_template_tiling_key.h       # 模板参数组合声明
    ├── util_regbase.h                # 运行时参数结构体
    └── vf/                           # 微核函数
```

---

## 3. 输入/输出/属性规格

### 3.1 输入张量
| 序号 | 名称 | 必选 | 数据类型 | Shape | 说明 |
|------|------|------|---------|-------|------|
| 0 | ... | 是 | ... | [...] | ... |

### 3.2 输出张量
| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| ... | ... | [...] | ... |

### 3.3 关键属性
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| ... | ... | ... | ... |

### 3.4 布局支持
| 布局 | 维度 | 适用张量 | 说明 |
|------|------|---------|------|
| ... | ... | ... | ... |

---

## 4. Host 侧逻辑

### 4.1 OpDef（*_def.cpp）
[注册的输入/输出/属性数量、各 SoC 的数据类型约束、特殊配置项]

### 4.2 InferShape（*_infershape.cpp）
[输出 shape 推导公式、数据类型推导规则]

### 4.3 Tiling 策略（*_tiling.h / *_tiling.cpp）

#### 4.3.1 Tiling 主流程
[函数调用链，从 DoOpTiling() 到各子步骤]

```
DoOpTiling()
  ├── GetPlatformInfo()
  ├── InitParams()
  ├── Split()
  │   └── SplitBalanced() / SplitXxx()
  ├── FillTiling()
  ├── CalcBlockDim()
  ├── GetWorkspaceSize()
  └── GenTilingKey()
```

#### 4.3.2 关键切分参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| ... | ... | ... |

#### 4.3.3 TilingData 结构体
[完整的结构体层次，各子结构及其核心字段]

#### 4.3.4 Workspace 组成
[各部分的大小计算公式]

#### 4.3.5 参数校验
[校验规则列表]

---

## 5. Kernel 侧逻辑

### 5.1 整体架构：AIC/AIV 异构协同
[架构图：AIC/AIV 角色、通信通道、内存层级]

```
┌──────────────────────────────────────────────┐
│                 NPU AI Core                   │
│  ┌──────────┐   SSBuf   ┌──────────┐         │
│  │ AIC(Cube)│◄────────►│AIV(Vector)│         │
│  │ Service  │  同步通道  │ Service   │         │
│  └──────────┘           └──────────┘         │
│       │                      │                │
│       ▼                      ▼                │
│  ┌─────────────────────────────────┐         │
│  │   L1 Buffer / L0A/L0B/L0C      │         │
│  └─────────────────────────────────┘         │
│                   │                           │
│                   ▼                           │
│  ┌─────────────────────────────────┐         │
│  │       Global Memory (GM/HBM)    │         │
│  └─────────────────────────────────┘         │
└──────────────────────────────────────────────┘
```

### 5.2 Kernel 入口（*.cpp）
[__global__ 函数签名、模板参数说明、数据类型组合表、AIC/AIV 实例化方式]

### 5.3 主 Kernel 类（*_kernel*.h）

#### 5.3.1 内存层级与 Buffer 布局
[各内存层级（GM/L1/L0/UB）的 Buffer 分配表]

#### 5.3.2 初始化流程
[Init() 函数的步骤分解]

#### 5.3.3 主循环
[ProcessMainLoop 的多级循环结构、三级/多级流水编排、时序图]

```
时间 →    T0          T1          T2          T3
AIC:   BMM1[0]     BMM1[1]     BMM1[2]     ...
                   BMM2[0]     BMM2[1]     ...
AIV:   Vec0[0]     Vec0[1]     Vec0[2]     ...
                   Vec1[0]     Vec1[1]     ...
                               Vec2[0]     ...
```

### 5.4 Cube 服务（*_service_cube*.h）
[核心常量、Buffer 分配、IterateBmm1/Bmm2 等核心方法]

### 5.5 Vector 服务（*_service_vector*.h）
[核心常量、三阶段处理 ProcessVec0/Vec1/Vec2 的详细流程]

### 5.6 微核函数（vf/ 目录）
[如有：调度路由、各优化路径、Flash Update 公式]

### 5.7 KV Cache 参数计算
[如有：关键函数及其功能、稀疏索引映射逻辑]

---

## 6. 运行时参数结构体（util_regbase.h）

### 6.1 RunInfo / RunParamStr
| 字段 | 说明 |
|------|------|
| ... | ... |

### 6.2 ConstInfo
| 字段 | 说明 |
|------|------|
| ... | ... |

### 6.3 CVSharedParams（核间共享参数）
[位域结构定义、各字段说明]

---

## 7. 端到端数据流

[从 Host 到 NPU 的完整数据通路图]

```
         Host (CPU)                    NPU Kernel
    ┌──────────────┐          ┌─────────────────────┐
    │ OpDef 校验    │          │ Stage 0: 数据加载    │
    │ InferShape   │          │ Stage 1: 核心计算    │
    │ Tiling 切分  │ ───────► │ Stage 2: 输出写回    │
    │ 下发 TilingData│         │                     │
    └──────────────┘          └─────────────────────┘
```

---

## 8. 关键设计总结

| 设计要点 | 实现方式 |
|---------|---------|
| ... | ... |
| ... | ... |
```

## Writing Guidelines

生成代码梳理文档时的注意事项：

1. **完整性**：必须覆盖 op_host 和 op_kernel 的所有文件，不可遗漏
2. **层次性**：从宏观架构到微观实现，逐层展开
3. **准确性**：代码片段、结构体字段、函数签名必须与实际代码一致
4. **可读性**：使用 ASCII 图表、表格、代码块增强可读性
5. **实用性**：重点描述"为什么这样设计"而不仅仅是"代码做了什么"
6. **与详设衔接**：代码梳理的输出要能直接支撑详设文档的编写（特别是第 4 章模板设计）
