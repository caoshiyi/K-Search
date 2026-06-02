# api_reference 检索指南

**关联指南**: [basic-knowledge.md](basic-knowledge.md)（编程概念）、[basic-data-api.md](basic-data-api.md)（Host 侧接口）

## 文档概览

Ascend C 算子开发接口文档，涵盖 Kernel 侧所有 API 的函数签名、参数说明、支持的数据类型、约束条件和示例代码。

## 何时使用

- 查 API 函数原型和参数说明（"DataCopy 的参数是什么？"）
- 查某个 API 支持的数据类型（"Sqrt 支持 half 吗？"）
- 查 mask 设置方式（"mask 参数怎么用？"）
- 查同步控制接口（"SetFlag/WaitFlag 怎么用？"）
- 查高阶 API 使用方法（"Matmul API 怎么调用？"）

## 文档结构

```
./references/api_reference_docs/
├── INDEX.md
├── AI_CPU_API/                          # printf, assert, DataStoreBarrier
├── 基础API/
│   ├── Cube分组管理（ISASI）/            # CubeResGroupHandle
│   ├── Kernel_Tiling/                   # GetBlockIdx, GetBlockNum
│   ├── 原子操作/                        # SetAtomicAdd/Max/Min
│   ├── 同步控制/
│   │   ├── 核内同步/                    # SetFlag/WaitFlag/TQueSync/PipeBarrier
│   │   └── 核间同步/                    # SyncAll/CrossCoreSetFlag
│   ├── 数据搬运/                        # DataCopy/DataCopyPad/DataCopyExtParams
│   ├── 矢量计算/
│   │   ├── 基础算术/                    # Add/Sub/Mul/Div/Sqrt/Abs/Exp/Log...
│   │   ├── 比较/                        # Compare/Select/CompareScalar
│   │   ├── 规约/                        # ReduceMin/Max/Sum/WholeReduceSum
│   │   └── 类型转换/                    # Cast/Ceil/Floor/Round
│   ├── 矩阵计算（ISASI）/              # Mmad
│   ├── 标量计算/                        # Muls/Adds/ScalarOps
│   ├── 缓存控制/                        # SetL2CacheHint
│   ├── 系统变量访问/                    # GetBlockIdx/GetBlockNum
│   ├── 调试接口/                        # DumpTensor/printf
│   ├── 资源管理/                        # TBuf/TQue/AllocTensor/FreeTensor
│   └── 工具函数/                        # Duplicate/CreateVecIndex
├── 高阶API/
│   ├── 矩阵计算/                        # Matmul Kernel/Tiling 侧接口
│   ├── 卷积计算/                        # Conv3D
│   ├── 数学计算/                        # Erf/Exp/Log/Pow/Sigmoid/Tanh/Gelu
│   ├── 归约操作/                        # ReduceCustom
│   ├── 排序操作/                        # Sort/TopK
│   ├── 激活函数/                        # Swish/Silu
│   ├── 量化操作/                        # Quant/DeQuant
│   ├── 张量变换/                        # Transpose/Permute
│   ├── 索引计算/                        # Gather/Scatter
│   ├── 数据过滤/                        # NonZero
│   ├── 归一化操作/                      # LayerNorm/GroupNorm
│   └── HCCL通信类/                      # AllReduce/AllGather Kernel/Tiling
├── Utils_API/
│   ├── Tiling模板编程/                  # TilingData 注册宏
│   ├── Tiling数据结构注册/              # BEGIN/END_TILING_DATA_DEF
│   ├── Tiling调测/                      # ContextBuilder/TilingDebug
│   ├── Tiling下沉/                      # GetTilingData
│   ├── 平台信息获取/                    # AscendC::GetSocVersion/PlatformAscendC
│   ├── 原型注册与管理/                  # OpDef 注册
│   ├── RTC/                             # 运行时编译
│   ├── C++标准库/                       # tuple/max/min/enable_if
│   └── log/                             # LogDebug/LogInfo
├── 基础数据结构/
│   ├── LocalTensor/                     # 核心数据容器
│   ├── GlobalTensor/                    # 全局内存访问
│   ├── Layout/                          # 数据布局
│   ├── Coordinate/                      # 坐标系
│   └── TensorTrait/                     # 张量特性
└── 其他数据类型/
    └── TensorDesc/                      # 描述符
```

## 极速检索示例

### find — 按文件名定位

```bash
# 找特定 API
find "./references/api_reference_docs" -name "Sqrt.md"
find "./references/api_reference_docs" -name "DataCopy*.md"
find "./references/api_reference_docs" -name "SetFlag.md"
find "./references/api_reference_docs" -name "Matmul*.md"

# 列出某类别下所有 API
find "./references/api_reference_docs/基础API/矢量计算/" -name "*.md"
find "./references/api_reference_docs/基础API/同步控制/" -name "*.md"
find "./references/api_reference_docs/基础API/数据搬运/" -name "*.md"
find "./references/api_reference_docs/高阶API/矩阵计算/" -name "*.md"
find "./references/api_reference_docs/基础API/资源管理/" -name "*.md"

# 统计某类 API 数量
find "./references/api_reference_docs/基础API/矢量计算/" -name "*.md" | wc -l
```

### grep — 按内容搜索

```bash
# 查函数签名
grep -r "template.*void Sqrt" "./references/api_reference_docs/"
grep -r "void DataCopy" "./references/api_reference_docs/基础API/数据搬运/"
grep -r "void SetFlag" "./references/api_reference_docs/基础API/同步控制/"

# 查某数据类型支持
grep -rl "bfloat16\|bf16" "./references/api_reference_docs/"
grep -rl "half" "./references/api_reference_docs/基础API/矢量计算/"

# 查参数说明
grep -A10 "参数说明" "./references/api_reference_docs/基础API/矢量计算/基础算术/Sqrt.md"

# 查约束条件
grep -r "32字节对齐\|32Byte\|32B对齐" "./references/api_reference_docs/" | head -10

# 查 mask 用法
grep -rl "mask" "./references/api_reference_docs/基础API/矢量计算/" | head -10

# 查 LocalTensor 方法
grep -rl "LocalTensor" "./references/api_reference_docs/基础数据结构/"

# 查 PIPE 类型
grep -r "PIPE_V\|PIPE_MTE\|PIPE_M\|PIPE_S" "./references/api_reference_docs/基础API/同步控制/"
```

### cat — 直接阅读

```bash
# 看索引
cat "./references/api_reference_docs/INDEX.md"

# 读特定 API
cat "./references/api_reference_docs/基础API/矢量计算/基础算术/Sqrt.md"
cat "./references/api_reference_docs/基础API/数据搬运/DataCopy.md"
cat "./references/api_reference_docs/基础API/同步控制/核内同步/SetFlag.md"
cat "./references/api_reference_docs/基础API/资源管理/AllocTensor.md"
```

## 组合查询模式

```bash
# find + cat: 找到文件直接读
find "./references/api_reference_docs" -name "Sqrt.md" -exec cat {} \;

# grep + xargs cat: 搜函数名然后读上下文
grep -rl "void Sqrt" "./references/api_reference_docs/" | xargs cat

# grep 交集: 多关键词
grep -rl "DataCopy" "./references/api_reference_docs/" | xargs grep -l "Pad"

# INDEX 导航: 先看索引找到路径
cat "./references/api_reference_docs/INDEX.md" | grep -i "matmul"
```
