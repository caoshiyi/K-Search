# basic_data_api 检索指南

**关联指南**: [api-reference.md](api-reference.md)（Kernel 侧 API）

## 文档概览

Host 侧基础数据结构和接口文档，主要覆盖 `gert` 和 `ge` 命名空间下的类和方法，用于 OpDef、Tiling、InferShape 等 Host 侧开发。

## 何时使用

- 查 Host 侧接口（"TilingContext 有哪些方法？"）
- 查 InferShape 开发（"GetInputShape 怎么用？"）
- 查 TensorDesc 操作（"怎么获取 tensor 的 shape？"）
- 查 OpDef 注册相关数据结构

## 文档结构

```
./references/basic_data_api_docs/
├── INDEX.md
├── gert命名空间/
│   ├── TilingContext/                   # Tiling 上下文
│   ├── TilingData/                      # Tiling 数据结构
│   ├── InferShapeContext/               # 推导 shape
│   ├── InferShapeRangeContext/          # 推导 shape 范围
│   ├── InferDataTypeContext/            # 推导数据类型
│   ├── InferFormatContext/              # 推导格式
│   ├── OpCheckContext/                  # 算子校验
│   ├── Shape/                           # Shape 类
│   ├── Stride/                          # Stride 类
│   ├── StorageShape/                    # 存储 shape
│   ├── StorageFormat/                   # 存储格式
│   ├── Tensor/                          # Tensor 类
│   ├── TensorV2/                        # TensorV2
│   ├── TensorData/                      # Tensor 数据
│   ├── Range/                           # Range 类
│   ├── RuntimeAttrs/                    # 运行时属性
│   ├── CompileTimeTensorDesc/           # 编译期描述符
│   ├── ComputeNodeInfo/                 # 计算节点信息
│   ├── AnchorInstanceInfo/              # 锚点实例
│   ├── ExtendedKernelContext/           # 扩展 kernel 上下文
│   ├── ContinuousVector/               # 连续向量
│   ├── ContextHolder/                   # 上下文持有
│   ├── OpContextBuilderBase/            # 上下文构建器基类
│   ├── OpImplRegisterV2/               # 算子实现注册
│   ├── Op*ContextBuilder/              # 各类上下文构建器
│   └── ExpandDimsType/                  # 维度扩展
├── ge命名空间/
│   ├── Shape/                           # GE Shape
│   ├── Tensor/                          # GE Tensor
│   ├── TensorDesc/                      # GE TensorDesc
│   ├── AscendString/                    # 字符串
│   ├── Allocator/                       # 内存分配
│   ├── MemBlock/                        # 内存块
│   ├── KernelLaunchInfo/                # Kernel 启动信息
│   ├── OpRegistrationData/              # 算子注册数据
│   ├── OpLibRegister/                   # 算子库注册
│   ├── FrameworkRegistry/               # 框架注册
│   ├── IntegerChecker/                  # 整数检查
│   ├── TypeUtils/                       # 类型工具
│   └── OpReceiver/PassReceiver/         # 接收器
└── C接口/                               # C 语言接口
```

## 极速检索示例

### find — 按文件名定位

```bash
# 找 TilingContext 所有方法
find "./references/basic_data_api_docs/gert命名空间/TilingContext/" -name "*.md"

# 找 InferShape 相关
find "./references/basic_data_api_docs/gert命名空间/InferShapeContext/" -name "*.md"
find "./references/basic_data_api_docs/gert命名空间/InferShapeRangeContext/" -name "*.md"

# 找 Shape 类方法
find "./references/basic_data_api_docs/gert命名空间/Shape/" -name "*.md"

# 找 ge 命名空间
find "./references/basic_data_api_docs/ge命名空间/" -maxdepth 1 -type d
```

### grep — 按内容搜索

```bash
# 查 Tiling 相关方法
grep -rl "GetInputShape\|GetOutputShape" "./references/basic_data_api_docs/gert命名空间/"
grep -rl "SetTilingKey\|GetTilingKey" "./references/basic_data_api_docs/"

# 查 Shape 操作
grep -rl "GetDimNum\|SetDimNum\|GetDim" "./references/basic_data_api_docs/gert命名空间/Shape/"

# 查 Tensor 数据获取
grep -rl "GetData\|GetSize\|GetStorageShape" "./references/basic_data_api_docs/gert命名空间/"

# 查属性获取
grep -rl "GetAttr\|RuntimeAttrs" "./references/basic_data_api_docs/gert命名空间/"

# 查特定方法
grep -rl "GetWorkspaceSize\|SetWorkspace" "./references/basic_data_api_docs/"
```

### cat — 直接阅读

```bash
# 看索引
cat "./references/basic_data_api_docs/INDEX.md"

# 读 TilingContext 方法
find "./references/basic_data_api_docs/gert命名空间/TilingContext/" -name "GetInputShape.md" -exec cat {} \;

# 读 Shape 类概述
cat "./references/basic_data_api_docs/gert命名空间/Shape/"*.md | head -50
```

## 组合查询

```bash
# 找 TilingContext 中所有 Get 开头的方法
find "./references/basic_data_api_docs/gert命名空间/TilingContext/" -name "Get*.md"

# 找包含 "OPTIONAL" 的接口文档
grep -rl "OPTIONAL\|optional" "./references/basic_data_api_docs/gert命名空间/"
```
