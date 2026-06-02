# Init

**页面ID:** atlasascendc_api_07_0630  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0630.html

---

#### 产品支持情况

| 产品 | Tiling参数传入栈地址的接口 | Tiling参数传入GM地址的接口 |
| --- | --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ | √ |
| Atlas 200I/500 A2 推理产品 | √ | x |
| Atlas 推理系列产品AI Core | √ | x |
| Atlas 推理系列产品Vector Core | x | x |
| Atlas 训练系列产品 | x | x |

#### 功能说明

Init主要用于对Matmul对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考Matmul Tiling侧接口。

开发者可以先通过REGIST_MATMUL_OBJ不传入Tiling参数对单个Matmul对象进行初始化，后续通过Init接口单独传入Tiling参数，对Matmul对象中的Tiling数据进行调整。比如，Tiling参数可变的场景下，可以通过多次调用Init来重新设置Tiling参数。

不需要Tiling变更的场景下，推荐使用REGIST_MATMUL_OBJ传入Tiling参数进行初始化。

#### 函数原型

- Tiling参数传入栈地址

```
__aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
```

- Tiling参数传入GM地址

```
__aicore__ inline void Init(const __gm__ TCubeTiling* gmCubeTiling, TPipe* tpipe = nullptr)
```

#### 参数说明

**表1 **Tiling参数传入栈地址接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cubeTiling | 输入 | Matmul Tiling参数，TCubeTiling结构体定义请参见表1 TCubeTiling结构说明。 Tiling参数可以通过host侧GetTiling接口获取，并传递到kernel侧使用。在kernel侧调用GET_TILING_DATA实现将Tiling参数搬运到AI Core内的栈空间中，本接口传入Tiling参数中TCubeTiling结构体的栈地址。 |
| tpipe | 输入 | Tpipe对象。 |

**表2 **Tiling参数传入GM地址接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gmCubeTiling | 输入 | Matmul Tiling参数，该参数指向gm上的一块内存地址，其中的数据类型是TCubeTiling结构体，TCubeTiling结构体定义请参见表1 TCubeTiling结构说明。 Tiling参数可以通过host侧GetTiling接口获取，并传递到kernel侧使用。在kernel侧调用GET_TILING_DATA_PTR_WITH_STRUCT获取gm上Tiling参数的指针，本接口传入Tiling参数中TCubeTiling结构体的GM地址。 |
| tpipe | 输入 | Tpipe对象。 |

#### 约束说明

- Tiling参数传入栈地址的接口：

无。

- Tiling参数传入GM地址的接口：

  - 仅支持Matmul Tiling参数的部分常量化场景。
  - 不支持CPU域调试。

#### 调用示例

- Tiling参数传入栈地址

```
GET_TILING_DATA(tilingData, tiling);
// ...
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&(tiling.cubeTilingData));
```

- Tiling参数传入GM地址

  - 纯Cube模式

```
#define ASCENDC_CUBE_ONLY

GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
// ...
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&(tilingDataPtr->cubeTilingData));
```

  - MIX模式

```
GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// ...
// MIX模式下，只调用REGIST_MATMUL_OBJ接口，传入Tiling参数的GM地址，不需调用Init接口
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &(tilingDataPtr->cubeTilingData));
```
