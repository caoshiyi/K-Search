# 使用标准C++语法定义Tiling结构体

**页面ID:** atlas_ascendc_10_00024  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00024.html

---

#### 具体步骤

在定义Tiling结构体时，可以使用标准C++语法定义一个**POD类型（Plain Old Data）**，即与C语言兼容的数据类型。具体步骤如下。完整样例请参考[标准C++语法定义Tiling结构体样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomMultiCore)。

1. 使用C++语法定义Tiling结构体。

该结构体定义所在的头文件应放置在算子工程的op_kernel目录下。由于只有该目录下的文件会被打包进算子包，供在线编译场景中使用，若将文件放置在其他目录中，可能导致在线编译因找不到相关文件而失败。

       用户在使用高阶API的Tiling结构体时，通过AscendC::tiling命名空间引用"kernel_tiling/kernel_tiling.h"中预定义的Tiling结构体，如下代码所示。

```
#ifndef MATMUL_CUSTOM_TILING_H
#define MATMUL_CUSTOM_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"    // for TCubeTiling

struct MatmulCustomTilingData {
    uint64_t localMemSize;
    AscendC::tiling::TCubeTiling cubeTilingData;
};
#endif  // MATMUL_CUSTOM_TILING_H
```

2. Host侧Tiling函数中对Tiling结构体赋值。

  - 需要包含Tiling结构体定义头文件。
  - 通过GetTilingData获取Tiling结构体指针，并对其成员变量进行赋值。

```
#include "../op_kernel/matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件
...

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ...
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    ...
    // 获取Tiling结构体指针
    MatmulCustomTilingData *tiling = context->GetTilingData<MatmulCustomTilingData>();
    // 对tiling的成员变量赋值
    if (cubeTiling.GetTiling(tiling->cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    tiling->localMemSize = localMemSize;
    ...
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

3. Kernel侧注册Tiling结构体，解析Tiling数据至TilingData结构并使用。

  - 需要包含Tiling结构体定义头文件。
  - 通过REGISTER_TILING_DEFAULT或者REGISTER_TILING_FOR_TILINGKEY注册Tiling结构体；通过GET_TILING_DATA解析Tiling数据至TilingData结构并使用。其中REGISTER_TILING_DEFAULT同时也用于标识使用标准C++语法定义TilingData结构体。

```
#include "kernel_operator.h"
#include "matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernel<half, half, float, float> matmulKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData); // Initialize the matmul object.
    matmulKernel.Init(a, b, bias, c, workspace, tilingData.localMemSize, tilingData.cubeTilingData);
    ...
}
```

#### 使用标准C++语法定义Tiling结构体的优势

相比较使用BEGIN_TILING_DATA_DEF等宏进行定义的方式，该方式不仅更符合C++开发者的开发习惯，并且提供了强大的灵活性。

- 支持bool类型，支持数组、结构体数组及列表初始化。

```
class A {
public:
    bool xxx;
    uint32_t xxx[2][128] = {0};
};

class B {
public:
    bool xxx = false;
    uint8_t xxx[2][2]{0};
    A[10];
};
```

- 不同算子可以支持定义同名但结构不同的Tiling结构体，通过算子引用对应的头文件即可实现区分。这种方式允许每个算子使用符合自身需求的Tiling结构定义，而不会与其他算子产生冲突。

相比之下，使用BEGIN_TILING_DATA_DEF等宏方式定义同名但结构不同的Tiling结构体时，由于这些结构体会被注册到全局的Tiling结构体管理变量中，可能导致后续通过结构体名称访问时，无法准确获取当前算子实际使用的Tiling结构体，从而引发未定义行为。

       算子A：

```
class TilingData {
public:
    uint32_t length;
};
```

算子B：

```
class TilingData {
public:
    uint32_t length;
    uint16_t coreNum;
};
```

- 支持自定义Tiling赋值，用户可以通过访问Tiling结构体成员变量直接赋值，或自定义Tiling赋值函数（宏定义方式下，用户仅可通过框架生成的set_xx/get_xx方法赋值/访问）
      Tiling结构体定义：

```
class TilingData {
public:
    uint32_t xxx1;
    uint32_t xxx2;
    uint8_t xxx3;
    bool xxx4;
};
```

Host侧Tiling函数：

```
#include "../op_kernel/xxx_custom_tiling.h"  // 包含Tiling结构体定义头文件
...

namespace optiling {
static void ComputeTiling(TilingData* tiling, ...)
{
    // 计算Tiling逻辑
    ...
    tiling->xxx1 = xxx;
    tiling->xxx2 = xxx;
    tiling->xxx3 = xxx;
    tiling->bool = xxx;
}
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{    
    ...
    TilingData *tiling = context->GetTilingData<TilingData>();
    ...
    ComputeTiling(tiling, ...)
    ...

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 使用约束

使用标准C++语法定义Tiling结构体时存在如下约束限制：

- Tiling结构体内不支持定义成员函数，因为成员函数存在Device侧和Host侧的差异（Device侧的函数需要__aicore__修饰符），而Tiling结构体Device侧和Host侧共用，所以会在编译或执行时出现问题：

```
class TilingData {
public:
    uint32_t xxx;

    __aicore__ funcA() { ... }  // 错误，host侧编译时不支持__aicore__修饰符，会出现编译错误
    void func() { ... }         // 错误，device侧缺少__aicore__修饰符，无法执行
};
```

- Tiling结构体成员变量不支持指针、引用类型，此类数据类型会导致Host侧到Device侧数据解析异常：

```
class TilingData {
public:
    uint32_t* totalLength; // 指针场景不支持，Host无法传递指针到Device
    uint32_t& tileNum;       // 引用场景不支持，Host无法传递指针到Device
};
```

- Tiling结构体仅支持POD类型，没有虚函数、虚继承等面向对象特性，也不支持模板类：

```
class A {
public:
    uint32_t totalLength;
    uint32_t tileNum;
};
class B: public A {
public:
    uint32_t xxx;
    uint32_t xxx;
};
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 错误用法
    B *tiling = context->GetTilingData<A>(); // 不支持，会触发未知问题
    // 正确用法
    B *tiling = context->GetTilingData<B>();
    ......
    return ge::GRAPH_SUCCESS;
}
```

- GetTilingData获取的Tiling不包含初值，需显式赋值或在Tiling结构体定义并调用Tiling赋值函数；

```
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData *tiling = context->GetTilingData<TilingData>(); //获取Tiling结构体，此时totalLength、tileNum为0，并不会带入初始值
    ......
    // 需显式赋值
    tiling->totalLength = totalLength;  // 赋值Tiling结构体成员变量
    tiling->tileNum = TILE_NUM;         // 赋值Tiling结构体成员变量
    ......
    return ge::GRAPH_SUCCESS;
}
```

#### 如何将宏定义Tiling结构体修改为标准C++语法

本节介绍如何将使用BEGIN_TILING_DATA_DEF等宏进行定义的方式改造成使用标准C++语法的方式。

1. **首先**将之前位于op_host目录下的Tiling结构体定义头文件移至op_kernel目录下，内容前后对比如下，**注意此时包含的头文件变化，不需要再包含宏定义相关的头文件**。

**表1 **两种方式对比

| 宏定义方式 | 标准C++语法定义方式 |
| --- | --- |
| ``` #include "register/tilingdata_base.h" #include "tiling/tiling_api.h" // TCubeTiling结构体通过宏定义  namespace optiling { BEGIN_TILING_DATA_DEF(MatmulCustomTilingData) TILING_DATA_FIELD_DEF(uint64_t, localMemSize); TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData); END_TILING_DATA_DEF; REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulCustomTilingData) } // namespace optiling ``` | ``` #include <cstdint> #include "kernel_tiling/kernel_tiling.h" // TCubeTiling结构体通过C++语法定义  struct MatmulCustomTilingData {     uint64_t localMemSize;     AscendC::tiling::TCubeTiling cubeTilingData; }; ``` |

2. **然后**修改Host侧的Tiling函数实现，此时对Tiling结构体的成员变量赋值无需使用宏定义生成的set方法，而是使用用户熟悉的C++指针赋值方式。

**表2 **两种方式对比

| 宏定义方式 | 标准C++语法定义方式 |
| --- | --- |
| ``` namespace optiling { static ge::graphStatus TilingFunc(gert::TilingContext *context) {     ...     MultiCoreMatmulTiling cubeTiling(ascendcPlatform);     ...     MatmulCustomTilingData tiling;     if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) { // Get matmul tiling.         return ge::GRAPH_FAILED;     }      uint64_t localMemSize;     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);     tiling.set_localMemSize(localMemSize);  // 需要使用宏定义方式生成的set方法      ...     // 需要将局部变量保存至context上下文     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());       ...      return ge::GRAPH_SUCCESS; } } // namespace optiling ``` | ``` #include "../op_kernel/matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件 ...  namespace optiling { static ge::graphStatus TilingFunc(gert::TilingContext *context) {     ...     MultiCoreMatmulTiling cubeTiling(ascendcPlatform);     ...     MatmulCustomTilingData *tiling = context->GetTilingData<MatmulCustomTilingData>();     if (cubeTiling.GetTiling(tiling->cubeTilingData) == -1) {         return ge::GRAPH_FAILED;     }      uint64_t localMemSize;     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);     tiling->localMemSize = localMemSize;  // 使用用户友好的C++指针方式赋值成员变量      ...      return ge::GRAPH_SUCCESS; } } // namespace optiling ``` |

3. **最后**，在Kernel 函数入口处新增REGISTER_TILING_DEFAULT调用，用于注册Tiling结构体。该注册操作的作用是：告知框架用户已使用标准 C++ 语法定义Tiling结构体，并明确其类型，以便框架在进行Tiling数据解析时能够正确识别和使用该结构体。

```
#include "matmul_custom_tiling.h"
...

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulCustomTilingData);  // 新增REGISTER_TILING_DEFAULT调用注册Tiling结构体
    ...
}
```
