# 使能Tiling下沉

**页面ID:** atlas_ascendc_10_00014  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00014.html

---

在静态图模式下，可以通过**整图下沉**优化调度性能。将完整的计算图一次性下发至Device侧，后续执行则无需Host参与，由Device自主完成计算，从而减少Host-Device交互开销，提升执行效率。部分算子的Tiling计算依赖运行时输入的具体数值（**Tiling值依赖**），需在执行时动态计算Tiling参数。针对该场景，可采用**Tiling下沉**优化方案：将Tiling计算下沉至Device侧的AI CPU上执行，从而实现计算全程在Device侧高效完成。

> **注意:** 

- 基于新版本CANN包（支持Tiling下沉特性）编译生成的Tiling下沉算子，不兼容旧版CANN（不支持Tiling下沉特性）运行环境。
- 当前仅融合算子（矢量计算和矩阵计算融合）支持进行Tiling下沉。
- Tiling下沉功能仅支持如下产品型号：
>        

  - 
>            Atlas A3 训练系列产品
>           /
>            Atlas A3 推理系列产品
>           
  - 
>            Atlas A2 训练系列产品
>           /
>            Atlas A2 推理系列产品
>           

自定义算子使能Tiling下沉的步骤如下，完整样例请参考[Tiling下沉算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/17_tiling_sink)。

Tiling下沉场景下，算子工程的op_host目录结构如下，Tiling实现文件需单独放在在一个cpp文件中，示例中为add_custom_tiling_sink_tiling.cpp。

```
├── op_host
│   ├── add_custom_tiling_sink.cpp // 算子原型定义、InferShape、InferDataType实现
│   ├── add_custom_tiling_sink_tiling.cpp // Tiling函数实现
│   ├── add_custom_tiling_sink_tiling.h // TilingData结构体定义、Tiling函数声明
│   └── CMakeLists.txt
```

以AddCustom算子为例，讲解关键代码文件的具体实现方法如下：

- TilingData结构体定义、Tiling函数声明头文件add_custom_tiling_sink_tiling.h

  - 进行TilingData结构体的定义
  - 进行Tiling实现函数的声明

```
#ifndef ADD_CUSTOM_TILING_SINK_TILING_H
#define ADD_CUSTOM_TILING_SINK_TILING_H
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingSinkTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustomTilingSink, TilingSinkTilingData)  // Tiling结构体定义

ge::graphStatus AddCustomSinkTilingFunc(gert::TilingContext* context);  // Tiling函数声明
} // namespace optiling
#endif // ADD_CUSTOM_TILING_SINK_TILING_H
```

- 算子原型定义、InferShape、InferDataType实现文件add_custom_tiling_sink.cpp，需包含add_custom_tiling_sink_tiling.h，进行Tiling函数和算子原型定义的关联。
     Tiling下沉仅适用于存在Tiling值依赖（即当InferShape不依赖输入值，仅Tiling计算需要输入值）且算子输入为非Const类型的场景，本示例中的输入y通过ValueDepend配置了非Const类型的Tiling值依赖。

```
#include "add_custom_tiling_sink_tiling.h" // 包含头文件

// ...

namespace ops {
class AddCustomTilingSink : public OpDef {
public:
    explicit AddCustomTilingSink(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL, DependScope::TILING); // 表示输入y为Tiling值依赖
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::AddCustomSinkTilingFunc); // Tiling函数和算子原型定义的关联

        // 请替换为实际的昇腾AI处理器型号
        this->AICore().AddConfig("ascendxxx");
    }
};
OP_ADD(AddCustomTilingSink);
} // namespace ops
```

- Tiling函数的实现文件add_custom_tiling_sink_tiling.cpp

  - Tiling函数中通过判断值依赖InputTensor即输入y的数据指针是否为空指针来确认当前是否处于编译期。Tiling下沉场景，编译期需要为算子分配内存，包括其所需的workspace。为了保证运行时的高效性，编译期应根据算子的执行需求，合理设置所需的workspace最大值，以避免内存不足或浪费。AddCustomTilingSink样例不需要用户workspace，不涉及设置，此处设置为固定值仅作为示例。
  - 完成下沉Tiling函数注册：包含device_op_impl_registry.h头文件，使用宏DEVICE_IMPL_OP_OPTILING进行注册。

```
#include "add_custom_tiling_sink_tiling.h"
#include "register/device_op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 8;
static constexpr uint32_t TILE_NUM = 3;
static constexpr size_t MAX_WORKSPACE_SIZE = 32; // 算子所需用户workspace空间最大值，AddCustomTilingSink算子本身逻辑无需用户workspace空间，此处设置为固定值仅作为示例
static constexpr size_t DEFAULT_WORKSPACE_SIZE = 0;
ge::graphStatus AddCustomSinkTilingFunc(gert::TilingContext *context)
{
    TilingSinkTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = sysWorkspaceSize + DEFAULT_WORKSPACE_SIZE; // 设置运行时workspace大小，此处为系统workspace空间 + 用户workspace空间
    if (context->GetInputTensor(1) != nullptr && context->GetInputTensor(1)->GetData<float>() == nullptr) {
        // 通过判断值依赖InputTensor的Data是否为空指针来确认当前是否处于编译期。
        // Tiling下沉场景，编译期需要为算子分配内存，包括其所需的workspace。为了保证运行时的高效性，编译期应根据算子的执行需求，合理设置所需的workspace最大值，以避免内存不足或浪费。
        currentWorkspace[0] = sysWorkspaceSize + MAX_WORKSPACE_SIZE; // 设置编译期workspace大小，此处为系统workspace空间 + 用户workspace空间最大值
    }
    return ge::GRAPH_SUCCESS;
}
DEVICE_IMPL_OP_OPTILING(AddCustomTilingSink).Tiling(optiling::AddCustomSinkTilingFunc); // 下沉Tiling函数注册
} // namespace optiling
```

- 算子核函数实现
     当前Tiling下沉仅支持融合算子，为了模拟融合算子场景，通过KERNEL_TASK_TYPE_DEFAULT接口强制指定算子在AIC、AIV混合场景运行。

```
extern "C" __global__ __aicore__ void add_custom_tiling_sink(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 将算子强制指定在AIC、AIV混合场景运行，模拟融合算子场景
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

- 修改op_host目录下的编译脚本CMakeLists.txt，添加Tiling下沉编译命令。具体代码如下所示：

```
# 通过ascendc_device_library添加Tiling下沉编译任务
ascendc_device_library( TARGET cust_opmaster # 任务名称，固定为cust_opmaster
                        OPTION SHARED # 动态库（当前仅支持动态库入图下沉）
                        SRC ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_tiling_sink_tiling.cpp ) # Tiling函数实现代码源文件
```
