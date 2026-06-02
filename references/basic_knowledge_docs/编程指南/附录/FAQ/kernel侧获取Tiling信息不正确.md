# kernel侧获取Tiling信息不正确

**页面ID:** atlas_ascendc_10_0108  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0108.html

---

#### 现象描述

通过算子在kernel侧实现代码中添加PRINTF打印发现kernel侧获取的Tiling信息不正确。

比如下文样例，增加的打印代码如下：

```
PRINTF("tiling_data.totalLength: %d tiling_data.tileNum: %d.\n", tiling_data.totalLength, tiling_data.tileNum);
```

打印的Tiling数据如下，全为0：

```
tiling_data.totalLength: 0 tiling_data.tileNum: 0.
```

#### 问题根因

kernel侧获取Tiling信息不正确的原因一般有以下两种：

- host侧计算Tiling的逻辑不正确
- kernel侧核函数的参数未按照正确顺序填写

#### 处理步骤

1. 参考如下示例，打印TilingData的数据，确认host侧序列化保存的TilingData是否正确。如果此时打印值有误，说明Tiling的计算逻辑可能不正确，需要进一步检查host侧Tiling实现代码，排查计算逻辑是否有误。

```
std::cout<<*reinterpret_cast<uint32_t *>(context->GetRawTilingData()->GetData())<<std::endl; //按照实际数据类型打印TilingData第一个参数值，如需确认其他值，取值指针向后偏移即可
```

2. 如果上一步骤中打印的TilingData正确，需要排查kernel侧核函数的参数是否按照正确顺序填写。

使用msOpGen工具创建算子工程，并基于工程进行kernel侧算子开发时，核函数的定义模板已通过msOpGen工具自动生成，样例如下所示**。**参数按照 “输入、输出、workspace、tiling”的顺序排布。请检查是否调整过参数顺序导致和正确顺序不一致。

```
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);// 获取Tiling参数
    // TODO: user kernel impl
}
```
