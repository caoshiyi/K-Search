# AddInputTd

**页面ID:** atlasascendc_api_07_1017  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1017.html

---

#### 功能说明

为算子增加输入Tensor的描述

#### 函数原型

```
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape)
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape, void* constValues)
ContextBuilder &AddInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
ge::Format storageFormat, gert::StorageShape storageShape, const std::string &filePath)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |
| dtype | 输入 | 算子输入tensor的数据类型 |
| originFormat | 输入 | 算子输入tensor原始格式 |
| storageFormat | 输入 | 算子输入tensor运行时格式 |
| storageShape | 输入 | 算子输入tensor的shape |
| constValues | 输入 | 值依赖场景下该输入tensor需要设置的数据指针。 bfloat16与float16的数据依赖场景，请传入float格式的数据，接口内部自行转换成bfloat16或float16。 |
| filePath | 输入 | 值依赖场景下该输入tensor的bin格式数据文件路径 |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

输入的index需要基于算子IR定义，按照IrInstanceNum声明顺序来排布；

调用AddInputTd前需要调用NodeIoNum与IrInstanceNum接口

#### 调用示例

```
gert::StorageShape x_shape = {{1024, 5120}, {1024, 5120}};
gert::StorageShape expert_tokens_shape = {{16}, {16}};
gert::StorageShape weight1_shape = {{16, 5120, 0}, {16, 5120, 0}};
gert::StorageShape bias1_shape = {{16, 0}, {16, 0}};

std::vector<float> x_const_value (1024 * 5120, 2.f);
std::vector<float> bias_value (16 * 5120, 3.f);
context_ascendc::ContextBuilder builder
(void)builder.NodeIoNum(5, 1) // 声明算子有5个输入，1个输出
    .IrInstanceNum({1, 1, 2, 1, 1}) // 声明index 2的算子tensor有两个dynamic实例
    .SetOpNameType("tmpName", "tmpType")
    .AddInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, x_shape, reinterpret_cast<void *>(x_const_value.data()))  // 内部会将该指针指向的数据转为float16类型
    .AddInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, weight1_shape)
    .AddInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND, expert_tokens_shape, "./expert_tokens_data.bin")   // index2 的 第一个dynamic tensor，值依赖场景传入数据路径
    .AddInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND, expert_tokens_shape, "./expert_tokens_data.bin")   // index2 的 第二个dynamic tensor， 值依赖场景传入数据路径
    .AddInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, bias1_shape)
    .AddInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND, bias2_shape, reinterpret_cast<void*>(bias_value.data()))  // 内部会将该指针指向的数据转为Bf16类型
```
