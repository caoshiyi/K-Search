# AddOutputTd

**页面ID:** atlasascendc_api_07_1018  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1018.html

---

#### 功能说明

为算子增加输入Tensor的描述。

#### 函数原型

```
ContextBuilder &AddOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat, gert::StorageShape storageShape)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引，从0开始计数。 |
| dtype | 输入 | 算子输出tensor的数据类型 |
| originFormat | 输入 | 算子输出tensor原始格式 |
| storageFormat | 输入 | 算子输出tensor运行时格式 |
| storageShape | 输入 | 算子输出tensor的shape |

#### 返回值说明

当前ContextBuilder的对象

#### 约束说明

调用AddOutputTd前需要调用NodeIoNum与IrInstanceNum接口

#### 调用示例

```
gert::StorageShape x_shape = {{1024, 5120}, {1024, 5120}};
gert::StorageShape output_shape = {{1024, 5210}, {1024, 5210}};
context_ascendc::ContextBuilder builder;
(void)builder
    .NodeIoNum(1, 1)
    .IrInstanceNum({1})
    .SetOpNameType("tmpName", "tmpType")
    .AddInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, x_shape)
    .AddOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, output_shape)
```
