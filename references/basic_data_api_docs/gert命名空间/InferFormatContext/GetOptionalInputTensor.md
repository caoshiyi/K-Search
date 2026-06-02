# GetOptionalInputTensor

**页面ID:** atlasopapi_07_00583  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00583.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的可选输入Tensor指针。

#### 函数原型

```
const Tensor *GetOptionalInputTensor(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

输入Tensor指针，index非法或该输入没有实例化时，返回空指针。

关于Tensor类型的定义，请参见Tensor。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto data = context->GetOptionalInputTensor(1U)->GetData<uint8_t>();
  EXPECT_EQ(data[0], 85);
  // ...
}
```
