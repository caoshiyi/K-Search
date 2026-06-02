# GetInputTensor

**页面ID:** atlasopapi_07_00581  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00581.html

---

#### 函数功能

根据算子输入索引获取对应的输入Tensor指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
const Tensor *GetInputTensor(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 输入索引，从0开始计数。 |

#### 返回值说明

输入Tensor指针，index非法时，返回空指针。

关于Tensor类型的定义，请参见Tensor。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto data = context->GetInputTensor(1U)->GetData<uint8_t>();
  EXPECT_EQ(data[0], 85);
  // ...
}
```
