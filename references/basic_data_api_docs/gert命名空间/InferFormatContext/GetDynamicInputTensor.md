# GetDynamicInputTensor

**页面ID:** atlasopapi_07_00584  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00584.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的动态输入Tensor指针。

#### 函数原型

```
const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 输入在算子IR原型定义中的索引，从0开始计数。 |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个动态输入实例化了3个输入，那么relative_index的有效范围是[0,2]。 |

#### 返回值说明

输入Tensor指针，ir_index或relative_index非法时，返回空指针。

关于Tensor类型的定义，请参见Tensor。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  auto axes_tensor_10 = context->GetDynamicInputTensor(1U, 0U);
  auto axes_tensor_11 = context->GetDynamicInputTensor(1U, 1U);
  ...
}
```
