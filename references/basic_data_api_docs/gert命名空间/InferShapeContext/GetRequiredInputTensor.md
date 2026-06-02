# GetRequiredInputTensor

**页面ID:** atlasopapi_07_00099  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00099.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入tensor指针。

#### 函数原型

```
const Tensor *GetRequiredInputTensor(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

返回指定输入的tensor指针，若输入的ir_index非法，返回空指针。

关于Tensor类型的定义，请参见Tensor。

#### 约束说明

仅在设置数据依赖时可以获取tensor的数据地址。如果输入没有被设置为数据依赖，调用此接口获取tensor时，只能在tensor中获取到正确的shape、format、datatype信息，无法获取到真实的tensor数据地址（获取到的地址为nullptr）。

#### 调用示例

```
ge::graphStatus InferShapeForXXX(InferShapeContext *context) {
  auto in_shape = context->GetRequiredInputTensor(2);
  // ...
}
```
