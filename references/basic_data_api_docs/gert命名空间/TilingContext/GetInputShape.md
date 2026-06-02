# GetInputShape

**页面ID:** atlasopapi_07_00224  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00224.html

---

#### 函数功能

根据算子输入索引获取对应的输入shape指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

**const StorageShape *GetInputShape(const size_t index) const**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

指定的输入shape指针，输入shape中包含了原始shape与运行时shape信息。关于StorageShape类型的定义，请参见StorageShape。

当输入index非法时返回空指针。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  ...
}
```
