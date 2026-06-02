# GetRequiredInputShape

**页面ID:** atlasopapi_07_00231  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00231.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入shape指针。

#### 函数原型

**const StorageShape *GetRequiredInputShape(const size_t ir_index) const**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

指定的输入shape指针，shape中包含了原始shape与运行时shape。关于StorageShape类型的定义，请参见StorageShape。

当输入ir_index非法时，返回空指针。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferShape4ConcatD(TilingContext* context) {
  auto in_shape = context->GetRequiredInputShape(0);
  ...
}
```
