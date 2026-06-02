# GetOptionalInputShape

**页面ID:** atlasopapi_07_00229  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00229.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的可选输入shape指针。

#### 函数原型

**const StorageShape *GetOptionalInputShape(const size_t ir_index) const**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 可选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

指定的输入shape指针，shape中包含了原始shape与运行时shape。关于StorageShape类型的定义，请参见StorageShape。

当输入ir_index非法，或该INPUT没有实例化时，返回空指针。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4ConcatD(TilingContext* context) {
  const Shape *shape_bias = context->GetOptionalInputShape(kBatchMatMulBiasIdx);
  ...
}
```
