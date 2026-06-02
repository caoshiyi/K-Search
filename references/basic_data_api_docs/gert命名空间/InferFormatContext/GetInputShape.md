# GetInputShape

**页面ID:** atlasopapi_07_00577  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00577.html

---

#### 函数功能

根据算子输入索引获取对应的输入Shape指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
const Shape *GetInputShape(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

输入Shape指针，index非法时，返回空指针。

关于Shape类型的定义，请参见Shape。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto shape= context->GetInputShape(0);        // 获取第0个输入的shape
  GE_ASSERT_NOTNULL(shape);
  // ...
}
```
