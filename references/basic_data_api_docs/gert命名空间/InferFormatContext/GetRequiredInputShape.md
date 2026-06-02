# GetRequiredInputShape

**页面ID:** atlasopapi_07_00578  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00578.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入Shape指针。

#### 函数原型

```
const Shape *GetRequiredInputShape(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

输入Shape指针，ir_index非法时，返回空指针。

关于Shape类型的定义，请参见Shape。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto shape= context->GetRequiredInputShape(0);        // 获取第0个输入的shape
  GE_ASSERT_NOTNULL(shape);
  // ...
}
```
