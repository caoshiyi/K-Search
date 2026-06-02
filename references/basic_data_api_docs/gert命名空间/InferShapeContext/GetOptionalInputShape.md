# GetOptionalInputShape

**页面ID:** atlasopapi_07_00097  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00097.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的可选输入shape指针。

#### 函数原型

```
const Shape *GetOptionalInputShape(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |

#### 返回值说明

返回输入shape的指针，输入ir_index非法，或该输入没有实例化时，返回空指针。

关于Shape类型的定义，请参见Shape。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferShapeForXXX(InferShapeContext *context) {
  auto in_shape = context->GetOptionalInputShape(2);
  // ...
}
```
