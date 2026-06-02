# GetDynamicInputFormat

**页面ID:** atlasopapi_07_00576  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00576.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的动态输入Format指针。

#### 函数原型

```
StorageFormat *GetDynamicInputFormat(const size_t ir_index, const size_t relative_index)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的index。 |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个动态输入实例化了3个输入，那么relative_index的有效范围是[0,2]。 |

#### 返回值说明

输入Format指针，ir_index或relative_index非法时，返回空指针。

关于StorageFormat类型的定义，请参见StorageFormat。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto format = context->GetDynamicInputFormat(1U, 0U);
  GE_ASSERT_NOTNULL(format);
  // ...
}
```
