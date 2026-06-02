# GetInputFormat

**页面ID:** atlasopapi_07_00573  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00573.html

---

#### 函数功能

根据算子输入索引获取对应的输入Format指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
StorageFormat *GetInputFormat(const size_t index)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

输入Format指针，index非法时，返回空指针。

关于StorageFormat类型的定义，请参见3.18-StorageFormat。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto format = context->GetInputFormat(0);        // 获取第0个输入的format
  GE_ASSERT_NOTNULL(format);
  // ...
}
```
