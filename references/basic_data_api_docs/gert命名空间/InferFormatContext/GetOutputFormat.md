# GetOutputFormat

**页面ID:** atlasopapi_07_00585  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00585.html

---

#### 函数功能

根据算子输出索引获取对应的输出Format指针。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
StorageFormat *GetOutputFormat(const size_t index)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引，从0开始计数。 |

#### 返回值说明

输出Format指针，index非法时，返回空指针。

关于StorageFormat类型的定义，请参见StorageFormat。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  auto format = context->GetOutputFormat(0U);        // 获取第0个输出的format
  // ...
}
```
