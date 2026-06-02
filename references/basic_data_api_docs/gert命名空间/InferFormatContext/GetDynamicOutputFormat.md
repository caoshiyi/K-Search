# GetDynamicOutputFormat

**页面ID:** atlasopapi_07_00587  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00587.html

---

#### 函数功能

根据算子原型定义中的输出索引获取对应的动态输出Format指针。

#### 函数原型

```
StorageFormat *GetDynamicOutputFormat(const size_t ir_index, const size_t relative_index)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的索引，从0开始计数。 |
| relative_index | 输入 | 该算子输出实例化后的相对index，例如某个动态输出实例化了3个输出，那么relative_index的有效范围是[0,2]。 |

#### 返回值说明

返回输出Format指针，ir_index或relative_index非法时，返回空指针。

关于StorageFormat类型的定义，请参见StorageFormat。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  auto format = context->GetDynamicOutputFormat(0U, 0U);
  // ...
}
```
