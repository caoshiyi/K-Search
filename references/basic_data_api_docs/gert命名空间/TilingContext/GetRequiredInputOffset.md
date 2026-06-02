# GetRequiredInputOffset

**页面ID:** atlasopapi_07_00785  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00785.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入Tensor的offset。

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

#### 函数原型

```
int64_t GetRequiredInputOffset(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

指定的输入Tensor的offset。当输入ir_index非法或Tensor不携带非连续描述信息时，返回-1。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  auto required_input_offset_0 = context->GetRequiredInputOffset(0);
  ...
}
```
