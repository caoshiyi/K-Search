# GetInputOffset

**页面ID:** atlasopapi_07_00773  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00773.html

---

#### 函数功能

根据算子输入索引获取对应的输入offset。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

#### 函数原型

```
int64_t GetInputOffset(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

指定的输入offset。当输入index非法或者Tensor不携带非连续描述信息时返回-1。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  auto offset_0 = context->GetInputOffset(0);
  ...
}
```
