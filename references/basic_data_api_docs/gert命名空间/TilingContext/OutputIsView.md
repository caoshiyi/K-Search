# OutputIsView

**页面ID:** atlasopapi_07_00774  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00774.html

---

#### 函数功能

根据算子输出索引获取对应的输出是否携带非连续信息。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

#### 函数原型

```
bool OutputIsView(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引，从0开始计数。 |

#### 返回值说明

返回输出Tensor是否携带非连续描述信息：

- true：输出Tensor携带非连续描述信息
- false：index非法或者输出Tensor不携带非连续描述信息。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  auto output_is_view_0 = context->InputIsView(0);
  ...
}
```
