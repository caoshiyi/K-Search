# InputIsView

**页面ID:** atlasopapi_07_00771  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00771.html

---

#### 函数功能

根据算子输入索引获取对应的输入是否携带非连续信息。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
bool InputIsView(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

返回输入Tensor是否携带非连续描述信息：

- true：输入Tensor携带非连续描述信息
- false：index非法或者输入Tensor不携带非连续描述信息。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  bool input_is_view_0 = context->InputIsView(0);
  ...
}
```
