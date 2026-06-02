# DynamicInputIsView

**页面ID:** atlasopapi_07_00780  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00780.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的动态输入Tensor是否携带非连续描述信息。

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

#### 函数原型

```
bool DynamicInputIsView(const size_t ir_index, const size_t relative_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]。 |

#### 返回值说明

返回指定ir_index的输入Tensor是否携带非连续描述信息：

- true：指定ir_index的输入Tensor携带非连续描述信息。
- false：输入ir_index非法或该INPUT没有实例化或该Tensor未携带非连续描述信息。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  bool dynamic_input_is_view_0 = context->DynamicInputIsView(0);
  ...
}
```
