# RequiredInputIsView

**页面ID:** atlasopapi_07_00783  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00783.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入Tensor是否携带非连续描述信息。

> **注意:** 

该接口为预留接口，为后续的功能做保留，当前版本暂不支持。

#### 函数原型

```
bool RequiredInputIsView(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

返回指定ir_index的输入Tensor是否携带非连续描述信息：

- true：指定ir_index的输入Tensor携带非连续描述信息。
- false：指定ir_index非法或该INPUT没有实例化或该Tensor未携带非连续描述信息。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus TilingForMul(TilingContext *context) {
  bool required_input_is_view_0 = context->RequiredInputIsView(0);
  ...
}
```
