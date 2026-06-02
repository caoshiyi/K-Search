# SetTilingCond

**页面ID:** atlasopapi_07_00238  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00238.html

---

#### 函数功能

设置tiling cond。tiling cond是一个整型值，用于选择算子实际使用的Tiling实现。

#### 函数原型

**ge::graphStatus SetTilingCond(int32_t tiling_cond)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_cond | 输入 | 需要设置的tiling cond。 |

#### 返回值说明

设置成功时返回“ge::GRAPH_SUCCESS”。

关于graphStatus的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ret = context->SetTilingCond(10);
  // ...
}
```
