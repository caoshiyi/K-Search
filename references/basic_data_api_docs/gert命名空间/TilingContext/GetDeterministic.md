# GetDeterministic

**页面ID:** atlasopapi_07_00535  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00535.html

---

#### 函数功能

获取确定性计算配置选项。

#### 函数原型

**int32_t GetDeterministic() const**

#### 参数说明

无。

#### 返回值说明

0：未开启确定性配置选项。

1：开启确定性配置选项。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  int32_t ret = context->GetDeterministic();
  // ...
}
```
