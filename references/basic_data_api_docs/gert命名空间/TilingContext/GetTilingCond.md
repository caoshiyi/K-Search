# GetTilingCond

**页面ID:** atlasopapi_07_00239  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00239.html

---

#### 函数功能

获取GetTilingCond中设置的tiling cond。

#### 函数原型

**int32_t GetTilingCond() const**

#### 参数说明

无。

#### 返回值说明

若返回值大于等于0，代表此tiling cond为有效的tiling cond。

若返回值为-1，代表此tiling cond为无效的tiling cond。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_cond = context->GetTilingCond();
  // ...
}
```
