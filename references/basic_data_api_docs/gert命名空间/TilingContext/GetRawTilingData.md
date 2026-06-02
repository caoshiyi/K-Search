# GetRawTilingData

**页面ID:** atlasopapi_07_00243  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00243.html

---

#### 函数功能

获取无类型的tiling data指针。

#### 函数原型

**TilingData *GetRawTilingData()**

#### 参数说明

无。

#### 返回值说明

tiling data指针，失败时返回空指针。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_data = context->GetRawTilingData();
  // ...
}
```
