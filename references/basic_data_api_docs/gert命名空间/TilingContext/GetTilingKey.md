# GetTilingKey

**页面ID:** atlasopapi_07_00235  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00235.html

---

#### 函数功能

获取tiling key。

#### 函数原型

**uint64_t GetTilingKey() const**

#### 参数说明

无

#### 返回值说明

返回tiling key。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_key = context->GetTilingKey();
  // ...
}
```
