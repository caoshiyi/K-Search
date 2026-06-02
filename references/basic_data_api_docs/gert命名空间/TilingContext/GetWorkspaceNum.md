# GetWorkspaceNum

**页面ID:** atlasopapi_07_00245  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00245.html

---

#### 函数功能

获取workspace个数。

#### 函数原型

**size_t ****GetWorkspaceNum() const**

#### 参数说明

无。

#### 返回值说明

workspace的个数。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto ws_num = context->GetWorkspaceNum();
  // ...
}
```
