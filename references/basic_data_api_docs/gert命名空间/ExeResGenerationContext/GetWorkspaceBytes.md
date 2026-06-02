# GetWorkspaceBytes

**页面ID:** atlasopapi_07_00700  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00700.html

---

#### 函数功能

获取Workspace大小。

#### 函数原型

```
std::vector<int64_t> GetWorkspaceBytes() const
```

#### 参数说明

无

#### 返回值说明

返回Workspace大小配置信息。

#### 约束说明

无

#### 调用示例

```
std::vector<int64_t> GetWorkspaceBytes(ExeResGenerationContext* context) {
  std::vector<int64_t> workspaceList = context->GetWorkspaceBytes();
  // ...
}
```
