# SetWorkspaceBytes

**页面ID:** atlasopapi_07_00701  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00701.html

---

#### 函数功能

设置Workspace大小。

#### 函数原型

```
void SetWorkspaceBytes(const std::vector<int64_t> &workspace_bytes) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspace_bytes | 输入 | Workspace大小配置信息。 |

#### 约束说明

无

#### 调用示例

```
void SetWorkspaceBytes(ExeResGenerationContext* context) {
  std::vector<int64_t> workspace_bytes;
  context->SetWorkspaceBytes(workspace_bytes);
  // ...
}
```
