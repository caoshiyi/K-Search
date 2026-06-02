# SetSyncResInfos

**页面ID:** atlasopapi_07_00698  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00698.html

---

#### 函数功能

将传入的同步资源信息列表（sync_info_vec）设置到当前对象中，用于配置图执行时的同步资源行为。

#### 函数原型

```
ge::graphStatus SetSyncResInfos(std::vector<SyncResInfo> &sync_info_vec) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| sync_info_vec | 输入 | 同步资源信息列表。std::vector<SyncResInfo>类型。 |

#### 返回值说明

返回graphStatus状态码。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus SetSyncResInfos(ExeResGenerationContext* context) {
  std::vector<SyncResInfo> sync_info_vec;
  ge::graphStatus status= context->SetSyncResInfos(sync_info_vec);
  ...
}
```
