# SetAttachedStreamInfos

**页面ID:** atlasopapi_07_00695  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00695.html

---

#### 函数功能

为当前算子附着一组Stream信息，用于指定该算子在哪个执行流（Stream）上运行。

#### 函数原型

```
ge::graphStatus SetAttachedStreamInfos(std::vector<StreamInfo> &stream_info_vec) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| stream_info_vec | 输入 | 执行流信息。std::vector<StreamInfo>类型。 |

#### 返回值说明

返回graphStatus状态码。

#### 约束说明

无

#### 调用示例

```
ge::graphStatus SetAttachedStreamInfos(ExeResGenerationContext* context) {
  std::vector<StreamInfo> stream_info_vec;
  auto status = context->SetAttachedStreamInfos(stream_info_vec);
  ...
}
```
