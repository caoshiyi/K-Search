# SetStreamId

**页面ID:** atlasopapi_07_00677  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00677.html

---

#### 函数功能

设置Task的流id。

#### 函数原型

```
void SetStreamId(uint32_t stream_id)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| stream_id | 输入 | 流id。 |

#### 约束说明

无

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
  wait_task.SetStreamId(attach_stream_id);
  ...
}
```
