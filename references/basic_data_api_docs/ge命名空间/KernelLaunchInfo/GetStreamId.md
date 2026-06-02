# GetStreamId

**页面ID:** atlasopapi_07_00676  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00676.html

---

#### 函数功能

获取当前Task所在流的id。

#### 函数原型

```
uint32_t GetStreamId() const
```

#### 参数说明

无

#### 返回值说明

返回当前Task所在流的id，默认值为0。

异常时，返回int32_max。

#### 约束说明

无

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  auto stream_id = aicore_task.GetStreamId();
  ...
}
```
