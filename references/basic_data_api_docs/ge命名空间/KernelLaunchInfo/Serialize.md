# Serialize

**页面ID:** atlasopapi_07_00675  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00675.html

---

#### 函数功能

将KernelLaunchInfo序列化成数据流。

#### 函数原型

```
std::vector<uint8_t> Serialize()
```

#### 参数说明

无

#### 返回值说明

返回序列化后的数据流。

#### 约束说明

无

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  // 创建WaitTask
  auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
  wait_task.SetStreamId(attach_stream_id);
  // 序列化
  tasks.insert(tasks.begin() + aicore_index, wait_task.Serialize());
  aicore_index++;
  ...
}
```
