# CreateHcomWaitTask

**页面ID:** atlasopapi_07_00674  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00674.html

---

#### 函数功能

创建一个Wait Task，此Task用于阻塞流，当与其有相同group_name的Record Task被执行时，阻塞会被解除。

#### 函数原型

```
static KernelLaunchInfo CreateHcomWaitTask(const gert::ExeResGenerationContext *context, const char *group_name = "group")
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| context | 输入 | GenerateTask函数的入参，保存了算子的基础信息。 |
| group_name | 输入 | Wait Task的分组名字，默认为group，用于与Record Task配套。 |

#### 返回值说明

返回创建出来的Wait Task信息。

#### 约束说明

group_name必须与算子原型中定义的属性一致。例如，某个mc2算子定义了一个属性group_ep，则可以使用group_name为group_ep创建Record任务和Wait任务。

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  // 创建WaitTask
  auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
  wait_task.SetStreamId(attach_stream_id);
  tasks.insert(tasks.begin() + aicore_index, wait_task.Serialize());
  aicore_index++;
  ...
}
```
