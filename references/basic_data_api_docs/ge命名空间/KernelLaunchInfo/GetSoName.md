# GetSoName

**页面ID:** atlasopapi_07_00682  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00682.html

---

#### 函数功能

获取AI CPU任务的so名称。

#### 函数原型

```
const char *GetSoName() const
```

#### 参数说明

无

#### 返回值说明

获取成功时返回算子的so名称。

获取失败时，返回nullptr。

#### 约束说明

只有AI Cpu任务可以获取到so名称。

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  // 获取到libcc_kernel.so
  auto so_name = aicpu_task.GetSoName();
  ...
}
```
