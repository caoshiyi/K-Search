# GetKernelName

**页面ID:** atlasopapi_07_00683  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00683.html

---

#### 函数功能

获取AI CPU任务的Kernel入口函数名字。

#### 函数原型

```
const char *GetKernelName() const
```

#### 参数说明

无

#### 返回值说明

获取成功时返回算子的Kernel入口函数名字。

获取失败时，返回nullptr。

#### 约束说明

只有AI Cpu任务可以获取到Kernel入口函数名字。

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  // 获取到RunAicpuKfcSrvLaunch
  auto so_name = aicpu_task.GetKernelName();
  ...
}
```
