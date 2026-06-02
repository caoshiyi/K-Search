# SetBlockDim

**页面ID:** atlasopapi_07_00679  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00679.html

---

#### 函数功能

设置算子BlockDim。

#### 函数原型

```
graphStatus SetBlockDim(uint32_t block_dim)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| block_dim | 输入 | 算子BlockDim。 |

#### 返回值说明

设置成功时返回“ge::GRAPH_SUCCESS”。

关于graphStatus的定义，请参见ge::graphStatus。

#### 约束说明

无

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  aicpu_task.SetBlockDim(4);
  ...
}
```
