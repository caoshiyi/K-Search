# GetBlockDim

**页面ID:** atlasopapi_07_00678  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00678.html

---

#### 函数功能

获取算子BlockDim。

#### 函数原型

```
uint32_t GetBlockDim() const
```

#### 参数说明

无

#### 返回值说明

返回此算子Task的BlockDim值，默认值为0。

异常时返回int32_max。

#### 约束说明

无

#### 调用示例

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  ...
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  auto block_dim = aicore_task.GetBlockDim();
  ...
}
```
