# GenerateTask

**页面ID:** atlasopapi_07_00713  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00713.html

---

#### 函数功能

GenerateTask阶段具体Task的生成和处理。

#### 函数原型

```
OpImplRegisterV2 &GenerateTask(OpGenTaskKernelFunc gen_task_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 待注册的OpGenTaskKernelFunc函数。 OpGenTaskKernelFunc类型定义如下： ``` using OpGenTaskKernelFunc = UINT32 (*)(const ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks); ``` |  |  |

#### 返回值说明

返回算子的OpImplRegisterV2对象本身，该对象新增注册OpGenTaskKernelFunc函数。

#### 约束说明

无
