# GetProgramCounter(ISASI)

**页面ID:** atlasascendc_api_07_0279  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0279.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。

#### 函数原型

```
__aicore__ inline int64_t GetProgramCounter()
```

#### 参数说明

无

#### 返回值说明

返回int64_t类型的程序计数器指针。

#### 约束说明

无

#### 调用示例

```
int64_t pc = AscendC::GetProgramCounter();
```
