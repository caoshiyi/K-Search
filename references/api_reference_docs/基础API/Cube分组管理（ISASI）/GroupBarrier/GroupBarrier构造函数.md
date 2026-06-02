# GroupBarrier构造函数

**页面ID:** atlasascendc_api_07_0303  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0303.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

创建GroupBarrier对象，通过设置构造函数参数，确定Arrive组的Block个数和Wait组的Block个数。

#### 函数原型

```
template <PipeMode pipeMode>
class GroupBarrier;
__aicore__ inline GroupBarrier(GM_ADDR groupWorkspace, uint32_t arriveSizeIn, uint32_t waitSizeIn);
```

#### 参数说明

**表1 **模板参数说明

| 数据类型 | 说明 |
| --- | --- |
| GroupBarrier发送组同步消息时使用的执行单元，仅支持MTE3_MODE。 ``` enum class PipeMode : uint8_t {    SCALAR_MODE = 0,   MTE3_MODE = 1,   MAX  } ``` |  |

**表2 **GroupBarrier构造函数参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| groupWorkspace | 输入 | 该GroupBarrier在GM上的起始地址，用于存储组同步消息，必须512B对齐。用户自行管理这部分GlobalMemory，包括地址对齐和清零。 |
| arriveSizeIn | 输入 | Arrive组AIV个数。 |
| waitSizeIn | 输入 | Wait组AIV个数。 |

#### 返回值说明

GroupBarrier对象实例。

#### 约束说明

- 使用该接口时，UB空间末尾的1600B被占用。
- 不能和REGIST_MATMUL_OBJ接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST_MATMUL_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

#### 调用示例

```
AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(startAddr, 3, 6);  // 6个AIV等3个AIV Arrive后再开始后续业务，总共需要6*512B地址空间，起始地址为用户指定的startAddr。
```
