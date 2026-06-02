# AssignQueue

**页面ID:** atlasascendc_api_07_0292  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0292.html

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

用于AIV绑定CubeResGroupHandle中某一个消息队列的序号。

#### 函数原型

```
__aicore__ inline void AssignQueue(uint8_t queueIdIn)
```

#### 参数说明

**表1 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| queueIdIn | 输入 | 在CubeResGroupHandle中消息队列的序号。 |

#### 约束说明

- queueIdIn小于CubeResGroupHandle的消息队列总数msgQueueSize，即取值范围[0，msgQueueSize-1]。
- CubeResGroupHandle中的每一个消息队列都需要进行绑定，且不可重复绑定。

#### 调用示例

```
AscendC::KfcWorkspace desc(workspaceGM); // 用户自行管理的workspace指针。
uint8_t blockStart = 0;
uint8_t blockSize = 12;
uint8_t msgQueueSize = 48;
auto handle = AscendC::CreateCubeResGroup<GROUPID, MatmulApiType, MyCallbackFunc, CubeMsgBody> (desc, blockStart, blockSize, msgQueueSize, tilingGM);
// 当前总计有48个Block，每个Block的视角下，都与handle里的msgQueue进行了绑定，msgQueue每个Queue都被绑定，且没有重复绑定的情况。
auto queIdx = AscendC::GetBlockIdx();
handle.AssignQueue(queIdx);
```
