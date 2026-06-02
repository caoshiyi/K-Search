# PostMessage

**页面ID:** atlasascendc_api_07_0294  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0294.html

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

通过AllocMessage接口获取到消息空间地址msg后，构造消息结构体CubeMsgType，发送该消息。

#### 函数原型

```
template <PipeMode pipeMode = PipeMode::SCALAR_MODE>
__aicore__ inline uint16_t PostMessage(__gm__ CubeMsgType* msg, CubeMsgType& msgInput)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| 用于配置发送消息的执行单元。PipeMode类型，其定义如下： ``` enum class PipeMode : uint8_t {    SCALAR_MODE = 0, // Scalar执行单元往GM上写消息。   MTE3_MODE = 1, // 使用MTE3单元往GM上写消息。   MAX  } ``` |  |

**表2 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| msg | 输入 | 该CubeResGroupHandle中某个任务的消息空间地址。 |
| msgInput | 输入 | 需要发送的消息内容。 |

#### 返回值说明

当前消息空间与该消息队列队首空间的地址偏移。

#### 约束说明

无

#### 调用示例

```
handle.AssignQueue(queIdx);  
auto msgPtr = handle.AllocMessage();        // 获取消息空间指针msgPtr
AscendC::CubeGroupMsgHead headA = {AscendC::CubeMsgState::VALID, 0};
AscendC::CubeMsgBody msgA = {headA, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
auto offset = handle.PostMessage(msgPtr, msgA);           // 在msgPtr指针位置，填充用户自定义的消息结构体，并发送
```
