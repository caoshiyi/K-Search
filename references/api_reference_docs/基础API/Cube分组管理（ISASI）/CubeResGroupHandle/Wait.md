# Wait

**页面ID:** atlasascendc_api_07_0297  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0297.html

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

在调用PostMessage或PostFakeMessage后，查询该消息是否已被AIC处理完。

#### 函数原型

```
template <bool sync = true>
__aicore__ inline bool Wait(uint16_t offset)
```

#### 参数说明

**表1 **模板参数说明

| 参数 | 说明 |
| --- | --- |
| sync | 查询消息时，程序的运行是否需要等待。参数取值如下： - true，必须等到AIC处理完该消息后，程序才可以继续运行。- false，仅查询AIC是否处理完该消息。 |

**表2 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| offset | 输入 | 消息空间地址偏移量，通过PostMessage或者PostFakeMessage的返回值获取。 |

#### 返回值说明

- true：当前消息已被AIC处理完。
- false：当前消息未被AIC处理完。

#### 约束说明

无

#### 调用示例

```
auto msgPtr = handle.AllocMessage();        // 在msgPtr指针这个位置，可以发送一个新消息
AscendC::CubeGroupMsgHead headA = {AscendC::CubeMsgState::VALID, 0};
AscendC::CubeMsgBody msgA = {headA, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
auto offset = handle.PostMessage(msgPtr, msgA);           // 在msgPtr指针位置，填充用户自定义的消息结构体，并发送
bool waitState = handle.template Wait<true>(offset);      // 等待AIC处理完msgA
// 假消息场景
auto msgFakePtr = handle.AllocMessage();
offset = handle.PostFakeMsg(msgFakePtr);
bool waitState = handle.template Wait<true>(offset); // 等待AIC处理完假消息msgFake
```
