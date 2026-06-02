# HCCL模板参数

**页面ID:** atlasascendc_api_07_0870  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0870.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

创建HCCL对象时需要传入模板参数。

#### 函数原型

Hccl类定义如下，模板参数说明见表1 Hccl类模板参数说明。

```
template <HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU, const auto &config = DEFAULT_CFG>
class Hccl;
```

#### 参数说明

**表1 **Hccl类模板参数说明

| 参数名称 | 描述 |
| --- | --- |
| 支持的服务端类型。HcclServerType类型，定义如下。 对于Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。 对于Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前仅支持HCCL_SERVER_TYPE_AICPU。 ``` enum HcclServerType { HCCL_SERVER_TYPE_AICPU = 0, HCCL_SERVER_TYPE_END  // 预留参数，不支持使用 } ``` |  |
| 用于指定向服务端下发任务的核。HcclServerConfig类型，定义如下，默认值DEFAULT_CFG = {CoreType::DEFAULT, 0}。 ``` struct HcclServerConfig {     CoreType type;  // 向服务端下发任务的核的类型     int64_t blockId;  // 向服务端下发任务的核的ID }; ```  CoreType的定义如下： ``` enum class CoreType: uint8_t {     DEFAULT,  // 表示不指定AIC核或者AIV核     ON_AIV,     // 表示指定为AIV核     ON_AIC     // 表示指定为AIC核 }; ``` |  |

#### 约束说明

无

#### 调用示例

通过如下传入模板参数config的方式创建Hccl类对象，指定HCCL客户端仅在AIV的10号核上发送通信消息给服务端，替代通过调用GetBlockIdx接口的方式指定运行的核。

```
static constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
// 选择AICPU作为服务端
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU, HCCL_CFG> hccl;
```
