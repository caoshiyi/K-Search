# CreateCubeResGroup

**页面ID:** atlasascendc_api_07_0300  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0300.html

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

快速创建CubeResGroupHandle对象，内部完成消息队列空间和同步事件分配。推荐使用该接口，避免使用CubeResGroupHandle的构造函数创建对象，出现不同对象的消息队列空间冲突、同步事件错误等情况。

#### 函数原型

```
template <int groupID, class MatmulApiType, template <class, class> class CallBack, typename CubeMsgType>
__aicore__ inline CubeResGroupHandle<CubeMsgType> CreateCubeResGroup(KfcWorkspace& desc, uint8_t blockStart, uint8_t blockSize, uint8_t msgQueueSize, GM_ADDR tiling)
```

#### 参数说明

**表1 **模板参数说明

| 参数 | 说明 |
| --- | --- |
| groupID | 用于表示Group的编号，int32取值范围。 |
| MatmulApiType | 定义的AIC计算对象类型。 |
| CallBack | 回调函数类，需要实现Init和Call两个接口。 |
| CubeMsgType | 用户自定义的消息结构体。 |

**表2 **接口参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| desc | 输入 | KfcWorkspace，用于维护消息队列空间。 |
| blockStart | 输入 | 该CubeResGroupHandle在AIV视角下的起始AIC对应的序号，即AIC起始序号 * 2。例如，如果AIC起始序号为0，则填入0 * 2；如果为1，则填入1 * 2。 |
| blockSize | 输入 | 该CubeResGroupHandle在AIV视角下分配的Block个数，即实际的AIC个数*2。 |
| msgQueueSize | 输入 | 该CubeResGroupHandle分配的消息队列总数。 |
| tiling | 输入 | AIC核计算所需tiling信息的地址。 |

#### 返回值说明

CubeResGroupHandle对象实例。

#### 约束说明

- 假设芯片的AIV核数为x，那么blockStart + blockSize <= x - 1, msgQueueSize <= x。
- 每个AIC至少被分配1个msgQueue。
- blockStart和blockSize必须为偶数。
- 使用该接口，UB空间末尾的1600B + sizeof(CubeMsgType)将被占用。
- 1个AIC只能属于1个CubeGroupHandle，即多个CubeGroupHandle的[blockStart / 2, blockStart / 2 + blockSize / 2]区间不能重叠。
- 不能和REGIST_MATMUL_OBJ接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST_MATMUL_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

#### 调用示例

```
auto handle = AscendC::CreateCubeResGroup<GROUPID, MatmulApiType, MyCallbackFunc, CubeMsgBody> (desc, BLOCKSTART, BLOCKSIZE, MSGQUEUESIZE, tilingGM);
```
