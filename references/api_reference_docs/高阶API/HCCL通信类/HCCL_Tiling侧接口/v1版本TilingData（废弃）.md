# v1版本TilingData（废弃）

**页面ID:** atlasascendc_api_07_0888  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0888.html

---

> **注意:** 

该结构体废弃，并将在后续版本移除，请不要使用该结构体。无需直接对该结构体中的成员进行设置，统一使用HCCL Tiling提供的接口设置即可。

对于本节介绍的TilingData结构体，当构建通信计算融合算子时，通算融合算子的TilingData结构体中，计算Tiling结构体部分必须在本节的通信Tiling结构体后追加。

对于v1和v2两个版本的TilingData，Tiling结构体的第一个uint32_t字段用于区分两个版本，即v1版本的preparePosition字段，v2版本的version字段。若使用v2版本的Tiling结构体，则必须设置version=2；若使用v1版本的Tiling结构体，则设置preparePosition=1。用户使用任意版本的TilingData时，都必须严格按照对应版本的Tiling结构体，将其作为算子TilingData结构体的组成部分。

#### 功能说明

AI CPU启动下发通信任务前，需获取固定的通信配置Mc2Msg。在算子实现中，由Tiling组装通信配置项，通过配置固定参数和固定参数顺序的Tiling Data，将通信配置信息在调用AI CPU通信接口时传递给AI CPU。

#### 参数说明

**表1 **Mc2Msg参数说明

| 参数名 | 描述 |
| --- | --- |
| preparePosition | 设置服务端组装任务的方式，用户需要在Tiling中显式赋值，uint32_t类型，当前支持的取值如下：          1：AI CPU与AI Core通过通信任务机制实现消息传递和任务下发；由AI Core侧通过消息通知时设置为1，即算子中使用HCCL时设置为1。 |
| sendOff | 预留参数，不可配置。 |
| recvOff | 预留参数，不可配置。 |
| tailSendOff | 预留参数，不可配置。 |
| tailRecvOff | 预留参数，不可配置。 |
| sendCnt | 预留参数，不可配置。 |
| recvCnt | 预留参数，不可配置。 |
| tailSendCnt | 预留参数，不可配置。 |
| tailRecvCnt | 预留参数，不可配置。 |
| totalCnt | 预留参数，不可配置。 |
| turnNum | 预留参数，不可配置。 |
| tailNum | 预留参数，不可配置。 |
| stride | 预留参数，不可配置。 |
| workspaceOff | 预留参数，不可配置。 |
| notifyOff | 预留参数，不可配置。 |
| notifyBeginCnt | 预留参数，不可配置。 |
| notifyEndCnt | 预留参数，不可配置。 |
| useBufferType | 设置通信算法获取输入数据的位置，uint8_t类型，参数取值如下：                     - 0：默认值，默认通信输入不放在windows中，其中windows为其他卡可访问的共享缓冲区。           - 1：通信输入不放在windows中，当前该参数取值1与取值0的功能一致。           - 2：通信输入放在windows中，仅适用于AllReduce算法。 |
| funID | 预留参数，不可配置。 |
| dataType | 预留参数，不可配置。 |
| groupNum | 预留参数，不可配置。 |
| reuseMode | 预留参数，不可配置。 |
| commType | 预留参数，不可配置。 |
| reduceOp | 预留参数，不可配置。 |
| commOrder | 预留参数，不可配置。 |
| waitPolicy | 预留参数，不可配置。 |
| rspPolicy | 预留参数，不可配置。 |
| exitPolicy | 预留参数，不可配置。 |
| commAlg | 设置具体通信算法，用户需要在Tiling中显式赋值，uint8_t类型，当前支持的取值如下：          1：FullMesh算法，即NPU之间的全连接，任意两个NPU之间可以直接进行数据收发。详细的算法内容可参见集合通信算法介绍。 |
| taskType | 预留参数，不可配置。 |
| debugMode | 预留参数，不可配置。 |
| stepSize | 预留参数，不可配置。 |
| sendArgIndex | 预留参数，不可配置。 |
| recvArgIndex | 预留参数，不可配置。 |
| commOutArgIndex | 预留参数，不可配置。 |
| hasCommOut | 本卡的通信算法的计算结果是否输出到recvBuf（目的数据buffer地址）。仅AllGather算法与AlltoAll算法支持配置该参数。uint8_t类型，参数取值如下：                     - 0：不输出本卡通信算法的计算结果。在无需输出通信结果时，配置参数值为0，此时不会拷贝本卡的通信结果数据，可提升算子性能。例如，在8卡场景下，本卡只取其他卡的部分数据，这时可配置本参数为0。           - 1：输出本卡通信算法的计算结果。 |
| reserve | 保留字段。 |
| reserve2 | 保留字段。 |

#### 约束说明

- 算子的Tiling Data结构需要按顺序完整包含Mc2Msg参数。
- AI CPU需获取固定数据结构的通信配置，算子注册Tiling Data时保持该结构的一致性。
- 
        Atlas A3 训练系列产品
       /
        Atlas A3 推理系列产品
       暂不支持该版本TilingData。

#### 调用示例

以自定义算子AllGatherMatmulCustom为例，如下为该算子的算子原型，"gather_out"为通信任务AllGather的输出。

```
[
    {
        "op": "AllGatherMatmulCustom",
        "input_desc": [
            {
                "name": "x1",
                "param_type": "required",
                "format": [
                    "ND",
		    "ND"
                ],
                "type": [
                    "float16",
                    "bfloat16"
                ]
            },
            {
                "name": "x2",
                "param_type": "required",
                "format": [
                    "ND",
		    "ND"
                ],
                "type": [
                    "float16",
                    "bfloat16"
                ]
            },
            {
                "name": "bias",
                "param_type": "optional",
                "format": [
                    "ND",
		    "ND"
                ],
                "type": [
                    "float16",
                    "bfloat16"
                ]
            }
        ],
        "output_desc":[
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND",
		    "ND"
                ],
                "type": [
                    "float16",
                    "bfloat16"
                ]
            },
            {
                "name": "gather_out",
                "param_type": "required",
                "format": [
                    "ND",
		    "ND"
                ],
                "type": [
                    "float16",
                    "bfloat16"
                ]
            }
        ],
        "attr": [
            {
                "name": "group",
                "type": "string",
                "default_value":"",
                "param_type":"required"
            },
            {
                "name": "rank_size",
                "type": "int",
                "default_value":0,
                "param_type":"optional"
            },
            {
                "name": "is_gather_out",
                "type": "bool",
                "default_value":true,
                "param_type":"optional"
            }
        ]
    }
]
```

算子的Tiling Data结构需要按顺序完整包含Mc2Msg参数，如下为算子Tiling Data代码示例。

```
// 声明Mc2Msg结构
BEGIN_TILING_DATA_DEF(Mc2Msg)
    TILING_DATA_FIELD_DEF(uint32_t, preparePosition);
    TILING_DATA_FIELD_DEF(uint32_t, sendOff); 
    TILING_DATA_FIELD_DEF(uint32_t, recvOff);
    TILING_DATA_FIELD_DEF(uint32_t, tailSendOff);
    TILING_DATA_FIELD_DEF(uint32_t, tailRecvOff);
    TILING_DATA_FIELD_DEF(uint64_t, sendCnt);
    TILING_DATA_FIELD_DEF(uint32_t, recvCnt);
    TILING_DATA_FIELD_DEF(uint32_t, tailSendCnt);
    TILING_DATA_FIELD_DEF(uint32_t, tailRecvCnt);
    TILING_DATA_FIELD_DEF(uint32_t, totalCnt);
    TILING_DATA_FIELD_DEF(uint32_t, turnNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, workspaceOff);
    TILING_DATA_FIELD_DEF(uint32_t, notifyOff);
    TILING_DATA_FIELD_DEF(uint16_t, notifyBeginCnt);
    TILING_DATA_FIELD_DEF(uint16_t, notifyEndCnt);
    TILING_DATA_FIELD_DEF(uint8_t, useBufferType);
    TILING_DATA_FIELD_DEF(uint8_t, funID);
    TILING_DATA_FIELD_DEF(uint8_t, dataType);
    TILING_DATA_FIELD_DEF(uint8_t, groupNum);
    TILING_DATA_FIELD_DEF(uint8_t, reuseMode);
    TILING_DATA_FIELD_DEF(uint8_t, commType);
    TILING_DATA_FIELD_DEF(uint8_t, reduceOp);
    TILING_DATA_FIELD_DEF(uint8_t, commOrder);
    TILING_DATA_FIELD_DEF(uint8_t, waitPolicy);
    TILING_DATA_FIELD_DEF(uint8_t, rspPolicy);
    TILING_DATA_FIELD_DEF(uint8_t, exitPolicy);
    TILING_DATA_FIELD_DEF(uint8_t, commAlg);
    TILING_DATA_FIELD_DEF(uint8_t, taskType);
    TILING_DATA_FIELD_DEF(uint8_t, debugMode);
    TILING_DATA_FIELD_DEF(uint8_t, stepSize);
    TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex);
    TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex);
    TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex);
    TILING_DATA_FIELD_DEF(uint8_t, hasCommOut);
    TILING_DATA_FIELD_DEF(uint8_t, reserve);
    TILING_DATA_FIELD_DEF(uint32_t, reserve2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MsgOp, Mc2Msg)

BEGIN_TILING_DATA_DEF(AllGatherMatmulCustomTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
END_TILING_DATA_DEF;
```

```
// 配置Mc2Msg
AllGatherMatmulCustomTilingData tiling;
tiling.msg.set_preparePosition(1);
tiling.msg.set_commAlg(1);
tiling.msg.set_useBufferType(1);
tiling.msg.set_hasCommOut(1);
```
