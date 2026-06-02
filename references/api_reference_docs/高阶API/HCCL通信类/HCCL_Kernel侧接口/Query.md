# Query

**页面ID:** atlasascendc_api_07_0879  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0879.html

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

查询handleId对应的通信任务已经完成的轮次，最多返回repeat轮。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。

#### 函数原型

```
__aicore__ inline int32_t Query(HcclHandle handleId)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 对应通信任务的标识ID，只能使用Prepare原语接口的返回值。 ``` using HcclHandle = int8_t; ``` |  |  |

#### 返回值说明

- 返回handleId对应的通信任务已执行的次数，最大值为repeat。
- 当执行异常时，返回-1。

#### 约束说明

- 调用本接口前确保已调用过InitV2和SetCcTilingV2接口。
- 入参handleId只能使用Prepare原语对应接口的返回值。
- 本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。

#### 调用示例

```
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>();  // AscendC自定义算子kernel中，通过此方式获取HCCL context
hccl.InitV2(contextGM, &tilingData);
auto ret = hccl.SetCcTiling(offsetof(ReduceScatterCustomTilingData, mc2CcTiling));
if (ret != HCCL_SUCCESS) {
  return;
}
if (AscendC::g_coreType == AIC) {
    auto repeat = 10;
    HcclHandle handleId = hccl.ReduceScatter(sendBuf, recvBuf, 100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, repeat);
    hccl.Commit(handleId ); // 通知服务端可以执行上述的ReduceScatter任务
    int32_t finishedCount = hccl.Query(handleId);
    while (hccl.Query(handleId) < repeat) {} // 等待查询到handleId对应的通信任务执行repeat次
    hccl.Finalize(); // 后续无其他通信任务，通知服务端执行上述ReduceScatter任务之后即可以退出
}
```
