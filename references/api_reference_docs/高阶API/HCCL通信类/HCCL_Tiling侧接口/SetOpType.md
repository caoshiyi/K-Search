# SetOpType

**页面ID:** atlasascendc_api_07_10038  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10038.html

---

#### 功能说明

设置通信任务类型。

#### 函数原型

```
uint32_t SetOpType(uint32_t opType)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opType | 输入 | 表示通信任务类型。uint32_t类型。HCCL API提供HcclCMDType枚举定义作为该参数的取值。 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前支持的通信任务类型为HCCL_CMD_ALLREDUCE、HCCL_CMD_ALLGATHER、HCCL_CMD_REDUCE_SCATTER、HCCL_CMD_ALLTOALL、HCCL_CMD_BATCH_WRITE。 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前支持的通信任务类型为HCCL_CMD_ALLREDUCE、HCCL_CMD_ALLGATHER、HCCL_CMD_REDUCE_SCATTER、HCCL_CMD_ALLTOALL、HCCL_CMD_ALLTOALLV、HCCL_CMD_BATCH_WRITE。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

无

#### 调用示例

```
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:doublering";
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, HCCL_REDUCE_RESERVED);
mc2CcTilingConfig.SetReduceType(HCCL_REDUCE_SUM);
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);

algConfig = "AllGather=level0:doublering";
mc2CcTilingConfig.SetGroupName(groupName);
mc2CcTilingConfig.SetOpType(HCCL_CMD_ALLGATHER); // 设置通信任务类型
mc2CcTilingConfig.SetAlgConfig(algConfig);
mc2CcTilingConfig.SetSkipLocalRankCopy(0);
mc2CcTilingConfig.SetSkipBufferWindowCopy(1);
mc2CcTilingConfig.GetTiling(tiling->allGatherTiling);
```
