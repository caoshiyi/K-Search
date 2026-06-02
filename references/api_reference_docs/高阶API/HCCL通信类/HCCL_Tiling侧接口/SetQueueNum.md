# SetQueueNum

**页面ID:** atlasascendc_api_07_10201  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10201.html

---

#### 功能说明

设置每个向服务端下发任务的核上的BatchWrite通信队列数量。

#### 函数原型

```
uint32_t SetQueueNum(uint16_t num)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| num | 输入 | 表示队列的数量。参与通信的核数*队列数量支持设置的取值范围为[0, 40]，参与通信的核数的设置请参考SetCommBlockNum。 |

#### 返回值说明

- 0表示设置成功。
- 非0表示设置失败。

#### 约束说明

本接口仅在Atlas A3 训练系列产品/Atlas A3 推理系列产品上通信类型为HCCL_CMD_BATCH_WRITE时生效。

#### 调用示例

```
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_BATCH_WRITE;
std::string algConfig = "BatchWrite=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.SetQueueNum(2U);
```
