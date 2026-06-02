# SetCommBlockNum

**页面ID:** atlasascendc_api_07_10202  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10202.html

---

#### 功能说明

设置参与BatchWrite通信的核数。

#### 函数原型

```
uint32_t SetCommBlockNum(uint16_t num)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| num | 输入 | 表示核的数量。 |

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
mc2CcTilingConfig.SetCommBlockNum(24U);
```
