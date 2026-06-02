# AIC和AIV独立运行机制

**页面ID:** atlas_ascendc_10_10028  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_10028.html

---

#### 功能介绍

AIC和AIV独立运行机制，又称双主模式。在分离模式下，区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，使能双主模式能够提高Matmul计算性能。默认情况下，双主模式不使能，需要通过MatmulConfig中的enableMixDualMaster参数开启。

#### 使用场景

算子中的矩阵计算和矢量计算相关代码独立运行，不依赖消息驱动时，可以开启双主模式，以提高Matmul计算性能。

#### 约束说明

- 该功能仅支持Norm模板和MDL模板。
- 算子核函数的类型为MIX，同时AIC核数 : AIV核数为1:1。
- 算子核函数的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时使能IBSHARE参数。

- 同一算子中所有Matmul对象的该参数取值必须保持一致。
- A、B、Bias矩阵只支持从Global Memory输入。
- 获取矩阵计算结果只支持调用IterateAll接口输出到GlobalTensor，即计算结果放置于Global Memory的地址，不能调用GetTensorC等接口获取结果。

#### 调用示例

完整的算子样例请参考[使能双主模式的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_mixdualmaster)。

```
// 修改模板参数enableMixDualMaster=true，Norm模板开启双主模式，MDL模板使用GetMDLConfig接口获取模板参数。
constexpr static MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT, false, true/*enableMixDualMaster*/);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;

// 常规Matmul计算
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
