# TSCM输入的矩阵乘

**页面ID:** atlas_ascendc_10_10024  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_10024.html

---

#### 功能介绍

TSCM表示L1 Buffer空间对应的逻辑内存，L1 Buffer相关内容见存储单元，开发者可以自行管理TSCM以高效利用硬件资源。比如，开发者可缓存一份TSCM数据，在不同使用场景中灵活配置为Matmul操作的A矩阵、B矩阵或Bias偏置矩阵，实现内存复用与计算效率优化。在TSCM输入场景，用户管理整块TSCM内存空间，Matmul直接使用传入的TSCM内存地址，不进行Global Memory到TSCM的数据搬入。

#### 使用场景

用户需要自定义数据搬入到TSCM及自定义管理的场景，即需要自定义实现数据搬入功能，如非连续搬入或对搬入数据进行预处理等。用户通过自定义管理TSCM可灵活配置MTE2流水，实现跨Matmul对象的全局DoubleBuffer，MTE2相关内容见搬运单元。

#### 约束说明

设置为TSCM输入的矩阵必须在TSCM中全载，全载即全部的矩阵数据同时搬入及保持在TSCM中。

#### 调用示例

完整的算子样例请参考[自定义数据来源为GM的TSCM输入的Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/blob/master/examples/matrix/matmul_tscm)、[BatchMatmul自定义TSCM输入的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/batch_matmul_tscm)。

```
TQue<TPosition::A1, 1> scm; // 队列逻辑位置A1，队列深度为1
pipe->InitBuffer(scm, 1, tiling.M * tiling.Ka * sizeof(A_T)); 
// A_TYPE的TPosition为TSCM， B_TYPE的TPosition为GM
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm1;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
mm1.Init(&tiling);
// 自定义A矩阵GM到TSCM的搬运
auto scmTensor = scm.AllocTensor<A_T>();
DataCopy(scmTensor, gm_a, tiling.M * tiling.Ka);
scm.EnQue(scmTensor);
LocalTensor<A_T> scmLocal = scm.DeQue<A_T>();
// A矩阵设置为TSCM输入，B矩阵为GM输入
mm1.SetTensorA(scmLocal);
mm1.SetTensorB(gm_b);
mm1.SetBias(gm_bias);
mm1.IterateAll(gm_c);
scm.FreeTensor(scmLocal);
```
