# SetOrgShape

**页面ID:** atlasascendc_api_07_0651  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0651.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。

#### 函数原型

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
```

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgM | 输入 | 设置原始完整的形状M大小，单位为元素。 |
| orgN | 输入 | 设置原始完整的形状N大小，单位为元素。 |
| orgK | 输入 | 设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。 |
| orgKa | 输入 | 设置矩阵A原始完整的形状Ka大小，单位为元素。 |
| orgKb | 输入 | 设置矩阵B原始完整的形状Kb大小，单位为元素。 |
| orgKc | 输入 | 设置输出C矩阵的N，单位为元素。需要输入B矩阵的N和输出C矩阵的N不一样时可设置，默认为0（即使用B矩阵的N，不进行修改）。 |

#### 约束说明

本接口需要在SetTensorA接口、SetTensorB接口、SetBias接口及SetSingleShape接口前调用。

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
//  复用mm对象
mm.SetOrgShape(orgM, orgN, orgK);
mm.SetTensorA(gm_a1);
mm.SetTensorB(gm_b1);
mm.SetBias(gm_bias1);
mm.IterateAll(gm_c1);
```
