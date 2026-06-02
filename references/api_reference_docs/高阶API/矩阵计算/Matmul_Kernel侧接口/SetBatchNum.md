# SetBatchNum

**页面ID:** atlasascendc_api_07_0648  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0648.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

在不改变Tiling的情况下，重新设置多Batch计算的Batch数。

#### 函数原型

```
__aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchA | 输入 | 设置的一次计算的A矩阵Batch数。 |
| batchB | 输入 | 设置的一次计算的B矩阵Batch数。 |

#### 约束说明

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 本接口仅支持在纯Cube模式（只有矩阵计算）下调用。

#### 调用示例

```
//  纯cube模式
#define ASCENDC_CUBE_ONLY
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false, LayoutMode::NORMAL> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NORMAL> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType, false, LayoutMode::NORMAL> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType> biasType;
AscendC::Matmul<aType, bType, cType, biasType> mm1;
mm1.SetTensorA(gm_a, isTransposeAIn);
mm1.SetTensorB(gm_b, isTransposeBIn);
if(tiling.isBias) {
    mm1.SetBias(gm_bias);
}
mm1.SetBatchNum(batchA, batchB);
// 多batch Matmul计算
mm1.IterateBatch(gm_c, false, 0, false);
```
