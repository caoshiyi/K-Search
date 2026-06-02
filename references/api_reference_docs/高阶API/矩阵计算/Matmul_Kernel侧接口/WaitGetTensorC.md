# WaitGetTensorC

**页面ID:** atlasascendc_api_07_0659  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0659.html

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

当使用GetTensorC异步接口将结果矩阵从GM拷贝到UB，且UB后续需要进行Vector计算时，需要调用WaitGetTensorC进行同步。

#### 函数原型

```
__aicore__ inline void WaitGetTensorC()
```

#### 参数说明

无

#### 约束说明

当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

#### 调用示例

```
// 异步模式样例
mm.template Iterate<false>();
// 其他操作
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {   
    mm.template GetTensorC<false>(ubCmatrix); 
    mm.WaitGetTensorC();
    // Vector 操作 
}
```
