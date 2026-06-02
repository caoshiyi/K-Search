# Async

**页面ID:** atlasascendc_api_07_00163  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00163.html

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

Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）。

#### 函数原型

```
template <EngineType engine, auto funPtr, class... Args>
__aicore__ void Async(Args... args)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| 引擎模式，参数取值分别为AIC、AIV。 ``` enum class EngineType : int32_t {     AIC = 1, // 仅AIC     AIV = 2  // 仅AIV }; ``` |  |
| funPtr | 函数指针，指定要执行的函数，函数签名和参数类型由class... Args决定。 |
| class... Args | 可变参数模板，表示函数参数的类型列表，用于传递给funPtr。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| Args... args | 输入 | 与class... Args对应的参数列表，表示传递给funPtr的实际参数。 |

#### 约束说明

无

#### 调用示例

```
extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    Async<EngineType::AIC, aicOperation>(a, b, bias, c, workspace, tiling, &pipe);
    Async<EngineType::AIV, aivOperation>(c, tiling, &pipe);

}
```

配套的AIC侧和AIV侧执行的函数示例：

```
// 其他代码逻辑
    ...
__aicore__ inline void aicOperation(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, pipe);
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process(pipe);
}

__aicore__ inline void aivOperation(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
    LeakyReluKernel<float> leakyReluKernel;
    leakyReluKernel.Init(c, tiling, pipe);
    leakyReluKernel.Process(pipe);
}
    ...
 // 其他代码逻辑
```
