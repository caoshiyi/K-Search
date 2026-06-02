# SetKernelMode

**页面ID:** atlasascendc_api_07_1211  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1211.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

针对分离模式，CPU调测时，设置内核模式为单AIV模式，单AIC模式或者MIX模式，以分别支持单AIV矢量算子，单AIC矩阵算子，MIX混合算子的CPU调试。不调用该接口的情况下，默认为MIX模式。为保证算子代码在多个硬件平台兼容，耦合模式下也可以调用，该场景下接口不会生效，不影响正常调试。

#### 函数原型

```
void SetKernelMode(KernelMode mode)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 内核模式，针对AIC，AIV，MIX算子的CPU调试，参数取值分别为AIC_MODE，AIV_MODE，MIX_MODE。                                                                                                                           ``` enum class KernelMode {     MIX_MODE = 0,     AIC_MODE,     AIV_MODE }; ``` |  |  |

#### 调用示例

```
int32_t main(int32_t argc, char* argv[])
{
    ...
#ifdef ASCENDC_CPU_DEBUG
    ...
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, blockDim, x, y, z); // use this macro for cpu debug
    ...
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
#else
    ...
#endif
    return 0;
}
```
