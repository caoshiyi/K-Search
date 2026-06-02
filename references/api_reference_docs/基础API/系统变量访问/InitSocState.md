# InitSocState

**页面ID:** atlasascendc_api_07_00094  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00094.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

由于AI Core上存在一些全局状态，如原子累加状态、Mask模式等，在实际运行中，这些值可以被前序执行的算子修改而导致计算出现不符合预期的行为，在静态Tensor编程的场景中用户必须在Kernel入口处调用此函数来初始化AI Core状态 。

#### 函数原型

```
__aicore__ inline void InitSocState()
```

#### 参数说明

无

#### 约束说明

不调用该接口，在部分场景下可能导致计算结果出现精度错误或者卡死等问题。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // 初始化全局状态寄存器（在TPipe框架编程中，初始化过程由TPipe完成，无需开发者关注；静态Tensor编程方式中需要开发者手动调用）
    AscendC::InitSocState();
    KernelAdd op;
    op.Init(x, y, z);
    op.Process(); 
}
```
