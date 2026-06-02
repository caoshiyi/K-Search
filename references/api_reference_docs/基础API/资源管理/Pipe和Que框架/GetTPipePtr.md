# GetTPipePtr

**页面ID:** atlasascendc_api_07_0120  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0120.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

创建TPipe对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行TPipe相关的操作。

#### 函数原型

```
__aicore__ inline AscendC::TPipe* GetTPipePtr()
```

#### 约束说明

无

#### 调用示例

如下样例中，在核函数入口处创建TPipe对象，对象初始化会设置全局唯一的TPipe指针。在调用KernelAdd类Init函数时，无需显式传入TPipe指针，而是在函数内直接使用GetTPipePtr获取全局TPipe指针，用来做InitBuffer等操作。

```
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + 2048 * AscendC::GetBlockIdx(), 2048);
        yGm.SetGlobalBuffer((__gm__ half *)y + 2048 * AscendC::GetBlockIdx(), 2048);
        zGm.SetGlobalBuffer((__gm__ half *)z + 2048 * AscendC::GetBlockIdx(), 2048);
        GetTPipePtr()->InitBuffer(inQueueX, 2, 128 * sizeof(half));
        GetTPipePtr()->InitBuffer(inQueueY, 2, 128 * sizeof(half));
        GetTPipePtr()->InitBuffer(outQueueZ, 2, 128 * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        // 算子kernel逻辑
        ...
    }
private:
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
};
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    AscendC::TPipe pipe;
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
```
