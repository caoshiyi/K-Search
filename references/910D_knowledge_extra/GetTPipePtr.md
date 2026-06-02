# GetTPipePtr<a name="ZH-CN_TOPIC_0000002523343802"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

创建[TPipe](TPipe.md)对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行[TPipe](TPipe.md)相关的操作。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline AscendC::TPipe* GetTPipePtr()
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section6191129670"></a>

如下样例中，在核函数入口处创建TPipe对象，对象初始化会设置全局唯一的TPipe指针。在调用KernelAdd类Init函数时，无需显式传入TPipe指针，而是在函数内直接使用GetTPipePtr获取全局TPipe指针，用来做InitBuffer等操作。如需运行，请参考[get\_tpipe\_ptr](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/10_memory_management/get_tpipe_ptr)实现完整的代码。

```
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + 2048 * AscendC::GetBlockIdx(), 2048);
        yGm.SetGlobalBuffer((__gm__ half *)y + 2048 * AscendC::GetBlockIdx(), 2048);
        zGm.SetGlobalBuffer((__gm__ half *)z + 2048 * AscendC::GetBlockIdx(), 2048);
        // 获取全局TPipe指针，使用TPipe的InitBuffer接口给TQue分配内存
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

