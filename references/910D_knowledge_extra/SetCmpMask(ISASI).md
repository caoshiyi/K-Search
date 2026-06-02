# SetCmpMask\(ISASI\)<a name="ZH-CN_TOPIC_0000002523303718"></a>

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

为[Select](Select.md)不传入mask参数的接口设置比较寄存器。配合不同的selMode传入不同的数据。

-   模式0（SELMODE::VSEL\_CMPMASK\_SPR）

    SetCmpMask中传入selMask LocalTensor。

-   模式1（SELMODE::VSEL\_TENSOR\_SCALAR\_MODE）

    SetCmpMask中传入src1 LocalTensor。

-   模式2（SELMODE::VSEL\_TENSOR\_TENSOR\_MODE）

    SetCmpMask中传入LocalTensor，LocalTensor中存放的是selMask的地址。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void SetCmpMask(const LocalTensor<T>& src)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="9.969999999999999%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="90.03%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11492616168"><td class="cellrowborder" valign="top" width="9.969999999999999%" headers="mcps1.2.3.1.1 "><p id="p19933113132715"><a name="p19933113132715"></a><a name="p19933113132715"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="90.03%" headers="mcps1.2.3.1.2 "><p id="p593343122716"><a name="p593343122716"></a><a name="p593343122716"></a>操作数的数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="9.85%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.450000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.7%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="9.85%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.450000000000001%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.7%" headers="mcps1.2.4.1.3 "><p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p82871514192410"><a name="p82871514192410"></a><a name="p82871514192410"></a>LocalTensor的起始地址需要16字节对齐。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

-   当selMode为模式0或模式2时：

    ```
    uint32_t dataSize = 256;
    uint32_t selDataSize = 8;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECIN, 1> inQueueSel;
    TQue<TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
    pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> src1 = inQueueY.AllocTensor<float>();
    uint8_t repeat = 4;
    uint32_t mask = 64;
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
    
    // selMode为模式0（SELMODE::VSEL_CMPMASK_SPR）
    AscendC::SetCmpMask(sel);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(dst, src0, src1, repeat, repeatParams);
    
    // selMode为模式2（SELMODE::VSEL_TENSOR_TENSOR_MODE）
    AscendC::LocalTensor<int32_t> tempBuf;
    #if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)  // cpu调试
    tempBuf.ReinterpretCast<int64_t>().SetValue(0, reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
    event_t eventIdSToV = static_cast<event_t>(AscendC::GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    #else // npu调试
    uint32_t selAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
    AscendC::SetVectorMask<uint32_t>(32);
    AscendC::Duplicate<uint32_t, false>(tempBuf.ReinterpretCast<uint32_t>(), selAddr, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    #endif
    AscendC::SetCmpMask<int64_t>(tempBuf.ReinterpretCast<int64_t>());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select<float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(dst, src0, src1, repeat, repeatParams);
    ```

-   当selMode为模式1时：

    ```
    uint32_t dataSize = 256;
    uint32_t selDataSize = 8;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECIN, 1> inQueueSel;
    TQue<TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
    pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> tmpScalar = inQueueY.AllocTensor<float>();
    
    uint8_t repeat = 4;
    uint32_t mask = 64;
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
    
    // selMode为模式1（SELMODE::VSEL_TENSOR_SCALAR_MODE）
    AscendC::SetVectorMask<uint32_t>(32);
    AscendC::Duplicate<float, false>(tmpScalar, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetCmpMask(tmpScalar);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select(dst, sel, src0, repeat, repeatParams);
    ```

