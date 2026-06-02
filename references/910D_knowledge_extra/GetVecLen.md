# GetVecLen<a name="ZH-CN_TOPIC_0000002554343627"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取[RegTensor](RegTensor.md)位宽VL（Vector Length）的大小。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline constexpr uint32_t GetVecLen()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

Vector Length的大小，单位为byte。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section177231425115410"></a>

如下样例通过GetVecLen获取循环迭代次数：

```
template <typename T>
__aicore__ inline void AddCustomImpl(__local_mem__ T *dst, __local_mem__ T *src0, __local_mem__ T *src1,
    uint32_t calCount)
{
    AscendC::MicroAPI::RegTensor<T> reg0;
    AscendC::MicroAPI::RegTensor<T> reg1;
    AscendC::MicroAPI::RegTensor<T> reg2;
    AscendC::MicroAPI::MaskReg mask;
    constexpr uint32_t repeatElm = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTime = AscendC::CeilDivision(calCount, repeatElm);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(calCount);
        AscendC::MicroAPI::LoadAlign(reg0, src0 + i * repeatElm);
        AscendC::MicroAPI::LoadAlign(reg1, src1 + i * repeatElm);
        AscendC::MicroAPI::Add(reg2, reg0, reg1, mask);
        AscendC::MicroAPI::StoreAlign(dst + i * repeatElm, reg2, mask);
    }
}
template <typename T>
__aicore__ inline void AddCustom(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<T> &src1Local, const uint32_t calCount)
{
    VF_CALL<AddCustomImpl<T>>((__local_mem__ T *)dstLocal.GetPhyAddr(), (__local_mem__ T *)src0Local.GetPhyAddr(),
        (__local_mem__ T *)src1Local.GetPhyAddr(), calCount);
}
```

