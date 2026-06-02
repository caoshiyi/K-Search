# CeilDivision<a name="ZH-CN_TOPIC_0000002523344964"></a>

## 产品支持情况<a name="section14940119141919"></a>

<a name="table7940171951915"></a>
<table><thead align="left"><tr id="row16940319191910"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p129401619121917"><a name="p129401619121917"></a><a name="p129401619121917"></a><span id="ph5940319101919"><a name="ph5940319101919"></a><a name="ph5940319101919"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p4940171931911"><a name="p4940171931911"></a><a name="p4940171931911"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row09408199199"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p194181911915"><a name="p194181911915"></a><a name="p194181911915"></a><span id="ph12941111911195"><a name="ph12941111911195"></a><a name="ph12941111911195"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p794117190197"><a name="p794117190197"></a><a name="p794117190197"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section725016594193"></a>

计算两个整数num1和num2相除后向上取整结果。

## 函数原型<a name="section1225085919197"></a>

```
__aicore__ constexpr inline int32_t CeilDivision(int32_t num1, int32_t num2)
```

## 参数说明<a name="section72501459191914"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row52838432246"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p1122192003011"><a name="p1122192003011"></a><a name="p1122192003011"></a>num1</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p11416232133016"><a name="p11416232133016"></a><a name="p11416232133016"></a>参数1，被除数。</p>
</td>
</tr>
<tr id="row1673605372520"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p1173716533257"><a name="p1173716533257"></a><a name="p1173716533257"></a>num2</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p19500754133217"><a name="p19500754133217"></a><a name="p19500754133217"></a>参数2，除数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section192502596197"></a>

两个整数相除的向上取整结果。

## 约束说明<a name="section10250859101910"></a>

-   当num2为0时，结果为0。
-   该接口仅支持在num1和num2为正数场景下使用。

## 调用示例<a name="section0250195981918"></a>

本示例中使用CeilDivision计算迭代次数 repeatTimes，通过对数据量count与单次处理数据量进行向上取整除法，确保所有数据（包括尾块）均被完整处理。

```
template <typename T>
__aicore__ inline void AddCustomImpl(__local_mem__ T *dst, __local_mem__ T *src0, __local_mem__ T *src1,
    uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTime = AscendC::CeilDivision(count, oneRepeatSize);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(calCount);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0 + i * oneRepeatSize );
        AscendC::MicroAPI::LoadAlign(srcReg1, src1 + i * oneRepeatSize );
        AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dst + i * repeatElm, reg2, mask);
    }
}
```

