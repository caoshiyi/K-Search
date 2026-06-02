# Unsqueeze<a name="ZH-CN_TOPIC_0000002523343670"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622020982912"><a name="p1622020982912"></a><a name="p1622020982912"></a><span id="ph1522010992915"><a name="ph1522010992915"></a><a name="ph1522010992915"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将dstReg中数据根据mask进行解压缩。解压缩方式：dstReg中第0个元素置为0，dstReg中的第i个元素等于mask中从第0个到第\(i-1\)个元素中1的数量。mask最高位被忽略不参与统计。以uint8\_t类型为例，算法逻辑表示如下：

```
dstReg[0] = 0;
for(int i = 1; i < AscendC::GetVecLen() / sizeof(uint8_t); i++){
dstReg[i] = mask[i - 1] ? (dstReg[i - 1] + 1) : dstReg[i - 1];
}
```

其中AscendC::GetVecLen\(\) / sizeof\(uint8\_t\)为dstReg包含的元素个数。

## 定义原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U>
__simd_callee__ inline void Unsqueeze(U& dstReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.970000000000002%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.03%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.970000000000002%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.03%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>目的操作数的数据类型。</p>
<p id="p1474319251999"><a name="p1474319251999"></a><a name="p1474319251999"></a><span id="ph191588264914"><a name="ph191588264914"></a><a name="ph191588264914"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t。</p>
</td>
</tr>
<tr id="row2013785594119"><td class="cellrowborder" valign="top" width="18.970000000000002%" headers="mcps1.2.3.1.1 "><p id="p141384557419"><a name="p141384557419"></a><a name="p141384557419"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.03%" headers="mcps1.2.3.1.2 "><p id="p1513845554113"><a name="p1513845554113"></a><a name="p1513845554113"></a>目的操作数的RegTensor类型，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  函数参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数和源操作数。</p>
<p id="p1818424418593"><a name="p1818424418593"></a><a name="p1818424418593"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
</td>
</tr>
<tr id="row17332152321719"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1538745616196"><a name="p1538745616196"></a><a name="p1538745616196"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p73331523101710"><a name="p73331523101710"></a><a name="p73331523101710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1559991025517"><a name="p1559991025517"></a><a name="p1559991025517"></a>mask用于提供dstReg解压缩信息。</p>
<p id="p911252842816"><a name="p911252842816"></a><a name="p911252842816"></a>类型为MaskReg。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section177921451558"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template<typename T>
__simd_vf__ inline void UnsqueezeVF(__ubuf__ T* dstAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    mask = AscendC::MicroAPI::CreateMask<T>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::Unsqueeze(dstReg, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

