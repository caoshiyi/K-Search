# GetCmpMask\(ISASI\)<a name="ZH-CN_TOPIC_0000002523303574"></a>

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

此接口用于获取[Compare（结果存入寄存器）](Compare（结果存入寄存器）.md)指令的比较结果。

[Compare（结果存入寄存器）](Compare（结果存入寄存器）.md)指令会将比较后的结果写入CmpMask寄存器中，使用GetCmpMask接口可以获取到CmpMask寄存器的值从而得到Compare的结果。

## 函数原型<a name="section620mcpsimp"></a>

```
template<typename T>
__aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)
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
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="10.08%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.75%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.17%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.75%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="77.17%" headers="mcps1.2.4.1.3 "><p id="p493112487441"><a name="p493112487441"></a><a name="p493112487441"></a><a href="Compare（结果存入寄存器）.md">Compare（结果存入寄存器）</a>指令的比较结果。</p>
<p id="p17287403258"><a name="p17287403258"></a><a name="p17287403258"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p635923282310"><a name="p635923282310"></a><a name="p635923282310"></a>LocalTensor的起始地址需要16字节对齐。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

dst的空间大小不能少于128字节。

## 调用示例<a name="section837496171220"></a>

[Compare（结果存入寄存器）](Compare（结果存入寄存器）.md)指令的结果使用uint8\_t类型数据存储，因此dstLocal使用uint8\_t类型。

```
AscendC::LocalTensor<float> src0Local;
AscendC::LocalTensor<float> src1Local;
AscendC::LocalTensor<uint8_t> dstLocal;
uint64_t mask = 256 / sizeof(float); // 256为每个迭代处理的字节数，结果为64
AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
AscendC::Compare(src0Local, src1Local, AscendC::CMPMODE::LT, mask, repeatParams);
AscendC::GetCmpMask(dstLocal); // mask为0x40, 比较数据类型为float，则每次迭代的32B里只有第7个float数字参与compare
```

