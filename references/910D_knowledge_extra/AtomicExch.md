# AtomicExch<a name="ZH-CN_TOPIC_0000002523343562"></a>

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

在GM内存中执行原子交换操作。具体来说，它读取指定GM地址上的数据，并将新的值存储回同一地址。函数返回旧值。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline T AtomicExch(__gm__ T *address, T value)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="8.63%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="91.36999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="8.63%" headers="mcps1.2.3.1.1 "><p id="p611771320276"><a name="p611771320276"></a><a name="p611771320276"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="91.36999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p1541974719227"><a name="p1541974719227"></a><a name="p1541974719227"></a><span id="ph19303201913016"><a name="ph19303201913016"></a><a name="ph19303201913016"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint32_t/uint64_t</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="8.959999999999999%" id="mcps1.2.4.1.1"><p id="p1663233972817"><a name="p1663233972817"></a><a name="p1663233972817"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.04%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="83%" id="mcps1.2.4.1.3"><p id="p663216395286"><a name="p663216395286"></a><a name="p663216395286"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="8.959999999999999%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>address</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>输入GM的地址。</p>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="8.959999999999999%" headers="mcps1.2.4.1.1 "><p id="p6255648487"><a name="p6255648487"></a><a name="p6255648487"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.4.1.2 "><p id="p182559481483"><a name="p182559481483"></a><a name="p182559481483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="83%" headers="mcps1.2.4.1.3 "><p id="p62551148687"><a name="p62551148687"></a><a name="p62551148687"></a>标量值，支持数据类型和address指向的数据类型保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

GM地址上做原子操作前的数据。

## 约束说明<a name="section123861726193513"></a>

原子操作涉及标量计算，如果标量计算单元和搬运单元（MTE2/MTE3）在读写GM时存在数据依赖，开发者需要根据实际情况插入同步。

## 调用示例<a name="section177231425115410"></a>

```
extern "C" __global__ __aicore__ void atomic_simple_kernel(__gm__ uint8_t* dst， uint32_t dataSize)
{
    // ...
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm), dataSize);
    LocalTensor<T> dstLocal = inQueueX.AllocTensor<T>();
    uint32_t value = 2;
    uint32_t a = AscendC::AtomicExch(reinterpret_cast<__gm__ int32_t *>(dst), value);
    // 先执行完原子操作之后才能进行搬运操作，有数据依赖，需要手动插入MTE2等待Scalar的同步
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>());
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    DataCopy(dstLocal, dst_global, dataSize);
    // ...
}
```

假设上述函数在3个核上执行，核1、核2、核3依次调度，结果示例如下：

```
原GM数据dst: [1,1,1,1,1,...,1] 
核1：
原子计算后GM数据dst: [2,1,1,1,1,...,1] 
返回值 a: 1
核2：
原子计算后GM数据dst: [2,1,1,1,1,...,1] 
返回值 a: 2
核3：
原子计算后GM数据dst: [2,1,1,1,1,...,1] 
返回值 a: 2
```

