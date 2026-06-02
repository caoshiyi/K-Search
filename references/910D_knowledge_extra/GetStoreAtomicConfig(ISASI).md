# GetStoreAtomicConfig\(ISASI\)<a name="ZH-CN_TOPIC_0000002523343706"></a>

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

获取原子操作使能位与原子操作类型的值，详细说明见[表1](SetStoreAtomicConfig(ISASI).md#zh-cn_topic_0235751031_table33761356)。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>atomicType</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p178192610120"><a name="p178192610120"></a><a name="p178192610120"></a>原子操作使能位。</p>
<p id="p112647578426"><a name="p112647578426"></a><a name="p112647578426"></a>0：无原子操作</p>
<p id="p138283312433"><a name="p138283312433"></a><a name="p138283312433"></a>1：使能原子操作，进行原子操作的数据类型为float</p>
<p id="p624819319433"><a name="p624819319433"></a><a name="p624819319433"></a>2：使能原子操作，进行原子操作的数据类型为half</p>
<p id="p441815408439"><a name="p441815408439"></a><a name="p441815408439"></a>3：使能原子操作，进行原子操作的数据类型为int16_t</p>
<p id="p8754174934319"><a name="p8754174934319"></a><a name="p8754174934319"></a>4：使能原子操作，进行原子操作的数据类型为int32_t</p>
<p id="p14302659144317"><a name="p14302659144317"></a><a name="p14302659144317"></a>5：使能原子操作，进行原子操作的数据类型为int8_t</p>
<p id="p437111044410"><a name="p437111044410"></a><a name="p437111044410"></a>6：使能原子操作，进行原子操作的数据类型为bfloat16_t</p>
</td>
</tr>
<tr id="row1441613274210"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p14168210429"><a name="p14168210429"></a><a name="p14168210429"></a>atomicOp</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p54161122420"><a name="p54161122420"></a><a name="p54161122420"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p54166204211"><a name="p54166204211"></a><a name="p54166204211"></a>原子操作类型。</p>
<p id="p13921939194420"><a name="p13921939194420"></a><a name="p13921939194420"></a>0：求和操作</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

此接口需要与[SetStoreAtomicConfig\(ISASI\)](SetStoreAtomicConfig(ISASI).md)配合使用，用以获取原子操作使能位与原子操作类型的值。

## 调用示例<a name="section837496171220"></a>

```
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
uint16_t type = 0;       // 原子操作使能位
uint16_t op = 0;         // 原子操作类型
AscendC::GetStoreAtomicConfig(type, op);
```

