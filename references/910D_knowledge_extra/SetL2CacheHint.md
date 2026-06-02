# SetL2CacheHint<a name="ZH-CN_TOPIC_0000002554424605"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p820164051310"><a name="p820164051310"></a><a name="p820164051310"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

设置GlobalTensor是否使能L2 Cache，默认使能L2 Cache。

## 函数原型<a name="section620mcpsimp"></a>

```
template<CacheRwMode rwMode = CacheRwMode::RW>
__aicore__ inline void SetL2CacheHint(CacheMode mode);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="27.839999999999996%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="72.16%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18835145716587"><td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a><strong id="b53621266338"><a name="b53621266338"></a><a name="b53621266338"></a>rwMode</strong></p>
</td>
<td class="cellrowborder" valign="top" width="72.16%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>设置L2 Cache读写模式。</p>
<a name="screen774625064615"></a><a name="screen774625064615"></a><pre class="screen" codetype="Cpp" id="screen774625064615">enum CacheRwMode {
READ = 1,
WRITE = 2,
RW = 3
};</pre>
<p id="p18689719202918"><a name="p18689719202918"></a><a name="p18689719202918"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.989999999999998%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.07000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a><strong id="b16238121793314"><a name="b16238121793314"></a><a name="b16238121793314"></a>mode</strong></p>
</td>
<td class="cellrowborder" valign="top" width="12.989999999999998%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.07000000000001%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>用户指定的L2 Cache模式。</p>
<a name="screen1440994812502"></a><a name="screen1440994812502"></a><pre class="screen" codetype="Cpp" id="screen1440994812502">enum class CacheMode : uint8_t {
CACHE_MODE_DISABLE = 0, // 不使能L2 Cache
CACHE_MODE_NORMAL = 1,  // 使能L2 Cache
};</pre>
<p id="p595815581177"><a name="p595815581177"></a><a name="p595815581177"></a>如果用户在写算子时，相比不使能L2 Cache，某GlobalTensor使能L2 Cache反而会导致实测性能下降，可以手动禁止该GlobalTensor使能L2 Cache。比如某算子仅会读一次某个GlobalTensor数据，数据进L2 Cache并不会对算子产生收益，反而会因为数据频繁的搬入L2 Cache造成性能损耗，可以考虑不使能该GlobalTensor L2 Cache能力。</p>
<p id="p165951277168"><a name="p165951277168"></a><a name="p165951277168"></a>如果不调用该接口，默认为CacheMode::CACHE_MODE_NORMAL，即GlobalTensor会使能L2 Cache。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="section633mcpsimp"></a>

使用mssanitizer工具时，默认使能L2 Cache，无法通过本接口设置L2 Cache模式为CACHE\_MODE\_DISABLE。

## 调用示例<a name="section17531157161314"></a>

```
uint64_t dataSize = 256; //设置input_global的大小为256

AscendC::GlobalTensor<int32_t> inputGlobal; // 类型为int32_t
inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(src_gm), dataSize); // 设置源操作数在Global Memory上的起始地址为src_gm，所占外部存储的大小为256个int32_t
inputGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE); // 设置GlobalTensor不会写入L2 Cache

AscendC::LocalTensor<int32_t> inputLocal = inQueueX.AllocTensor<int32_t>();    
AscendC::DataCopy(inputLocal, inputGlobal, dataSize); // 将Global Memory上的inputGlobal拷贝到Local Memory的inputLocal上
```

