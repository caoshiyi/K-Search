# HCCL模板参数<a name="ZH-CN_TOPIC_0000002523303894"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section91181434183612"></a>

创建HCCL对象时需要传入模板参数。

## 函数原型<a name="section1575516397366"></a>

Hccl类定义如下，模板参数说明见[表1 Hccl类模板参数说明](#table884518212555)。

```
template <HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU, const auto &config = DEFAULT_CFG>
class Hccl;
```

## 参数说明<a name="section13114185312367"></a>

**表 1**  Hccl类模板参数说明

<a name="table884518212555"></a>
<table><thead align="left"><tr id="row1584512213553"><th class="cellrowborder" valign="top" width="13.700000000000001%" id="mcps1.2.3.1.1"><p id="p550734719348"><a name="p550734719348"></a><a name="p550734719348"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="86.3%" id="mcps1.2.3.1.2"><p id="p1784517215550"><a name="p1784517215550"></a><a name="p1784517215550"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row484511235519"><td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.3.1.1 "><p id="p205071947133410"><a name="p205071947133410"></a><a name="p205071947133410"></a>serverType</p>
</td>
<td class="cellrowborder" valign="top" width="86.3%" headers="mcps1.2.3.1.2 "><p id="p4885153562018"><a name="p4885153562018"></a><a name="p4885153562018"></a>支持的服务端类型。HcclServerType类型，定义如下。</p>
<p id="p1880057123714"><a name="p1880057123714"></a><a name="p1880057123714"></a>对于<span id="ph1788095712378"><a name="ph1788095712378"></a><a name="ph1788095712378"></a>Ascend 950PR/Ascend 950DT</span>，当前仅支持HCCL_SERVER_TYPE_CCU。</p>
<a name="screen519263825511"></a><a name="screen519263825511"></a><pre class="screen" codetype="Cpp" id="screen519263825511">enum HcclServerType {
HCCL_SERVER_TYPE_AICPU = 0,
HCCL_SERVER_TYPE_CCU,
HCCL_SERVER_TYPE_END  // 预留参数，不支持使用
}</pre>
</td>
</tr>
<tr id="row143138574332"><td class="cellrowborder" valign="top" width="13.700000000000001%" headers="mcps1.2.3.1.1 "><p id="p150710476349"><a name="p150710476349"></a><a name="p150710476349"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="86.3%" headers="mcps1.2.3.1.2 "><p id="p753110515362"><a name="p753110515362"></a><a name="p753110515362"></a>用于指定向服务端下发任务的核。HcclServerConfig类型，定义如下，默认值DEFAULT_CFG = {CoreType::DEFAULT, 0}。</p>
<a name="screen13319154084018"></a><a name="screen13319154084018"></a><pre class="screen" codetype="Cpp" id="screen13319154084018">struct HcclServerConfig {
    CoreType type;  // 向服务端下发任务的核的类型
    int64_t blockId;  // 向服务端下发任务的核的ID
};</pre>
<p id="p0241446123614"><a name="p0241446123614"></a><a name="p0241446123614"></a>CoreType的定义如下：</p>
<a name="screen184661921124110"></a><a name="screen184661921124110"></a><pre class="screen" codetype="Cpp" id="screen184661921124110">enum class CoreType: uint8_t {
    DEFAULT,  // 表示不指定AIC核或者AIV核
    ON_AIV,     // 表示指定为AIV核
    ON_AIC     // 表示指定为AIC核
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section11493459173619"></a>

通过如下传入模板参数config的方式创建Hccl类对象，指定HCCL客户端仅在AIV的10号核上发送通信消息给服务端，替代通过调用[GetBlockIdx](GetBlockIdx.md)接口的方式指定运行的核。

```
static constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
// 选择CCU作为服务端
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU, HCCL_CFG> hccl; 
```

