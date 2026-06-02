# Clamp<a name="ZH-CN_TOPIC_0000002523343854"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将输入中除nan值以外大于max的数替换为max，小于min的数替换为min，小于等于max和大于等于min的数保持不变，作为输出。当min大于max时，将除nan值外所有值替换为max。min和max可以为标量或LocalTensor。

<!-- img2text -->
$$
\operatorname{out}_{i} =
\begin{cases}
\min, & \text{if } \operatorname{in}_{i} < \min \\
\operatorname{in}_{i}, & \text{if } \min \le \operatorname{in}_{i} \le \max \\
\max, & \text{if } \operatorname{in}_{i} > \max \\
\operatorname{NaN}, & \text{if } \operatorname{in}_{i} = \operatorname{NaN}
\end{cases}
$$

<!-- img2text -->
$$
\operatorname{output}_{i}=
\begin{cases}
\max, & \operatorname{input}_{i}>\max \\
\min, & \operatorname{input}_{i}<\min \\
\operatorname{input}_{i}, & \min \le \operatorname{input}_{i} \le \max \\
\operatorname{NaN}, & \operatorname{input}_{i}=\operatorname{NaN}
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```
template <const ClampConfig& config = DEFAULT_CLAMP_CONFIG, typename T, typename U, typename S>
__aicore__ inline void Clamp(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& min, const S& max, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row65668013156"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1111714411513"><a name="p1111714411513"></a><a name="p1111714411513"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p81171141152"><a name="p81171141152"></a><a name="p81171141152"></a>Clamp算法的相关配置。此参数可选配，ClampConfig类型，具体定义如下方代码所示，其中参数的含义为：</p>
<p id="p1081288191515"><a name="p1081288191515"></a><a name="p1081288191515"></a>isReuseSource：该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>操作数的数据类型。</p>
<p id="p382544110205"><a name="p382544110205"></a><a name="p382544110205"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row15186940142711"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11463077177"><a name="p11463077177"></a><a name="p11463077177"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p203415015319"><a name="p203415015319"></a><a name="p203415015319"></a>LocalTensor类型或标量类型。根据输入参数min自动推导类型，开发者无需配置该参数，保证min满足数据类型的约束即可。</p>
<p id="p830213147175"><a name="p830213147175"></a><a name="p830213147175"></a><span id="ph1730221471711"><a name="ph1730221471711"></a><a name="ph1730221471711"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row17453855918"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p74538555118"><a name="p74538555118"></a><a name="p74538555118"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p427962037"><a name="p427962037"></a><a name="p427962037"></a>LocalTensor类型或标量类型。根据输入参数max自动推导类型，开发者无需配置该参数，保证max满足数据类型的约束即可。</p>
<p id="p2214931021"><a name="p2214931021"></a><a name="p2214931021"></a><span id="ph421412314216"><a name="ph421412314216"></a><a name="ph421412314216"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
</tbody>
</table>

```
struct ClampConfig {
    bool isReuseSource;
};
```

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.38%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。</p>
<p id="p261714443395"><a name="p261714443395"></a><a name="p261714443395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row989124311219"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p72709151310"><a name="p72709151310"></a><a name="p72709151310"></a>min</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p18911043324"><a name="p18911043324"></a><a name="p18911043324"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p188911043027"><a name="p188911043027"></a><a name="p188911043027"></a>数据下限。类型为标量或LocalTensor，类型为LocalTensor时，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p25871622165817"><a name="p25871622165817"></a><a name="p25871622165817"></a>数据类型与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row15270111518314"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p10967111616488"><a name="p10967111616488"></a><a name="p10967111616488"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p327012158320"><a name="p327012158320"></a><a name="p327012158320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p14270191518315"><a name="p14270191518315"></a><a name="p14270191518315"></a>数据上限。类型为标量或LocalTensor，类型为LocalTensor时，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p782510257587"><a name="p782510257587"></a><a name="p782510257587"></a>数据类型与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row216545817417"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p184961858133"><a name="p184961858133"></a><a name="p184961858133"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::LocalTensor<half> dst, src;
uint32_t count = 512;
half min = 30;
half max = 60;
AscendC::Clamp(dst, src, min, max, count); 
```

结果示例如下：

```
输入数据（src）: 
[13, 78, 35, 95, 83,  2,  2, 95, 51, 73, 98,  3, 55, 32, 61,  2, 40, 26, 95, ... 63]
输入数据（min）: 
[30]
输入数据（max）: 
[60]
输出数据（dst）: 
[30, 60, 35, 60, 60, 30, 30, 60, 51, 60, 60, 30, 55, 32, 60, 30, 40, 30, 60, ... 60]
```

