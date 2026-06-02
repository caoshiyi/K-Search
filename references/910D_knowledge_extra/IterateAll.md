# IterateAll<a name="ZH-CN_TOPIC_0000002523344580"></a>

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

调用一次IterateAll，会计算出singleCoreM \* singleCoreN大小的C矩阵。迭代顺序可通过tiling参数iterateOrder调整。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool sync = true> __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
```

```
template <bool sync = true> __aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table8746171282418"></a>
<table><thead align="left"><tr id="row8746191212419"><th class="cellrowborder" valign="top" width="17.04%" id="mcps1.2.3.1.1"><p id="p474618126245"><a name="p474618126245"></a><a name="p474618126245"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.96%" id="mcps1.2.3.1.2"><p id="p1574681216240"><a name="p1574681216240"></a><a name="p1574681216240"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1574617127244"><td class="cellrowborder" valign="top" width="17.04%" headers="mcps1.2.3.1.1 "><p id="p2746171214244"><a name="p2746171214244"></a><a name="p2746171214244"></a>sync</p>
</td>
<td class="cellrowborder" valign="top" width="82.96%" headers="mcps1.2.3.1.2 "><p id="p12551641162414"><a name="p12551641162414"></a><a name="p12551641162414"></a>获取C矩阵过程分为同步和异步两种模式：</p>
<a name="ul101321025155310"></a><a name="ul101321025155310"></a><ul id="ul101321025155310"><li><strong id="b18450458153116"><a name="b18450458153116"></a><a name="b18450458153116"></a>同步：</strong>需要同步等待IterateAll执行结束</li><li><strong id="b72671419326"><a name="b72671419326"></a><a name="b72671419326"></a>异步：</strong>不需要同步等待IterateAll执行结束</li></ul>
<p id="p826212484315"><a name="p826212484315"></a><a name="p826212484315"></a>通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false，默认为同步模式。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15339630201219"><a name="p15339630201219"></a><a name="p15339630201219"></a>gm</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p14339183015126"><a name="p14339183015126"></a><a name="p14339183015126"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p81172035142916"><a name="p81172035142916"></a><a name="p81172035142916"></a>C矩阵。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span></p>
<p id="p1811014241778"><a name="p1811014241778"></a><a name="p1811014241778"></a><span id="ph71109242712"><a name="ph71109242712"></a><a name="ph71109242712"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t/fp8_e4m3fn_t/hifloat8_t</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p15545815312"><a name="p15545815312"></a><a name="p15545815312"></a>ubCmatrix</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1433933041215"><a name="p1433933041215"></a><a name="p1433933041215"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p18126352102919"><a name="p18126352102919"></a><a name="p18126352102919"></a>C矩阵。<span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为TSCM。</span></span></span></p>
<p id="p41907431571"><a name="p41907431571"></a><a name="p41907431571"></a><span id="ph61901431474"><a name="ph61901431474"></a><a name="ph61901431474"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t/fp8_e4m3fn_t/hifloat8_t</p>
</td>
</tr>
<tr id="row6652117123"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1733933012128"><a name="p1733933012128"></a><a name="p1733933012128"></a>enAtomic</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p18557143813197"><a name="p18557143813197"></a><a name="p18557143813197"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p11557163871916"><a name="p11557163871916"></a><a name="p11557163871916"></a>是否开启Atomic操作，默认值为0。</p>
<p id="p553784617537"><a name="p553784617537"></a><a name="p553784617537"></a>参数取值：</p>
<p id="p19697610154618"><a name="p19697610154618"></a><a name="p19697610154618"></a>0：不开启Atomic操作</p>
<p id="p108028241466"><a name="p108028241466"></a><a name="p108028241466"></a>1：开启AtomicAdd累加操作</p>
<p id="p875272616463"><a name="p875272616463"></a><a name="p875272616463"></a>2：开启AtomicMax求最大值操作</p>
<p id="p337993234616"><a name="p337993234616"></a><a name="p337993234616"></a>3：开启AtomicMin求最小值操作</p>
</td>
</tr>
<tr id="row26852175375"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p17686171710377"><a name="p17686171710377"></a><a name="p17686171710377"></a>enSequentialWrite</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p145578389196"><a name="p145578389196"></a><a name="p145578389196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p8557183831912"><a name="p8557183831912"></a><a name="p8557183831912"></a>是否开启连续写模式（<a href="GetTensorC.md#fig580415103338">连续写</a>，写入[baseM, baseN]；<a href="GetTensorC.md#fig322951853119">非连续写</a>，写入[singleCoreM, singleCoreN]中对应的位置），默认值false（非连续写模式）。</p>
</td>
</tr>
<tr id="row1153482913714"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p145351329193714"><a name="p145351329193714"></a><a name="p145351329193714"></a>waitIterateAll</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p6535429123712"><a name="p6535429123712"></a><a name="p6535429123712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p205351629143710"><a name="p205351629143710"></a><a name="p205351629143710"></a>仅在异步场景下使用，是否需要通过<a href="WaitIterateAll.md">WaitIterateAll</a>接口等待IterateAll执行结束。</p>
<p id="p1859926195210"><a name="p1859926195210"></a><a name="p1859926195210"></a>true：需要通过WaitIterateAll接口等待IterateAll执行结束。</p>
<p id="p381710502478"><a name="p381710502478"></a><a name="p381710502478"></a>false：不需要通过WaitIterateAll接口等待IterateAll执行结束，开发者自行处理等待IterateAll执行结束的过程。</p>
</td>
</tr>
<tr id="row288603995510"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p168861239155511"><a name="p168861239155511"></a><a name="p168861239155511"></a>fakeMsg</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1188620391553"><a name="p1188620391553"></a><a name="p1188620391553"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1502113712101"><a name="p1502113712101"></a><a name="p1502113712101"></a>仅在IBShare场景（模板参数中开启了<a href="MatmulConfig.md#p1387745902510">doIBShareNorm</a>开关）和IntraBlockPartSum场景（模板参数中开启了<a href="MatmulConfig.md#p193339432166">intraBlockPartSum</a>开关）使用。</p>
<a name="ul1162964145610"></a><a name="ul1162964145610"></a><ul id="ul1162964145610"><li>IBShare场景<p id="p08781347121114"><a name="p08781347121114"></a><a name="p08781347121114"></a>该场景复用L1上相同的A矩阵或B矩阵数据，要求AIV分核调用IterateAll的次数必须匹配，此时需要调用IterateAll并设置fakeMsg为true，不执行真正的计算，仅用来保证IterateAll调用成对出现。默认值为false，表示执行真正的计算。</p>
</li></ul>
<a name="ul3301016155618"></a><a name="ul3301016155618"></a><ul id="ul3301016155618"><li>IntraBlockPartSum场景<p id="p8693826161616"><a name="p8693826161616"></a><a name="p8693826161616"></a>用于分离模式下的Vector计算、Cube计算融合，实现多个AIV核的一次Matmul计算结果（baseM * baseN大小的矩阵分片）在<span id="ph410020151596"><a name="ph410020151596"></a><a name="ph410020151596"></a>L0C Buffer</span>上累加。默认值为false，表示执行各AIV核的Matmul计算结果在<span id="ph857612521808"><a name="ph857612521808"></a><a name="ph857612521808"></a>L0C Buffer</span>上累加。</p>
</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

传入的C矩阵地址空间大小需要保证不小于singleCoreM \* singleCoreN个元素。

## 调用示例<a name="section1665082013318"></a>

IterateAll接口的调用示例如下，更多异步场景的算子样例请参考[IterateAll异步场景矩阵乘法](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_async_iterate_all)。

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);    // 计算
mm.End();
```

