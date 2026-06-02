# SetOrgShape<a name="ZH-CN_TOPIC_0000002523343686"></a>

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

设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
```

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p13101125318816"><a name="p13101125318816"></a><a name="p13101125318816"></a>orgM</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p7101195316815"><a name="p7101195316815"></a><a name="p7101195316815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p210118531786"><a name="p210118531786"></a><a name="p210118531786"></a>设置原始完整的形状M大小，单位为元素。</p>
<p id="p10718133161513"><a name="p10718133161513"></a><a name="p10718133161513"></a>对于<span id="ph1596824831314"><a name="ph1596824831314"></a><a name="ph1596824831314"></a>Ascend 950PR/Ascend 950DT</span>，在使用<a href="MatmulConfig.md">MDL模板</a>创建的Matmul对象调用本接口时，该参数用于设置GM或L1上完整的形状M大小，单位为元素。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1510114531681"><a name="p1510114531681"></a><a name="p1510114531681"></a>orgN</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p9101145313811"><a name="p9101145313811"></a><a name="p9101145313811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1510219534819"><a name="p1510219534819"></a><a name="p1510219534819"></a>设置原始完整的形状N大小，单位为元素。</p>
<p id="p1971719248157"><a name="p1971719248157"></a><a name="p1971719248157"></a>对于<span id="ph176824911141"><a name="ph176824911141"></a><a name="ph176824911141"></a>Ascend 950PR/Ascend 950DT</span>，在使用<a href="MatmulConfig.md">MDL模板</a>创建的Matmul对象调用本接口时，该参数用于设置GM或L1上完整的形状N大小，单位为元素。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p121021353385"><a name="p121021353385"></a><a name="p121021353385"></a>orgK</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p121028531183"><a name="p121028531183"></a><a name="p121028531183"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p610210531811"><a name="p610210531811"></a><a name="p610210531811"></a>设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。</p>
<p id="p3586131331610"><a name="p3586131331610"></a><a name="p3586131331610"></a>对于<span id="ph819471321418"><a name="ph819471321418"></a><a name="ph819471321418"></a>Ascend 950PR/Ascend 950DT</span>，在使用<a href="MatmulConfig.md">MDL模板</a>创建的Matmul对象调用本接口时，该参数用于设置GM或L1上完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。</p>
</td>
</tr>
<tr id="row1825910356718"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p162591355715"><a name="p162591355715"></a><a name="p162591355715"></a>orgKa</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p026013356715"><a name="p026013356715"></a><a name="p026013356715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p42608354713"><a name="p42608354713"></a><a name="p42608354713"></a>设置矩阵A原始完整的形状Ka大小，单位为元素。</p>
<p id="p16360201501514"><a name="p16360201501514"></a><a name="p16360201501514"></a>对于<span id="ph1868818146140"><a name="ph1868818146140"></a><a name="ph1868818146140"></a>Ascend 950PR/Ascend 950DT</span>，在使用<a href="MatmulConfig.md">MDL模板</a>创建的Matmul对象调用本接口时，该参数用于设置GM或L1上完整的形状Ka大小，单位为元素。</p>
</td>
</tr>
<tr id="row1569446685"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p176941261588"><a name="p176941261588"></a><a name="p176941261588"></a>orgKb</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p15695461480"><a name="p15695461480"></a><a name="p15695461480"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p116952061384"><a name="p116952061384"></a><a name="p116952061384"></a>设置矩阵B原始完整的形状Kb大小，单位为元素。</p>
<p id="p1192020537131"><a name="p1192020537131"></a><a name="p1192020537131"></a>对于<span id="ph9973191513149"><a name="ph9973191513149"></a><a name="ph9973191513149"></a>Ascend 950PR/Ascend 950DT</span>，在使用<a href="MatmulConfig.md">MDL模板</a>创建的Matmul对象调用本接口时，该参数用于设置GM或L1上完整的形状Kb大小，单位为元素。</p>
</td>
</tr>
<tr id="row1046745816362"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1746875863613"><a name="p1746875863613"></a><a name="p1746875863613"></a>orgKc</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p14468058143616"><a name="p14468058143616"></a><a name="p14468058143616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p194681583369"><a name="p194681583369"></a><a name="p194681583369"></a>设置输出C矩阵的N，单位为元素。需要输入B矩阵的N和输出C矩阵的N不一样时可设置，默认为0（即使用B矩阵的N，不进行修改）。</p>
</td>
</tr>
</tbody>
</table>

> **须知：** 
>-   对于Ascend 950PR/Ascend 950DT上使用[MDL模板](MatmulConfig.md)创建的Matmul对象，L1上数据的形状与Tiling侧接口[SetOrgShape](SetOrgShape-108.md)中的orgMIn/orgNIn/orgKIn/orgKaIn/orgKbIn一致时，不必须调用此接口。
>-   对于Ascend 950PR/Ascend 950DT上使用[MDL模板](MatmulConfig.md)创建的Matmul对象，L1上数据的形状与Tiling侧接口[SetOrgShape](SetOrgShape-108.md)中的orgMIn/orgNIn/orgKIn/orgKaIn/orgKbIn不一致时，必须调用本接口指定GM/L1上的orgM/orgN/orgK/orgKa/orgKb。
>    例如，使用[MDL模板](MatmulConfig.md)时，输入矩阵A在L1、输入矩阵B在GM的场景，L1上A的形状大小与Tiling侧原始的orgMIn/orgKIn/orgKaIn不一致时，调用SetOrgShape\(orgM, orgN, orgK\)/SetOrgShape\(orgM, orgN, orgKa, orgKb\)接口指定L1上A矩阵相关参数orgM/orgK/orgKa。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

本接口需要在SetTensorA接口、SetTensorB接口、SetBias接口及SetSingleShape接口前调用。

## 调用示例<a name="section1665082013318"></a>

-   设置矩阵原始完整的形状

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gm_a);
    mm.SetTensorB(gm_b);
    mm.SetBias(gm_bias);
    mm.IterateAll(gm_c);
    //  复用mm对象
    mm.SetOrgShape(orgM, orgN, orgK);
    mm.SetTensorA(gm_a1);
    mm.SetTensorB(gm_b1);
    mm.SetBias(gm_bias1);
    mm.IterateAll(gm_c1);
    ```

-   对于Ascend 950PR/Ascend 950DT上使用[MDL模板](MatmulConfig.md)的Matmul对象，设置GM或L1上完整的形状

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    for (int m = 0; m < mIter_; m++) {
         for (int n = 0; n < nIter_; n++) {
              for (int k = 0; k < kIter_; k++) {
                   // 复用mm，指定A在L1和B在GM上的shape
                   mm.SetOrgShape(alignedSingleM, tiling.N, alignedSingleK, tiling.Kb, tiling.N);
                   mm.SetSingleShape(curBaseM, curBaseN, curBaseK);
                   mm.SetTensorA(tscm_a[offset_a]); // Set aMatrix tscm input
                   mm.SetTensorB(gm_b[offset_b]);
                   mm.SetBias(gm_bias[offset_bias]);
                   mm.Iterate(k != 0);
                }
                matmulObj.GetTensorC(gm_c[offset_c]);
         }
    }
    ```

