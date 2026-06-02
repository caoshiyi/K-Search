# SetQuantScalar<a name="ZH-CN_TOPIC_0000002554344011"></a>

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

本接口提供对输出矩阵的所有值采用同一系数进行量化或反量化的功能，即整个C矩阵对应一个量化参数，量化参数的shape为\[1\]。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8\_t或int4b\_t类型，输出为half类型；或者左、右矩阵的输入为int8\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8\_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8\_t类型。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row36481043185619"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.1.4.1.1 "><p id="p956355231512"><a name="p956355231512"></a><a name="p956355231512"></a>quantScalar</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.1.4.1.2 "><p id="p3755148105719"><a name="p3755148105719"></a><a name="p3755148105719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.1.4.1.3 "><p id="p1754648185714"><a name="p1754648185714"></a><a name="p1754648185714"></a>量化或反量化系数。</p>
</td>
</tr>
</tbody>
</table>

将float数据类型的量化计算参数scale、offset转换为uint64类型的入参的计算公式如下：

1.  quantScalar为64位格式，初始为0。
2.  scale按bit位取高19位截断，存储于quantScalar的bit位32位处，并将46位修改为1。

    _quantScalar = quantScalar∣ \(__s__c__a__l__e__ & 0__x__F__F__F__F__E__000\) ∣ \(1 ≪ 46\)_

3.  根据offset取值进行后续计算：
    -   若offset不存在，不再进行后续计算。
    -   若offset存在：
        1.  将offset值处理为int，范围为\[-256, 255\]。

            _o__f__f__s__e__t __= __M__a__x__\(__M__i__n__\(__I__N__T__\(__R__o__u__n__d__\(__o__f__f__s__e__t__\)\), 255\), −256\)_

        2.  再将offset按bit位保留9位并存储于quantScalar的37到45位。

            _quantScalar __= \(quantScalar & 0__x__4000__F__F__F__F__F__F__F__F__\) ∣ \(\(__o__f__f__s__e__t__ & 0__x__1__F__F__\) ≪ 37\)_

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

需与[SetDequantType](SetDequantType.md)保持一致。

本接口必须在[Iterate](Iterate.md)或者[IterateAll](IterateAll.md)前调用。

## 调用示例<a name="section1665082013318"></a>

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
float tmp = 0.1;  // 输出gm时会乘以0.1
// 将浮点值的量化或反量化系数，转换为uint64_t类型
uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
mm.SetQuantScalar(ans);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(biasGlobal);
}
mm.IterateAll(gm_c);
mm.End();
```

