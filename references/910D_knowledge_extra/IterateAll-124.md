# IterateAll<a name="ZH-CN_TOPIC_0000002523343620"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section59092521779"></a>

通过设置结果矩阵Output在GM上的首地址，本接口一次性计算singleCo \* singleDo \* singleM大小的数据块，并写到结果矩阵Output中。

本接口提供单核内卷积计算能力，singleCo为多核切分后单个核内的输出通道大小；singleDo为多核切分后单个核内的Dout大小；singleM为多核切分后单个核内的M大小。singleCo、singleDo和singleM的大小通过[SetSingleOutputShape](SetSingleOutputShape.md)接口设置。

## 函数原型<a name="section1875131281"></a>

```
__aicore__ inline void IterateAll(const AscendC::GlobalTensor<OutputT>& output, bool enPartialSum = false)
```

## 参数说明<a name="section122574911817"></a>

<a name="table282014234518"></a>
<table><thead align="left"><tr id="row38391728458"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p13839523451"><a name="p13839523451"></a><a name="p13839523451"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p138391021456"><a name="p138391021456"></a><a name="p138391021456"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p15839172164512"><a name="p15839172164512"></a><a name="p15839172164512"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row38391324456"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p128401627451"><a name="p128401627451"></a><a name="p128401627451"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p484014217457"><a name="p484014217457"></a><a name="p484014217457"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p1761464914415"><a name="p1761464914415"></a><a name="p1761464914415"></a>Output在GM上的地址。<span id="ph15942199192220"><a name="ph15942199192220"></a><a name="ph15942199192220"></a><span id="ph1294215916225"><a name="ph1294215916225"></a><a name="ph1294215916225"></a><span id="ph894279182218"><a name="ph894279182218"></a><a name="ph894279182218"></a>类型为<a href="GlobalTensor.md">GlobalTensor</a>。</span></span></span>结果矩阵Output支持的数据类型为：half、bfloat16_t。</p>
</td>
</tr>
<tr id="row4840526454"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p5840102134513"><a name="p5840102134513"></a><a name="p5840102134513"></a>enPartialSum</p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p98403210456"><a name="p98403210456"></a><a name="p98403210456"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p178402218454"><a name="p178402218454"></a><a name="p178402218454"></a>预留参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1399513269816"></a>

无

## 约束说明<a name="section73747219912"></a>

-   IterateAll接口仅支持处理单batch数据，在多batch计算场景中，需要通过batch次循环调用IterateAll接口完成计算。

    ```
    for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
        conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
        conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
        conv3dApi.End();
    }
    ```

-   IterateAll接口必须在初始化接口及输入输出配置接口之后进行调用，完成Conv3D计算，调用顺序如下。

    ```
    Init(...);
    ... // 输入输出配置
    IterateAll(...);
    End();
    ```

## 调用示例<a name="section510151019917"></a>

```
TPipe pipe;
conv3dApi.Init(&tiling);
conv3dApi.SetWeight(weightGm);
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
    conv3dApi.SetInput(inputGm[batchIter * inputOneBatchSize]);
    conv3dApi.IterateAll(outputGm[batchIter * outputOneBatchSize]);
    conv3dApi.End();
}
```

