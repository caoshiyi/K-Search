# GetMrgSortResult<a name="ZH-CN_TOPIC_0000002523303940"></a>

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

获取MrgSort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

本接口和MrgSort相关指令的配合关系如下：

-   配合[MrgSort指令](MrgSort.md)使用，获取MrgSort指令处理过的队列里的Region Proposal个数。使用时，需要将MrgSort中的MrgSort4Info.ifExhaustedSuspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort指令即停止。

    以上说明适用于如下型号：

    Ascend 950PR/Ascend 950DT

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void GetMrgSortResult(uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>mrgSortList1</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6745143393911"><a name="p6745143393911"></a><a name="p6745143393911"></a>类型为uint16_t，表示MrgSort第一个队列里已经处理过的Region Proposal个数。</p>
</td>
</tr>
<tr id="row1441613274210"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p14168210429"><a name="p14168210429"></a><a name="p14168210429"></a>mrgSortList2</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p19130457111014"><a name="p19130457111014"></a><a name="p19130457111014"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p54166204211"><a name="p54166204211"></a><a name="p54166204211"></a>类型为uint16_t，表示MrgSort第二个队列里已经处理过的Region Proposal个数。</p>
</td>
</tr>
<tr id="row470666124311"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p67071617430"><a name="p67071617430"></a><a name="p67071617430"></a>mrgSortList3</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p23611588108"><a name="p23611588108"></a><a name="p23611588108"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p13852181720542"><a name="p13852181720542"></a><a name="p13852181720542"></a>类型为uint16_t，表示MrgSort第三个队列里已经处理过的Region Proposal个数。</p>
</td>
</tr>
<tr id="row59924436234"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p59928437239"><a name="p59928437239"></a><a name="p59928437239"></a>mrgSortList4</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p85228148413"><a name="p85228148413"></a><a name="p85228148413"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p19921543132312"><a name="p19921543132312"></a><a name="p19921543132312"></a>类型为uint16_t，表示MrgSort第四个队列里已经处理过的Region Proposal个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

-   配合[MrgSort指令](MrgSort.md)使用示例。

    ```
    AscendC::LocalTensor<float> dstLocal;
    AscendC::LocalTensor<float> workLocal;
    AscendC::LocalTensor<float> src0Local;
    AscendC::LocalTensor<uint32_t> src1Local;
    
    AscendC::Sort32(workLocal, src0Local, src1Local, 1);
    
    uint16_t elementLengths[4] = { 0 };
    uint32_t sortedNum[4] = { 0 };
    elementLengths[0] = 32;
    elementLengths[1] = 32;
    elementLengths[2] = 32;
    elementLengths[3] = 32;
    uint16_t validBit = 0b1111;
    
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = workLocal[0];
    srcList.src2 = workLocal[32 * 1 * 2];
    srcList.src3 = workLocal[32 * 2 * 2];
    srcList.src4 = workLocal[32 * 3 * 2];
    
    AscendC::MrgSort4Info mrgSortInfo(elementLengths, true, validBit, 1);
    AscendC::MrgSort(dstLocal, srcList, mrgSortInfo);
    
    uint16_t mrgRes1 = 0;
    uint16_t mrgRes2 = 0;
    uint16_t mrgRes3 = 0;
    uint16_t mrgRes4 = 0;
    AscendC::GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);
    ```

