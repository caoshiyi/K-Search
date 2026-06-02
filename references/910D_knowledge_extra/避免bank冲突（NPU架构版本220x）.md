# 避免bank冲突（NPU架构版本220x）<a name="ZH-CN_TOPIC_0000002523289092"></a>

【优先级】高

> **说明：** 
>该性能优化建议适用于如下产品型号：

【描述】为了提高数据访问的效率和吞吐量，Unified Buffer采用了bank（大小相等的内存模块）结构设计。Unified Buffer总大小为192K，划分为48个bank。每个bank由128行组成，每行长度为32B。这48个bank进一步组织为16个bank group，每个bank group包含3个bank，例如bank15、bank31和bank47组成一个bank group。

**图 1**  bank结构示意图（图中箭头方向表示内存排布的顺序）<a name="fig1132542915196"></a>  
<!-- img2text -->
```text
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ bank   │ bank   │ bank   │ bank   │ bank   │ bank   │ bank   │ bank   │ bank   │ bank   │ bank    │ bank    │ bank    │ bank    │ bank    │ bank    │
│ group0 │ group1 │ group2 │ group3 │ group4 │ group5 │ group6 │ group7 │ group8 │ group9 │ group10 │ group11 │ group12 │ group13 │ group14 │ group15 │
└───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬───┘
    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │  32B    │  32B    │  32B    │  32B    │  32B    │  32B    │
┌───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┐
│ bank0  │ bank1  │ bank2  │ bank3  │ bank4  │ bank5  │ bank6  │ bank7  │ bank8  │ bank9  │ bank10  │ bank11  │ bank12  │ bank13  │ bank14  │ bank15  │
│  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │
└───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┘
↑   │        │        │        │        │        │        │        │        │        │         │         │         │         │         │         ↑
│ 128                                                                                                                                        128 │
↓   │        │        │        │        │        │        │        │        │        │         │         │         │         │         │         ↓
┌───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┐
│ bank16 │ bank17 │ bank18 │ bank19 │ bank20 │ bank21 │ bank22 │ bank23 │ bank24 │ bank25 │ bank26  │ bank27  │ bank28  │ bank29  │ bank30  │ bank31  │
│  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │
└───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┬────┴─────┘
↑   │        │        │        │        │        │        │        │        │        │         │         │         │         │         │         ↑
│ 128                                                                                                                                        128 │
↓   │        │        │        │        │        │        │        │        │        │         │         │         │         │         │         ↓
┌───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┬───▼─────┐
│ bank32 │ bank33 │ bank34 │ bank35 │ bank36 │ bank37 │ bank38 │ bank39 │ bank40 │ bank41 │ bank42  │ bank43  │ bank44  │ bank45  │ bank46  │ bank47  │
│  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB   │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │  4KB    │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │ 32B    │  32B    │  32B    │  32B    │  32B    │  32B    │  32B

内存排布顺序:
bank0 → bank1 → bank2 → bank3 → bank4 → bank5 → bank6 → bank7 → bank8 → bank9 → bank10 → bank11 → bank12 → bank13 → bank14 → bank15
  ↓
bank16 → bank17 → bank18 → bank19 → bank20 → bank21 → bank22 → bank23 → bank24 → bank25 → bank26 → bank27 → bank28 → bank29 → bank30 → bank31
  ↓
bank32 → bank33 → bank34 → bank35 → bank36 → bank37 → bank38 → bank39 → bank40 → bank41 → bank42 → bank43 → bank44 → bank45 → bank46 → bank47

bank group 对应关系:
bank group0  : bank0,  bank16, bank32
bank group1  : bank1,  bank17, bank33
bank group2  : bank2,  bank18, bank34
bank group3  : bank3,  bank19, bank35
bank group4  : bank4,  bank20, bank36
bank group5  : bank5,  bank21, bank37
bank group6  : bank6,  bank22, bank38
bank group7  : bank7,  bank23, bank39
bank group8  : bank8,  bank24, bank40
bank group9  : bank9,  bank25, bank41
bank group10 : bank10, bank26, bank42
bank group11 : bank11, bank27, bank43
bank group12 : bank12, bank28, bank44
bank group13 : bank13, bank29, bank45
bank group14 : bank14, bank30, bank46
bank group15 : bank15, bank31, bank47
```

每个bank可以独立地进行数据的读写操作，允许多个数据请求同时进行。然而，当多个读写操作试图同时访问同一个bank或bank group时，由于硬件资源的限制，这些操作必须排队等待，会导致bank冲突，引起性能下降。

具体来说，Vector计算单元每拍（一个指令周期）能够从每个bank group中读取或写入一行数据。如果同一个API中的多个操作试图同时访问同一个bank或bank group，Vector计算单元无法在同一个周期内处理所有请求，导致这些请求排队等待。这种排队增加了数据访问的延迟，降低了系统的整体性能。

## bank冲突的典型场景<a name="section12644115352"></a>

bank冲突主要可以分为以下三种场景：

-   **读写冲突**：读操作和写操作同时尝试访问同一个bank。
-   **写写冲突**：多个写操作同时尝试访问同一个bank group。
-   **读读冲突**：多个读操作同时尝试访问同一个bank group。

下文给出了一些具体的示例，假设，0x10000地址在bank16上，0x10020在bank17上，0x20020在bank33上，如下图所示：

**图 2**  地址分配示意图<a name="fig129245311375"></a>  
<!-- img2text -->
``` 
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ bank group0  │ bank group1  │ bank group...│ bank group15 │
├──────────────┼──────────────┼──────────────┼──────────────┤
│              │              │              │              │
│   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │
│   │ bank   │ │   │ bank   │ │   │ bank   │ │   │ bank   │ │
│   │   0    │ │   │   1    │ │   │   -    │ │   │   15   │ │
│   └────────┘ │   └────────┘ │   └────────┘ │   └────────┘ │
│   0x10000    │   0x10020    │              │              │
│              │              │              │              │
│   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │
│   │ bank   │ │   │ bank   │ │   │ bank   │ │   │ bank   │ │
│   │   16   │ │   │   17   │ │   │   -    │ │   │   31   │ │
│   └────────┘ │   └────────┘ │   └────────┘ │   └────────┘ │
│              │              │              │              │
│   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │   ┌────────┐ │
│   │ bank   │ │   │ bank   │ │   │ bank   │ │   │ bank   │ │
│   │   32   │ │   │   33   │ │   │   -    │ │   │   47   │ │
│   └────────┘ │   └────────┘ │   └────────┘ │   └────────┘ │
│              │   0x20020    │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

-   读写冲突示例

    Vector指令的源操作数src和目的操作数dst同时读写到同一个bank时造成读写冲突，具体分析如下：

    **表 1**  读写冲突示例

    <a name="table06741342154717"></a>
    <table><thead align="left"><tr id="row1767434244710"><th class="cellrowborder" valign="top" width="5.9988002399520095%" id="mcps1.2.7.1.1"><p id="p17674144211470"><a name="p17674144211470"></a><a name="p17674144211470"></a>序号</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.948010397920417%" id="mcps1.2.7.1.2"><p id="p9674114213473"><a name="p9674114213473"></a><a name="p9674114213473"></a>src地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.507898420315936%" id="mcps1.2.7.1.3"><p id="p8674114294712"><a name="p8674114294712"></a><a name="p8674114294712"></a>dst地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="25.51489702059588%" id="mcps1.2.7.1.4"><p id="p17674114244718"><a name="p17674114244718"></a><a name="p17674114244718"></a>bank</p>
    </th>
    <th class="cellrowborder" valign="top" width="31.36372725454909%" id="mcps1.2.7.1.5"><p id="p1567444234713"><a name="p1567444234713"></a><a name="p1567444234713"></a>bank group</p>
    </th>
    <th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p1767410425479"><a name="p1767410425479"></a><a name="p1767410425479"></a>结论</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row126741742144713"><td class="cellrowborder" valign="top" width="5.9988002399520095%" headers="mcps1.2.7.1.1 "><p id="p1367415427473"><a name="p1367415427473"></a><a name="p1367415427473"></a>示例1</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.948010397920417%" headers="mcps1.2.7.1.2 "><p id="p7674134216478"><a name="p7674134216478"></a><a name="p7674134216478"></a>0x10020</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.507898420315936%" headers="mcps1.2.7.1.3 "><p id="p1567416428472"><a name="p1567416428472"></a><a name="p1567416428472"></a>0x10000</p>
    </td>
    <td class="cellrowborder" valign="top" width="25.51489702059588%" headers="mcps1.2.7.1.4 "><p id="p16674184212479"><a name="p16674184212479"></a><a name="p16674184212479"></a><span>bank_id0 != </span><span>bank_id</span><span>1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p7674042124711"><a name="p7674042124711"></a><a name="p7674042124711"></a><span>bank_group_id0 != bank_group_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p167515425479"><a name="p167515425479"></a><a name="p167515425479"></a>src地址和dst地址分别属于bank16和bank17，故无冲突。</p>
    </td>
    </tr>
    <tr id="row3675154284710"><td class="cellrowborder" valign="top" width="5.9988002399520095%" headers="mcps1.2.7.1.1 "><p id="p967514254716"><a name="p967514254716"></a><a name="p967514254716"></a>示例2</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.948010397920417%" headers="mcps1.2.7.1.2 "><p id="p1467514214471"><a name="p1467514214471"></a><a name="p1467514214471"></a>0x10020</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.507898420315936%" headers="mcps1.2.7.1.3 "><p id="p156751042114711"><a name="p156751042114711"></a><a name="p156751042114711"></a>0x10E20</p>
    </td>
    <td class="cellrowborder" valign="top" width="25.51489702059588%" headers="mcps1.2.7.1.4 "><p id="p1367534210471"><a name="p1367534210471"></a><a name="p1367534210471"></a><span>bank_id0 </span><span>== </span><span>bank_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p86756421472"><a name="p86756421472"></a><a name="p86756421472"></a><span>bank_group_id0 =</span><span>= </span><span>bank_group_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1667544264720"><a name="p1667544264720"></a><a name="p1667544264720"></a>src地址和dst地址的地址都在bank17，故存在冲突。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   写写冲突示例

    Vector指令目的操作数dst对应的8个DataBlock（block0-block7）同时写到一个bank group时造成写写冲突，具体分析如下：

    **表 2**  写写冲突示例

    <a name="table11121346153814"></a>
    <table><thead align="left"><tr id="row212044623812"><th class="cellrowborder" valign="top" width="5.789726356216995%" id="mcps1.2.9.1.1"><p id="p13120184619386"><a name="p13120184619386"></a><a name="p13120184619386"></a>序号</p>
    </th>
    <th class="cellrowborder" valign="top" width="7.959673547767644%" id="mcps1.2.9.1.2"><p id="p2012004612384"><a name="p2012004612384"></a><a name="p2012004612384"></a>dst地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.505520883341337%" id="mcps1.2.9.1.3"><p id="p112014610383"><a name="p112014610383"></a><a name="p112014610383"></a>blk_stride</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.660585693710996%" id="mcps1.2.9.1.4"><p id="p212024619386"><a name="p212024619386"></a><a name="p212024619386"></a><span>block0_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="8.775804128660585%" id="mcps1.2.9.1.5"><p id="p1120146103816"><a name="p1120146103816"></a><a name="p1120146103816"></a><span>block1_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="8.967834853576573%" id="mcps1.2.9.1.6"><p id="p141208461384"><a name="p141208461384"></a><a name="p141208461384"></a><span>block2_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="6.721075372059531%" id="mcps1.2.9.1.7"><p id="p202701742154610"><a name="p202701742154610"></a><a name="p202701742154610"></a>...</p>
    </th>
    <th class="cellrowborder" valign="top" width="43.61977916466635%" id="mcps1.2.9.1.8"><p id="p171201746183813"><a name="p171201746183813"></a><a name="p171201746183813"></a>结论</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row161211146173810"><td class="cellrowborder" valign="top" width="5.789726356216995%" headers="mcps1.2.9.1.1 "><p id="p111208465382"><a name="p111208465382"></a><a name="p111208465382"></a>示例1</p>
    </td>
    <td class="cellrowborder" valign="top" width="7.959673547767644%" headers="mcps1.2.9.1.2 "><p id="p212044615383"><a name="p212044615383"></a><a name="p212044615383"></a>0x1FE00</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.505520883341337%" headers="mcps1.2.9.1.3 "><p id="p14120046103813"><a name="p14120046103813"></a><a name="p14120046103813"></a>16</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.660585693710996%" headers="mcps1.2.9.1.4 "><p id="p1012016469382"><a name="p1012016469382"></a><a name="p1012016469382"></a>0x1FE00</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.775804128660585%" headers="mcps1.2.9.1.5 "><p id="p112019463387"><a name="p112019463387"></a><a name="p112019463387"></a>0x20000</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.967834853576573%" headers="mcps1.2.9.1.6 "><p id="p71213469382"><a name="p71213469382"></a><a name="p71213469382"></a>0x20200</p>
    </td>
    <td class="cellrowborder" valign="top" width="6.721075372059531%" headers="mcps1.2.9.1.7 "><p id="p127119427464"><a name="p127119427464"></a><a name="p127119427464"></a>...</p>
    </td>
    <td class="cellrowborder" valign="top" width="43.61977916466635%" headers="mcps1.2.9.1.8 "><p id="p81217462386"><a name="p81217462386"></a><a name="p81217462386"></a>8个DataBlock均在一个bank group下，故全部冲突，8拍完成一个Repeat的写入。</p>
    </td>
    </tr>
    <tr id="row1112120465382"><td class="cellrowborder" valign="top" width="5.789726356216995%" headers="mcps1.2.9.1.1 "><p id="p1012184610384"><a name="p1012184610384"></a><a name="p1012184610384"></a>示例2</p>
    </td>
    <td class="cellrowborder" valign="top" width="7.959673547767644%" headers="mcps1.2.9.1.2 "><p id="p41211746143815"><a name="p41211746143815"></a><a name="p41211746143815"></a>0x1FE00</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.505520883341337%" headers="mcps1.2.9.1.3 "><p id="p71211446173816"><a name="p71211446173816"></a><a name="p71211446173816"></a>8</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.660585693710996%" headers="mcps1.2.9.1.4 "><p id="p8121154673814"><a name="p8121154673814"></a><a name="p8121154673814"></a>0x1FE00</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.775804128660585%" headers="mcps1.2.9.1.5 "><p id="p512119464387"><a name="p512119464387"></a><a name="p512119464387"></a>0x1FF00</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.967834853576573%" headers="mcps1.2.9.1.6 "><p id="p912194611387"><a name="p912194611387"></a><a name="p912194611387"></a>0x20000</p>
    </td>
    <td class="cellrowborder" valign="top" width="6.721075372059531%" headers="mcps1.2.9.1.7 "><p id="p20271174214619"><a name="p20271174214619"></a><a name="p20271174214619"></a>...</p>
    </td>
    <td class="cellrowborder" valign="top" width="43.61977916466635%" headers="mcps1.2.9.1.8 "><p id="p1449153217134"><a name="p1449153217134"></a><a name="p1449153217134"></a>block0和block2在一个bank group，存在冲突，4拍完成一个Repeat的写入。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   读读冲突
    -   Vector指令多个源操作数同时读到同一个bank group时造成读读冲突，具体分析如下：

        **表 3**  双src场景读读冲突示例

        <a name="table1318881512255"></a>
        <table><thead align="left"><tr id="row8188715102517"><th class="cellrowborder" valign="top" width="6.5786842631473705%" id="mcps1.2.7.1.1"><p id="p111881154254"><a name="p111881154254"></a><a name="p111881154254"></a>序号</p>
        </th>
        <th class="cellrowborder" valign="top" width="12.787442511497702%" id="mcps1.2.7.1.2"><p id="p14188915172513"><a name="p14188915172513"></a><a name="p14188915172513"></a>src0地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="14.077184563087384%" id="mcps1.2.7.1.3"><p id="p4188181542510"><a name="p4188181542510"></a><a name="p4188181542510"></a>src1地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="18.52629474105179%" id="mcps1.2.7.1.4"><p id="p1618871572514"><a name="p1618871572514"></a><a name="p1618871572514"></a>bank</p>
        </th>
        <th class="cellrowborder" valign="top" width="31.36372725454909%" id="mcps1.2.7.1.5"><p id="p31886156253"><a name="p31886156253"></a><a name="p31886156253"></a>bank group</p>
        </th>
        <th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p61881115112512"><a name="p61881115112512"></a><a name="p61881115112512"></a>结论</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row6188915102519"><td class="cellrowborder" valign="top" width="6.5786842631473705%" headers="mcps1.2.7.1.1 "><p id="p10159543259"><a name="p10159543259"></a><a name="p10159543259"></a>示例1</p>
        </td>
        <td class="cellrowborder" valign="top" width="12.787442511497702%" headers="mcps1.2.7.1.2 "><p id="p17188131518251"><a name="p17188131518251"></a><a name="p17188131518251"></a>0x10020</p>
        </td>
        <td class="cellrowborder" valign="top" width="14.077184563087384%" headers="mcps1.2.7.1.3 "><p id="p1818811582515"><a name="p1818811582515"></a><a name="p1818811582515"></a>0x20020</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.52629474105179%" headers="mcps1.2.7.1.4 "><p id="p131881015122512"><a name="p131881015122512"></a><a name="p131881015122512"></a><span>bank_id0 != </span><span>bank_id</span><span>1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p518881513257"><a name="p518881513257"></a><a name="p518881513257"></a><span>bank_group_id0 == bank_group_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p3188215122513"><a name="p3188215122513"></a><a name="p3188215122513"></a>存在冲突。</p>
        </td>
        </tr>
        <tr id="row14188615142520"><td class="cellrowborder" valign="top" width="6.5786842631473705%" headers="mcps1.2.7.1.1 "><p id="p318815156259"><a name="p318815156259"></a><a name="p318815156259"></a>示例2</p>
        </td>
        <td class="cellrowborder" valign="top" width="12.787442511497702%" headers="mcps1.2.7.1.2 "><p id="p151881115192511"><a name="p151881115192511"></a><a name="p151881115192511"></a>0x10020</p>
        </td>
        <td class="cellrowborder" valign="top" width="14.077184563087384%" headers="mcps1.2.7.1.3 "><p id="p12188201519257"><a name="p12188201519257"></a><a name="p12188201519257"></a>0x10000</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.52629474105179%" headers="mcps1.2.7.1.4 "><p id="p71881515182513"><a name="p71881515182513"></a><a name="p71881515182513"></a><span>bank_id0 </span><span>!= </span><span>bank_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p1318817159254"><a name="p1318817159254"></a><a name="p1318817159254"></a><span>bank_group_id0 !</span><span>= </span><span>bank_group_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p11881815182511"><a name="p11881815182511"></a><a name="p11881815182511"></a>无冲突。</p>
        </td>
        </tr>
        </tbody>
        </table>

    -   Vector指令某一个源操作数对应的8个DataBlock（block0-block7）读到同一个bank group时造成读读冲突，具体分析如下：

        **表 4**  单src场景读读冲突示例

        <a name="table332972534717"></a>
        <table><thead align="left"><tr id="row832822516476"><th class="cellrowborder" valign="top" width="6.117919521374119%" id="mcps1.2.9.1.1"><p id="p157111614175011"><a name="p157111614175011"></a><a name="p157111614175011"></a>序号</p>
        </th>
        <th class="cellrowborder" valign="top" width="11.560358969410403%" id="mcps1.2.9.1.2"><p id="p33287255478"><a name="p33287255478"></a><a name="p33287255478"></a>src地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="9.910257647399403%" id="mcps1.2.9.1.3"><p id="p73282255477"><a name="p73282255477"></a><a name="p73282255477"></a>blk_stride</p>
        </th>
        <th class="cellrowborder" valign="top" width="10.798031458071987%" id="mcps1.2.9.1.4"><p id="p1332819254473"><a name="p1332819254473"></a><a name="p1332819254473"></a><span>block0_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="10.633986297404228%" id="mcps1.2.9.1.5"><p id="p33281125164710"><a name="p33281125164710"></a><a name="p33281125164710"></a><span>block1_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="9.234777574061564%" id="mcps1.2.9.1.6"><p id="p632818253478"><a name="p632818253478"></a><a name="p632818253478"></a><span>block2_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="6.754800733378366%" id="mcps1.2.9.1.7"><p id="p20451145754719"><a name="p20451145754719"></a><a name="p20451145754719"></a>...</p>
        </th>
        <th class="cellrowborder" valign="top" width="34.98986779889993%" id="mcps1.2.9.1.8"><p id="p11328625144713"><a name="p11328625144713"></a><a name="p11328625144713"></a>结论</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row7329725144718"><td class="cellrowborder" valign="top" width="6.117919521374119%" headers="mcps1.2.9.1.1 "><p id="p33281425114710"><a name="p33281425114710"></a><a name="p33281425114710"></a>示例1</p>
        </td>
        <td class="cellrowborder" valign="top" width="11.560358969410403%" headers="mcps1.2.9.1.2 "><p id="p1932817255472"><a name="p1932817255472"></a><a name="p1932817255472"></a>0x1FE00</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.910257647399403%" headers="mcps1.2.9.1.3 "><p id="p113284254479"><a name="p113284254479"></a><a name="p113284254479"></a>16</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.798031458071987%" headers="mcps1.2.9.1.4 "><p id="p183285257479"><a name="p183285257479"></a><a name="p183285257479"></a>0x1FE00</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.633986297404228%" headers="mcps1.2.9.1.5 "><p id="p12329925194719"><a name="p12329925194719"></a><a name="p12329925194719"></a>0x20000</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.234777574061564%" headers="mcps1.2.9.1.6 "><p id="p45321853204715"><a name="p45321853204715"></a><a name="p45321853204715"></a>0x20200</p>
        </td>
        <td class="cellrowborder" valign="top" width="6.754800733378366%" headers="mcps1.2.9.1.7 "><p id="p3451457164714"><a name="p3451457164714"></a><a name="p3451457164714"></a>...</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.98986779889993%" headers="mcps1.2.9.1.8 "><p id="p3329152520475"><a name="p3329152520475"></a><a name="p3329152520475"></a>8个<span>DataBlock</span>均在一个bank group下，故全部冲突，8拍完成一个Repeat的读操作。</p>
        </td>
        </tr>
        <tr id="row432932554719"><td class="cellrowborder" valign="top" width="6.117919521374119%" headers="mcps1.2.9.1.1 "><p id="p18329112524713"><a name="p18329112524713"></a><a name="p18329112524713"></a>示例2</p>
        </td>
        <td class="cellrowborder" valign="top" width="11.560358969410403%" headers="mcps1.2.9.1.2 "><p id="p163291925114715"><a name="p163291925114715"></a><a name="p163291925114715"></a>0x1FE00</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.910257647399403%" headers="mcps1.2.9.1.3 "><p id="p1032918252477"><a name="p1032918252477"></a><a name="p1032918252477"></a>8</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.798031458071987%" headers="mcps1.2.9.1.4 "><p id="p1732932594710"><a name="p1732932594710"></a><a name="p1732932594710"></a>0x1FE00</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.633986297404228%" headers="mcps1.2.9.1.5 "><p id="p103291125134710"><a name="p103291125134710"></a><a name="p103291125134710"></a>0x1FF00</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.234777574061564%" headers="mcps1.2.9.1.6 "><p id="p7329182515474"><a name="p7329182515474"></a><a name="p7329182515474"></a>0x20000</p>
        </td>
        <td class="cellrowborder" valign="top" width="6.754800733378366%" headers="mcps1.2.9.1.7 "><p id="p1945115714477"><a name="p1945115714477"></a><a name="p1945115714477"></a>...</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.98986779889993%" headers="mcps1.2.9.1.8 "><p id="p9329192544712"><a name="p9329192544712"></a><a name="p9329192544712"></a>block0和block2在同一个bank group下，存在冲突，4拍完成一个Repeat。</p>
        </td>
        </tr>
        </tbody>
        </table>

> **说明：** 
>通过msProf工具可以进行资源冲突占比的相关性能数据采集。
>工具的具体使用方法和资源冲突占比文件性能数据文件说明请参考《算子开发工具用户指南》。

## 如何避免bank冲突<a name="section12501642143515"></a>

避免bank冲突的方法有两种：**优化计算逻辑**和**优化地址分配**。

-   **优化计算逻辑**

    对一个shape为\(8, 16, 16\)的输入做\(1, 0, 2\)的transpose操作，输出shape为\(16, 8, 16\)。通过将计算逻辑由“跳读，连续写”修改为“连续读，跳写”可避免冲突问题。实现方案对比如下：

    <a name="table12921549195512"></a>
    <table><thead align="left"><tr id="row1229364945511"><th class="cellrowborder" valign="top" width="6.813978389954251%" id="mcps1.1.4.1.1"><p id="p2081249145715"><a name="p2081249145715"></a><a name="p2081249145715"></a>实现方案</p>
    </th>
    <th class="cellrowborder" valign="top" width="42.6652389759564%" id="mcps1.1.4.1.2"><p id="p2029374985519"><a name="p2029374985519"></a><a name="p2029374985519"></a>原始实现</p>
    </th>
    <th class="cellrowborder" valign="top" width="50.520782634089365%" id="mcps1.1.4.1.3"><p id="p152931149115516"><a name="p152931149115516"></a><a name="p152931149115516"></a>优化实现</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1629374995517"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p108124995710"><a name="p108124995710"></a><a name="p108124995710"></a>实现方法</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><p id="p55722115019"><a name="p55722115019"></a><a name="p55722115019"></a>跳读，连续写</p>
    <p id="p144638315712"><a name="p144638315712"></a><a name="p144638315712"></a>同一Repeat内输入的8个DataBlock都在同一个bank group而发生读读冲突。</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><p id="p2293249135516"><a name="p2293249135516"></a><a name="p2293249135516"></a>连续读，跳写</p>
    <p id="p6641340414"><a name="p6641340414"></a><a name="p6641340414"></a>同一个Repeat内输入的8个DataBlock不在同一个bank group内，避免了读读冲突。</p>
    </td>
    </tr>
    <tr id="row14922142214585"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p6922182210587"><a name="p6922182210587"></a><a name="p6922182210587"></a>示意图</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><p id="p10922222205810"><a name="p10922222205810"></a><a name="p10922222205810"></a><a name="image1757423545813"></a><a name="image1757423545813"></a><span><img class="eddx" id="image1757423545813" src="figures/矩阵编程逻辑位置示意图-48.png" width="422.94" height="376.36672500000003"></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><p id="p1922622115813"><a name="p1922622115813"></a><a name="p1922622115813"></a><a name="image6621154316580"></a><a name="image6621154316580"></a><span><img class="eddx" id="image6621154316580" src="figures/矩阵编程逻辑位置示意图-49.png" width="489.77250000000004" height="363.174721"></span></p>
    </td>
    </tr>
    <tr id="row3293124918559"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p12812993573"><a name="p12812993573"></a><a name="p12812993573"></a>示例代码</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><a name="screen924835613570"></a><a name="screen924835613570"></a><pre class="screen" codetype="Cpp" id="screen924835613570">uint64_t mask = 128;
    UnaryRepeatParams params;
    params.dstBlkStride  = 1;
    params.srcBlkStride = 16;
    for(uint32_t i=0; i&lt;16; i++)   {
        AscendC::Adds(dstLocal[i * 128], srcLocal[i * 16], 0, mask, 1, params);
    }</pre>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><a name="screen271414925813"></a><a name="screen271414925813"></a><pre class="screen" codetype="Cpp" id="screen271414925813">uint64_t mask = 128;
    UnaryRepeatParams params;
    params.dstBlkStride  = 8;
    params.srcBlkStride = 1;
    for(uint32_t i=0; i&lt;8; i++)   {
        AscendC::Adds(dstLocal[i * 16], srcLocal[i * 256], 0, mask, 2, params);
    }</pre>
    </td>
    </tr>
    </tbody>
    </table>

-   **优化地址分配**

    实现连续4096个float元素的加法z = x + y，通过在内存分配时适当扩大内存，保证在一个Repeat内，x和y不会同时出现在同一个bank group内，x/y和z不会同时出现同一个bank内。完整样例可参考[避免bank冲突样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/4_bank_conflict)。

    实现方案对比如下：

    <a name="table839512544411"></a>
    <table><thead align="left"><tr id="row73965541942"><th class="cellrowborder" valign="top" width="6.813978389954251%" id="mcps1.1.4.1.1"><p id="p63963545416"><a name="p63963545416"></a><a name="p63963545416"></a>实现方案</p>
    </th>
    <th class="cellrowborder" valign="top" width="42.01304390148935%" id="mcps1.1.4.1.2"><p id="p539610544415"><a name="p539610544415"></a><a name="p539610544415"></a>原始实现</p>
    </th>
    <th class="cellrowborder" valign="top" width="51.17297770855641%" id="mcps1.1.4.1.3"><p id="p6396185413418"><a name="p6396185413418"></a><a name="p6396185413418"></a>优化实现</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row143961754148"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p43962544416"><a name="p43962544416"></a><a name="p43962544416"></a>实现方法</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><p id="p33964541745"><a name="p33964541745"></a><a name="p33964541745"></a>不做地址优化，直接使用InitBuffer分配内存，各个Tensor的地址分别为：</p>
    <p id="p17357131764"><a name="p17357131764"></a><a name="p17357131764"></a>x：起始地址0x0，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p15372720618"><a name="p15372720618"></a><a name="p15372720618"></a>y：起始地址0x4000，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p138635471667"><a name="p138635471667"></a><a name="p138635471667"></a>z：起始地址0x8000，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p1927691223819"><a name="p1927691223819"></a><a name="p1927691223819"></a>在一个Repeat内，x和y同时读同一个bank group，x/y和z同时读写同一个bank。</p>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><p id="p330914381671"><a name="p330914381671"></a><a name="p330914381671"></a>优化地址，使用InitBuffer分配内存时适当扩大内存申请，各个Tensor的地址分别为：</p>
    <p id="p326294904420"><a name="p326294904420"></a><a name="p326294904420"></a>x：起始地址0x0，tensor长度为(4096 * sizeof(float) + 256)字节</p>
    <p id="p153096389720"><a name="p153096389720"></a><a name="p153096389720"></a>y：起始地址0x4100，tensor长度为(64 * 1024 - (4096 * sizeof(float) + 256))字节</p>
    <p id="p1130923814713"><a name="p1130923814713"></a><a name="p1130923814713"></a>z：起始地址0x10000，tensor长度为4096 * sizeof(float) 字节</p>
    <p id="p9897551154415"><a name="p9897551154415"></a><a name="p9897551154415"></a>x多申请256字节，避免一个Repeat内x y同时读同一个bank group；y多申请空间，确保z不会和x/y落入同一个bank</p>
    </td>
    </tr>
    <tr id="row83963547410"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p103961544412"><a name="p103961544412"></a><a name="p103961544412"></a>示意图</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><p id="p575310529216"><a name="p575310529216"></a><a name="p575310529216"></a><a name="image5993165819214"></a><a name="image5993165819214"></a><span><img class="eddx" id="image5993165819214" src="figures/矩阵编程逻辑位置示意图-50.png" width="357.105" height="231.19556250000002"></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><p id="p20186156039"><a name="p20186156039"></a><a name="p20186156039"></a><a name="image35231563314"></a><a name="image35231563314"></a><span><img class="eddx" id="image35231563314" src="figures/矩阵编程逻辑位置示意图-51.png" width="388.02750000000003" height="231.42000000000002"></span></p>
    </td>
    </tr>
    <tr id="row1539620548413"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p43961554645"><a name="p43961554645"></a><a name="p43961554645"></a>示例代码</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><a name="screen35066415507"></a><a name="screen35066415507"></a><pre class="screen" codetype="Cpp" id="screen35066415507">pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));</pre>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><a name="screen51721010145115"></a><a name="screen51721010145115"></a><pre class="screen" codetype="Cpp" id="screen51721010145115">pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256); // 多申请256字节
    pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256)); //多申请空间，确保z不会和x/y落入同一个bank， 64 * 1024是16个bank group的空间，4096 * sizeof(float) + 256是x所占的空间
    pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));</pre>
    </td>
    </tr>
    </tbody>
    </table>

