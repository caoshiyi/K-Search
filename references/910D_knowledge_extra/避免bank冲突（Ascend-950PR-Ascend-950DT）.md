# 避免bank冲突（Ascend 950PR/Ascend 950DT）<a name="ZH-CN_TOPIC_0000002554329041"></a>

为了提高数据访问的效率和吞吐量，Unified Buffer采用了bank（大小相等的内存模块）结构设计。Unified Buffer总大小为256K，划分为16个bank。每个bank由512行组成，每行长度为32B。这16个bank进一步组织为8个bank group，每个bank group包含2个bank，例如bank7和bank15组成一个bank group。

**图 1**  bank结构示意图（图中箭头方向表示内存排布的顺序）<a name="fig873359165316"></a>  
<!-- img2text -->
```
                         ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
                         │  bank  │  │  bank  │  │  bank  │  │  bank  │  │  bank  │  │  bank  │  │  bank  │  │  bank  │
                         │ group0 │  │ group1 │  │ group2 │  │ group3 │  │ group4 │  │ group5 │  │ group6 │  │ group7 │
                         └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘

  ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│
  │                  │                  │                  │                  │                  │                  │                  │                  │
  │      bank0       │      bank1       │      bank2       │      bank3       │      bank4       │      bank5       │      bank6       │      bank7       │
  │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │
  │                  │                  │                  │                  │                  │                  │                  │                  │
  └──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
  ↑                                                                                                                                                          ↑
  │                                                                                                                                                          │
512                                                                                                                                                        512
  │                                                                                                                                                          │
  ↓                                                                                                                                                          ↓

  ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│<────── 32B ─────>│
  │                  │                  │                  │                  │                  │                  │                  │                  │
  │      bank8       │      bank9       │      bank10      │      bank11      │      bank12      │      bank13      │      bank14      │      bank15      │
  │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │       16KB       │
  │                  │                  │                  │                  │                  │                  │                  │                  │
  └──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
  ↑                                                                                                                                                          ↑
  │                                                                                                                                                          │
512                                                                                                                                                        512
  │                                                                                                                                                          │
  ↓                                                                                                                                                          ↓

                         ↖──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────↗
```

说明:
- 图中箭头方向表示内存排布的顺序。
- 每个 bank 大小为 16KB。
- 每个 bank 的横向标注为 32B。
- 每列上下两个 bank 构成一个 bank group：
  - bank group0: bank0 + bank8
  - bank group1: bank1 + bank9
  - bank group2: bank2 + bank10
  - bank group3: bank3 + bank11
  - bank group4: bank4 + bank12
  - bank group5: bank5 + bank13
  - bank group6: bank6 + bank14
  - bank group7: bank7 + bank15
- 左右两侧的 512 表示每个 bank 有 512 行。

每个bank可以独立地进行数据的读写操作，允许多个数据请求同时进行。然而，当多个读写操作试图同时访问同一个bank，由于硬件资源的限制，这些操作必须排队等待，会导致bank冲突，引起性能下降。

具体来说，Vector计算单元每拍（一个指令周期）能够从每个bank group中读取或写入一行数据。当多个读写操作试图同时访问同一个bank，Vector计算单元无法在同一个周期内处理所有请求，导致这些请求排队等待。这种排队增加了数据访问的延迟，降低了系统的整体性能。

## bank冲突的典型场景<a name="section9689957379"></a>

bank冲突主要可以分为以下三种场景：

-   **读写冲突**：读操作和写操作同时尝试访问同一个bank。
-   **写写冲突**：多个写操作同时尝试访问同一个bank group。
-   **读读冲突**：两个读操作同时尝试访问同一个bank，或者两个以上读操作同时尝试访问同一个bank group。

下文给出了一些具体的示例，假设，0x10000地址在bank0上，0x10020在bank1上，如下图所示：

**图 2**  地址分配示意图<a name="fig1750123073213"></a>  
<!-- img2text -->
```
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ bank       │ │ bank       │ │ bank       │ │ bank       │
│ group0     │ │ group1     │ │ group...   │ │ group7     │
├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤
│            │ │            │ │            │ │            │
│   bank     │ │   bank     │ │   bank     │ │   bank     │
│    0       │ │    1       │ │    ...     │ │    7       │
│            │ │            │ │            │ │            │
├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤
│ 0x10000    │ │ 0x10020    │ │            │ │            │
├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤
│            │ │            │ │            │ │            │
│   bank     │ │   bank     │ │   bank     │ │   bank     │
│    8       │ │    9       │ │    ...     │ │    15      │
│            │ │            │ │            │ │            │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
                 0x20020
```

-   读写冲突示例

    Vector指令的源操作数src和目的操作数dst同时读写到同一个bank时造成读写冲突，具体分析如下：

    **表 1**  读写冲突示例

    <a name="table178973521409"></a>
    <table><thead align="left"><tr id="row20897752164015"><th class="cellrowborder" valign="top" width="5.9988002399520095%" id="mcps1.2.7.1.1"><p id="p13897195244019"><a name="p13897195244019"></a><a name="p13897195244019"></a>序号</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.948010397920417%" id="mcps1.2.7.1.2"><p id="p3897155224010"><a name="p3897155224010"></a><a name="p3897155224010"></a>src地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.507898420315936%" id="mcps1.2.7.1.3"><p id="p13897252164011"><a name="p13897252164011"></a><a name="p13897252164011"></a>dst地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="25.51489702059588%" id="mcps1.2.7.1.4"><p id="p138976528405"><a name="p138976528405"></a><a name="p138976528405"></a>bank</p>
    </th>
    <th class="cellrowborder" valign="top" width="31.36372725454909%" id="mcps1.2.7.1.5"><p id="p689795214408"><a name="p689795214408"></a><a name="p689795214408"></a>bank group</p>
    </th>
    <th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p98976528403"><a name="p98976528403"></a><a name="p98976528403"></a>结论</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row10897145224010"><td class="cellrowborder" valign="top" width="5.9988002399520095%" headers="mcps1.2.7.1.1 "><p id="p8897185254012"><a name="p8897185254012"></a><a name="p8897185254012"></a>示例1</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.948010397920417%" headers="mcps1.2.7.1.2 "><p id="p12897155215406"><a name="p12897155215406"></a><a name="p12897155215406"></a>0x10020</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.507898420315936%" headers="mcps1.2.7.1.3 "><p id="p19897165213403"><a name="p19897165213403"></a><a name="p19897165213403"></a>0x10000</p>
    </td>
    <td class="cellrowborder" valign="top" width="25.51489702059588%" headers="mcps1.2.7.1.4 "><p id="p1489705220408"><a name="p1489705220408"></a><a name="p1489705220408"></a><span>bank_id0 != </span><span>bank_id</span><span>1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p4897155264016"><a name="p4897155264016"></a><a name="p4897155264016"></a><span>bank_group_id0 != bank_group_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1589719525406"><a name="p1589719525406"></a><a name="p1589719525406"></a>src地址和dst地址分别属于bank0和bank1，故无冲突。</p>
    </td>
    </tr>
    <tr id="row1489775264016"><td class="cellrowborder" valign="top" width="5.9988002399520095%" headers="mcps1.2.7.1.1 "><p id="p12897152184019"><a name="p12897152184019"></a><a name="p12897152184019"></a>示例2</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.948010397920417%" headers="mcps1.2.7.1.2 "><p id="p108981152174015"><a name="p108981152174015"></a><a name="p108981152174015"></a>0x10020</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.507898420315936%" headers="mcps1.2.7.1.3 "><p id="p1389811520408"><a name="p1389811520408"></a><a name="p1389811520408"></a>0x10120</p>
    </td>
    <td class="cellrowborder" valign="top" width="25.51489702059588%" headers="mcps1.2.7.1.4 "><p id="p389885274014"><a name="p389885274014"></a><a name="p389885274014"></a><span>bank_id0 </span><span>== </span><span>bank_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p7898155214011"><a name="p7898155214011"></a><a name="p7898155214011"></a><span>bank_group_id0 =</span><span>= </span><span>bank_group_id1</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p989815214018"><a name="p989815214018"></a><a name="p989815214018"></a>src地址和dst地址的地址都在bank0，故存在冲突。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   写写冲突示例

    Vector指令目的操作数dst对应的8个DataBlock（block0-block7）同时写到一个bank group时造成写写冲突，具体分析如下：

    **表 2**  写写冲突示例

    <a name="table15913191235615"></a>
    <table><thead align="left"><tr id="row891371215561"><th class="cellrowborder" valign="top" width="5.789726356216995%" id="mcps1.2.9.1.1"><p id="p1913412195619"><a name="p1913412195619"></a><a name="p1913412195619"></a>序号</p>
    </th>
    <th class="cellrowborder" valign="top" width="7.959673547767644%" id="mcps1.2.9.1.2"><p id="p19133129560"><a name="p19133129560"></a><a name="p19133129560"></a>dst地址</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.505520883341337%" id="mcps1.2.9.1.3"><p id="p991371215560"><a name="p991371215560"></a><a name="p991371215560"></a>blk_stride</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.660585693710996%" id="mcps1.2.9.1.4"><p id="p10913171219568"><a name="p10913171219568"></a><a name="p10913171219568"></a><span>block0_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="8.775804128660585%" id="mcps1.2.9.1.5"><p id="p29131712145616"><a name="p29131712145616"></a><a name="p29131712145616"></a><span>block1_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="8.967834853576573%" id="mcps1.2.9.1.6"><p id="p4913151218567"><a name="p4913151218567"></a><a name="p4913151218567"></a><span>block2_addr </span></p>
    </th>
    <th class="cellrowborder" valign="top" width="6.721075372059531%" id="mcps1.2.9.1.7"><p id="p109131112115617"><a name="p109131112115617"></a><a name="p109131112115617"></a>...</p>
    </th>
    <th class="cellrowborder" valign="top" width="43.61977916466635%" id="mcps1.2.9.1.8"><p id="p491301211565"><a name="p491301211565"></a><a name="p491301211565"></a>结论</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row2091314128569"><td class="cellrowborder" valign="top" width="5.789726356216995%" headers="mcps1.2.9.1.1 "><p id="p8913151214562"><a name="p8913151214562"></a><a name="p8913151214562"></a>示例1</p>
    </td>
    <td class="cellrowborder" valign="top" width="7.959673547767644%" headers="mcps1.2.9.1.2 "><p id="p129131512125612"><a name="p129131512125612"></a><a name="p129131512125612"></a>0x10000</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.505520883341337%" headers="mcps1.2.9.1.3 "><p id="p3913191218564"><a name="p3913191218564"></a><a name="p3913191218564"></a>8</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.660585693710996%" headers="mcps1.2.9.1.4 "><p id="p1191321220560"><a name="p1191321220560"></a><a name="p1191321220560"></a>0x10000</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.775804128660585%" headers="mcps1.2.9.1.5 "><p id="p109132012115614"><a name="p109132012115614"></a><a name="p109132012115614"></a>0x10100</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.967834853576573%" headers="mcps1.2.9.1.6 "><p id="p1891371218569"><a name="p1891371218569"></a><a name="p1891371218569"></a>0x10200</p>
    </td>
    <td class="cellrowborder" valign="top" width="6.721075372059531%" headers="mcps1.2.9.1.7 "><p id="p1791371217565"><a name="p1791371217565"></a><a name="p1791371217565"></a>...</p>
    </td>
    <td class="cellrowborder" valign="top" width="43.61977916466635%" headers="mcps1.2.9.1.8 "><p id="p1691311124564"><a name="p1691311124564"></a><a name="p1691311124564"></a>8个DataBlock均在一个bank group下，故全部冲突，8拍完成一个Repeat的写入。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   读读冲突
    -   Vector指令两个源操作数同时读到同一个bank时造成读读冲突，具体分析如下：

        **表 3**  双src场景读读冲突示例

        <a name="table983101761813"></a>
        <table><thead align="left"><tr id="row7834178187"><th class="cellrowborder" valign="top" width="6.5786842631473705%" id="mcps1.2.7.1.1"><p id="p1283121713188"><a name="p1283121713188"></a><a name="p1283121713188"></a>序号</p>
        </th>
        <th class="cellrowborder" valign="top" width="12.76744651069786%" id="mcps1.2.7.1.2"><p id="p583117141810"><a name="p583117141810"></a><a name="p583117141810"></a>src0地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="14.097180563887221%" id="mcps1.2.7.1.3"><p id="p198312171181"><a name="p198312171181"></a><a name="p198312171181"></a>src1地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="18.52629474105179%" id="mcps1.2.7.1.4"><p id="p683917111815"><a name="p683917111815"></a><a name="p683917111815"></a>bank</p>
        </th>
        <th class="cellrowborder" valign="top" width="31.36372725454909%" id="mcps1.2.7.1.5"><p id="p1583151719187"><a name="p1583151719187"></a><a name="p1583151719187"></a>bank group</p>
        </th>
        <th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="p883171712184"><a name="p883171712184"></a><a name="p883171712184"></a>结论</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row1683121715185"><td class="cellrowborder" valign="top" width="6.5786842631473705%" headers="mcps1.2.7.1.1 "><p id="p13839177184"><a name="p13839177184"></a><a name="p13839177184"></a>示例1</p>
        </td>
        <td class="cellrowborder" valign="top" width="12.76744651069786%" headers="mcps1.2.7.1.2 "><p id="p583121718188"><a name="p583121718188"></a><a name="p583121718188"></a>0x10000</p>
        </td>
        <td class="cellrowborder" valign="top" width="14.097180563887221%" headers="mcps1.2.7.1.3 "><p id="p198321717189"><a name="p198321717189"></a><a name="p198321717189"></a>0x10100</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.52629474105179%" headers="mcps1.2.7.1.4 "><p id="p2836174181"><a name="p2836174181"></a><a name="p2836174181"></a><span>bank_id0 == </span><span>bank_id</span><span>1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p8831178183"><a name="p8831178183"></a><a name="p8831178183"></a><span>bank_group_id0 == bank_group_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1841179186"><a name="p1841179186"></a><a name="p1841179186"></a>存在冲突。</p>
        </td>
        </tr>
        <tr id="row1484151719181"><td class="cellrowborder" valign="top" width="6.5786842631473705%" headers="mcps1.2.7.1.1 "><p id="p1084417181817"><a name="p1084417181817"></a><a name="p1084417181817"></a>示例2</p>
        </td>
        <td class="cellrowborder" valign="top" width="12.76744651069786%" headers="mcps1.2.7.1.2 "><p id="p10849172181"><a name="p10849172181"></a><a name="p10849172181"></a>0x10000</p>
        </td>
        <td class="cellrowborder" valign="top" width="14.097180563887221%" headers="mcps1.2.7.1.3 "><p id="p1284131701810"><a name="p1284131701810"></a><a name="p1284131701810"></a>0x10020</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.52629474105179%" headers="mcps1.2.7.1.4 "><p id="p12849176181"><a name="p12849176181"></a><a name="p12849176181"></a><span>bank_id0 </span><span>!= </span><span>bank_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="31.36372725454909%" headers="mcps1.2.7.1.5 "><p id="p784181761810"><a name="p784181761810"></a><a name="p784181761810"></a><span>bank_group_id0 !</span><span>= </span><span>bank_group_id1</span></p>
        </td>
        <td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1684121791810"><a name="p1684121791810"></a><a name="p1684121791810"></a>无冲突。</p>
        </td>
        </tr>
        </tbody>
        </table>

    -   Vector指令某一个源操作数对应的8个DataBlock\(block0-block7）读到同一个bank时造成读读冲突，具体分析如下：

        **表 4**  单src场景读读冲突示例

        <a name="table1055918277182"></a>
        <table><thead align="left"><tr id="row05601327161818"><th class="cellrowborder" valign="top" width="6.117919521374119%" id="mcps1.2.9.1.1"><p id="p856062714185"><a name="p856062714185"></a><a name="p856062714185"></a>序号</p>
        </th>
        <th class="cellrowborder" valign="top" width="11.560358969410403%" id="mcps1.2.9.1.2"><p id="p556016272186"><a name="p556016272186"></a><a name="p556016272186"></a>src地址</p>
        </th>
        <th class="cellrowborder" valign="top" width="9.910257647399403%" id="mcps1.2.9.1.3"><p id="p1560127181817"><a name="p1560127181817"></a><a name="p1560127181817"></a>blk_stride</p>
        </th>
        <th class="cellrowborder" valign="top" width="10.798031458071987%" id="mcps1.2.9.1.4"><p id="p165607278181"><a name="p165607278181"></a><a name="p165607278181"></a><span>block0_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="10.633986297404228%" id="mcps1.2.9.1.5"><p id="p856022717184"><a name="p856022717184"></a><a name="p856022717184"></a><span>block1_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="9.234777574061564%" id="mcps1.2.9.1.6"><p id="p11560327181814"><a name="p11560327181814"></a><a name="p11560327181814"></a><span>block2_addr </span></p>
        </th>
        <th class="cellrowborder" valign="top" width="6.754800733378366%" id="mcps1.2.9.1.7"><p id="p175601827191811"><a name="p175601827191811"></a><a name="p175601827191811"></a>...</p>
        </th>
        <th class="cellrowborder" valign="top" width="34.98986779889993%" id="mcps1.2.9.1.8"><p id="p9560327191810"><a name="p9560327191810"></a><a name="p9560327191810"></a>结论</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row1456042701815"><td class="cellrowborder" valign="top" width="6.117919521374119%" headers="mcps1.2.9.1.1 "><p id="p656052771820"><a name="p656052771820"></a><a name="p656052771820"></a>示例1</p>
        </td>
        <td class="cellrowborder" valign="top" width="11.560358969410403%" headers="mcps1.2.9.1.2 "><p id="p1256016274187"><a name="p1256016274187"></a><a name="p1256016274187"></a>0x10000</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.910257647399403%" headers="mcps1.2.9.1.3 "><p id="p1456022741810"><a name="p1456022741810"></a><a name="p1456022741810"></a>8</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.798031458071987%" headers="mcps1.2.9.1.4 "><p id="p1956032761819"><a name="p1956032761819"></a><a name="p1956032761819"></a>0x10000</p>
        </td>
        <td class="cellrowborder" valign="top" width="10.633986297404228%" headers="mcps1.2.9.1.5 "><p id="p125607279187"><a name="p125607279187"></a><a name="p125607279187"></a>0x10100</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.234777574061564%" headers="mcps1.2.9.1.6 "><p id="p1560152716187"><a name="p1560152716187"></a><a name="p1560152716187"></a>0x10200</p>
        </td>
        <td class="cellrowborder" valign="top" width="6.754800733378366%" headers="mcps1.2.9.1.7 "><p id="p11560202719180"><a name="p11560202719180"></a><a name="p11560202719180"></a>...</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.98986779889993%" headers="mcps1.2.9.1.8 "><p id="p156052714183"><a name="p156052714183"></a><a name="p156052714183"></a>8个<span>DataBlock</span>均在一个bank下，故全部冲突，8拍完成一个Repeat的读操作。</p>
        </td>
        </tr>
        </tbody>
        </table>

## 如何避免bank冲突<a name="section12501642143515"></a>

避免bank冲突的方法有两种：**优化计算逻辑**和**优化地址分配**。

-   **优化计算逻辑**

    对一个数据类型为float，shape为\(16, 64\)的输入每个元素加1。通过将计算逻辑由逐列计算改为逐行计算可避免同一Repeat下的冲突问题，实现方案对比如下：

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
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><p id="p115401884217"><a name="p115401884217"></a><a name="p115401884217"></a>逐列计算，同一Repeat内输入的8个DataBlock都在同一个bank而发生读读冲突。</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><p id="p6641340414"><a name="p6641340414"></a><a name="p6641340414"></a>逐行计算，同一个Repeat内输入的8个DataBlock不在同一个bank内，避免了同一Repeat内的读读冲突。</p>
    </td>
    </tr>
    <tr id="row14922142214585"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p6922182210587"><a name="p6922182210587"></a><a name="p6922182210587"></a>示意图</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><p id="p1273333434818"><a name="p1273333434818"></a><a name="p1273333434818"></a><a name="image47338343486"></a><a name="image47338343486"></a><span><img class="eddx" id="image47338343486" src="figures/矩阵编程逻辑位置示意图-54.png"></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><p id="p1074101154916"><a name="p1074101154916"></a><a name="p1074101154916"></a><a name="image147416115491"></a><a name="image147416115491"></a><span><img class="eddx" id="image147416115491" src="figures/矩阵编程逻辑位置示意图-55.png"></span></p>
    </td>
    </tr>
    <tr id="row3293124918559"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p12812993573"><a name="p12812993573"></a><a name="p12812993573"></a>示例代码</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.6652389759564%" headers="mcps1.1.4.1.2 "><a name="screen924835613570"></a><a name="screen924835613570"></a><pre class="screen" codetype="Cpp" id="screen924835613570">uint64_t mask = 64;
    AscendC::UnaryRepeatParams params;
    params.dstBlkStride = 8;
    params.srcBlkStride = 8;
    for(uint16_t i = 0; i &lt; 8; ++i){
        AscendC::Adds(dst[i * 8], src[i * 8], 1, mask, 1, params);
    }</pre>
    </td>
    <td class="cellrowborder" valign="top" width="50.520782634089365%" headers="mcps1.1.4.1.3 "><a name="screen271414925813"></a><a name="screen271414925813"></a><pre class="screen" codetype="Cpp" id="screen271414925813">uint64_t mask = 64;
    AscendC::UnaryRepeatParams params;
    params.dstBlkStride = 1;
    params.srcBlkStride = 1;
    for(uint16_t i = 0; i &lt; 8; ++i){
        AscendC::Adds(dst[i * 64], src[i * 64], 1, mask, 1, params);
    }</pre>
    </td>
    </tr>
    </tbody>
    </table>

-   **优化地址分配**

    实现连续4096个float元素的加法z = x + y，通过在内存分配时适当扩大内存，保证在一个Repeat内，x/y和z不会同时出现同一个bank内。

    实现方案对比如下：

    <a name="table82050585313"></a>
    <table><thead align="left"><tr id="row1320525823115"><th class="cellrowborder" valign="top" width="6.813978389954251%" id="mcps1.1.4.1.1"><p id="p6205125853117"><a name="p6205125853117"></a><a name="p6205125853117"></a>实现方案</p>
    </th>
    <th class="cellrowborder" valign="top" width="42.01304390148935%" id="mcps1.1.4.1.2"><p id="p920565815314"><a name="p920565815314"></a><a name="p920565815314"></a>原始实现</p>
    </th>
    <th class="cellrowborder" valign="top" width="51.17297770855641%" id="mcps1.1.4.1.3"><p id="p10205105819318"><a name="p10205105819318"></a><a name="p10205105819318"></a>优化实现</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1220513586316"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p320516586313"><a name="p320516586313"></a><a name="p320516586313"></a>实现方法</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><p id="p5205155873113"><a name="p5205155873113"></a><a name="p5205155873113"></a>不做地址优化，直接使用InitBuffer分配内存，各个Tensor的地址分别为：</p>
    <p id="p120535813115"><a name="p120535813115"></a><a name="p120535813115"></a>x：起始地址0x00000，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p11205125814318"><a name="p11205125814318"></a><a name="p11205125814318"></a>y：起始地址0x04000，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p420545893110"><a name="p420545893110"></a><a name="p420545893110"></a>z：起始地址0x08000，tensor长度为4096 * sizeof(float)字节</p>
    <p id="p10205175810318"><a name="p10205175810318"></a><a name="p10205175810318"></a>在一个Repeat内，x和y同时读同一个bank group，x/y和z同时读写同一个bank。</p>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><p id="p1720555893117"><a name="p1720555893117"></a><a name="p1720555893117"></a>优化地址，使用InitBuffer分配内存时适当扩大内存申请，各个Tensor的地址分别为：</p>
    <p id="p1205105810316"><a name="p1205105810316"></a><a name="p1205105810316"></a>x：起始地址0x00000，tensor长度为4096 * sizeof(float) 字节</p>
    <p id="p420575820318"><a name="p420575820318"></a><a name="p420575820318"></a>y：起始地址0x04000，tensor长度为(8 * 16 * 1024 - (4096 * sizeof(float) )字节</p>
    <p id="p120525813116"><a name="p120525813116"></a><a name="p120525813116"></a>z：起始地址0x20000，tensor长度为4096 * sizeof(float) 字节</p>
    <p id="p162053589314"><a name="p162053589314"></a><a name="p162053589314"></a>y多申请空间，确保z不会和x/y落入同一个bank。</p>
    </td>
    </tr>
    <tr id="row92051958113118"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p820614589313"><a name="p820614589313"></a><a name="p820614589313"></a>示意图</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><p id="p11206958143119"><a name="p11206958143119"></a><a name="p11206958143119"></a><a name="image17957162514335"></a><a name="image17957162514335"></a><span><img class="eddx" id="image17957162514335" src="figures/矩阵编程逻辑位置示意图-56.png"></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><p id="p1979513571429"><a name="p1979513571429"></a><a name="p1979513571429"></a><a name="image6795145717421"></a><a name="image6795145717421"></a><span><img class="eddx" id="image6795145717421" src="figures/矩阵编程逻辑位置示意图-57.png"></span></p>
    </td>
    </tr>
    <tr id="row1520625893111"><td class="cellrowborder" valign="top" width="6.813978389954251%" headers="mcps1.1.4.1.1 "><p id="p22061858183118"><a name="p22061858183118"></a><a name="p22061858183118"></a>示例代码</p>
    </td>
    <td class="cellrowborder" valign="top" width="42.01304390148935%" headers="mcps1.1.4.1.2 "><a name="screen62061858143118"></a><a name="screen62061858143118"></a><pre class="screen" codetype="Cpp" id="screen62061858143118">pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));</pre>
    </td>
    <td class="cellrowborder" valign="top" width="51.17297770855641%" headers="mcps1.1.4.1.3 "><a name="screen7206858113117"></a><a name="screen7206858113117"></a><pre class="screen" codetype="Cpp" id="screen7206858113117">constexpr int32_t TOTAL_LENGTH = 1024 * 4;
    constexpr int32_t BUFFER_NUM = 1;
    constexpr int32_t BANKGROUP_SIZE  =  1024 * 128; 
    ...
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TOTAL_LENGTH * sizeof(float));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, BANKGROUP_SIZE - TOTAL_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TOTAL_LENGTH * sizeof(float));</pre>
    </td>
    </tr>
    </tbody>
    </table>

