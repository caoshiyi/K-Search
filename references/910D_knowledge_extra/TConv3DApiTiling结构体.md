# TConv3DApiTiling结构体<a name="ZH-CN_TOPIC_0000002523343716"></a>

TConv3DApiTiling结构体包含Conv3D算子规格信息及Tiling切分算法的相关参数，被传递给Conv3D Kernel侧，用于数据切分、数据搬运和计算等。TConv3DApiTiling结构体的参数说明见[表1](#table18244199192620)。

用户通过调用[GetTiling](GetTiling-126.md)接口获取TConv3DApiTiling结构体，具体流程请参考[Conv3D Tiling使用说明](Conv3D-Tiling使用说明.md)。当前暂不支持用户自定义配置TConv3DApiTiling结构体中的参数。

**表 1**  TConv3DApiTiling结构说明

<a name="table18244199192620"></a>
<table><thead align="left"><tr id="row1232218913265"><th class="cellrowborder" valign="top" width="32.65%" id="mcps1.2.4.1.1"><p id="p73221912260"><a name="p73221912260"></a><a name="p73221912260"></a><strong id="b10322109152611"><a name="b10322109152611"></a><a name="b10322109152611"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="14.29%" id="mcps1.2.4.1.2"><p id="p432209122610"><a name="p432209122610"></a><a name="p432209122610"></a><strong id="b032217932617"><a name="b032217932617"></a><a name="b032217932617"></a>数据类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="53.059999999999995%" id="mcps1.2.4.1.3"><p id="p632289192615"><a name="p632289192615"></a><a name="p632289192615"></a><strong id="b14322149112619"><a name="b14322149112619"></a><a name="b14322149112619"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1332211911262"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1732249122611"><a name="p1732249122611"></a><a name="p1732249122611"></a>groups</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p532216910266"><a name="p532216910266"></a><a name="p532216910266"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p432259132611"><a name="p432259132611"></a><a name="p432259132611"></a>预留参数，当前仅支持为1。</p>
</td>
</tr>
<tr id="row432212911261"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1532314912262"><a name="p1532314912262"></a><a name="p1532314912262"></a>singleCoreDo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p63235913263"><a name="p63235913263"></a><a name="p63235913263"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p732312982616"><a name="p732312982616"></a><a name="p732312982616"></a>单核上处理的Dout大小。</p>
</td>
</tr>
<tr id="row1532310992614"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p123237915262"><a name="p123237915262"></a><a name="p123237915262"></a>singleCoreCo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1232316982619"><a name="p1232316982619"></a><a name="p1232316982619"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p23231295264"><a name="p23231295264"></a><a name="p23231295264"></a>单核上处理的Cout大小。</p>
</td>
</tr>
<tr id="row73238992615"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p23238911267"><a name="p23238911267"></a><a name="p23238911267"></a>singleCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p16323109192614"><a name="p16323109192614"></a><a name="p16323109192614"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p13237942620"><a name="p13237942620"></a><a name="p13237942620"></a>单核上处理的M大小。</p>
</td>
</tr>
<tr id="row113232982611"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p14323892261"><a name="p14323892261"></a><a name="p14323892261"></a>orgDo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p153231493261"><a name="p153231493261"></a><a name="p153231493261"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p13323159162614"><a name="p13323159162614"></a><a name="p13323159162614"></a>Conv3D计算中原始Dout大小。</p>
</td>
</tr>
<tr id="row1132319992616"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p33239932611"><a name="p33239932611"></a><a name="p33239932611"></a>orgCo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p103231397265"><a name="p103231397265"></a><a name="p103231397265"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1532399152613"><a name="p1532399152613"></a><a name="p1532399152613"></a>Conv3D计算中原始Cout大小。</p>
</td>
</tr>
<tr id="row13231994267"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p17323997261"><a name="p17323997261"></a><a name="p17323997261"></a>orgHo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p11323179122613"><a name="p11323179122613"></a><a name="p11323179122613"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p232313911264"><a name="p232313911264"></a><a name="p232313911264"></a>Conv3D计算中原始Hout大小。</p>
</td>
</tr>
<tr id="row133233916267"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p532314912268"><a name="p532314912268"></a><a name="p532314912268"></a>orgWo</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p193231593268"><a name="p193231593268"></a><a name="p193231593268"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p332349192619"><a name="p332349192619"></a><a name="p332349192619"></a>Conv3D计算中原始Wout大小。</p>
</td>
</tr>
<tr id="row33237922613"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p132311916263"><a name="p132311916263"></a><a name="p132311916263"></a>orgCi</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p93237932620"><a name="p93237932620"></a><a name="p93237932620"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p23237913262"><a name="p23237913262"></a><a name="p23237913262"></a>Conv3D计算中原始Cin大小。</p>
</td>
</tr>
<tr id="row03239912264"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1132329122618"><a name="p1132329122618"></a><a name="p1132329122618"></a>orgDi</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p103232911260"><a name="p103232911260"></a><a name="p103232911260"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p132319982619"><a name="p132319982619"></a><a name="p132319982619"></a>Conv3D计算中原始Din大小。</p>
</td>
</tr>
<tr id="row1432317932615"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p16323139122615"><a name="p16323139122615"></a><a name="p16323139122615"></a>orgHi</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p143237914264"><a name="p143237914264"></a><a name="p143237914264"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p932389172614"><a name="p932389172614"></a><a name="p932389172614"></a>Conv3D计算中原始Hin大小。</p>
</td>
</tr>
<tr id="row123238992612"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p10323891262"><a name="p10323891262"></a><a name="p10323891262"></a>orgWi</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p153239911262"><a name="p153239911262"></a><a name="p153239911262"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p632314920262"><a name="p632314920262"></a><a name="p632314920262"></a>Conv3D计算中原始Win大小。</p>
</td>
</tr>
<tr id="row832369112618"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p73231911267"><a name="p73231911267"></a><a name="p73231911267"></a>kernelD</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p23231196263"><a name="p23231196263"></a><a name="p23231196263"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p232312952616"><a name="p232312952616"></a><a name="p232312952616"></a>Conv3D计算中卷积核原始kernel D维度大小。</p>
</td>
</tr>
<tr id="row332399102611"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p103234916264"><a name="p103234916264"></a><a name="p103234916264"></a>kernelH</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p73231914261"><a name="p73231914261"></a><a name="p73231914261"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p11323179112617"><a name="p11323179112617"></a><a name="p11323179112617"></a>Conv3D计算中卷积核原始kernel H维度大小。</p>
</td>
</tr>
<tr id="row1932349142616"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1032315932612"><a name="p1032315932612"></a><a name="p1032315932612"></a>kernelW</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p73231910266"><a name="p73231910266"></a><a name="p73231910266"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p432316911262"><a name="p432316911262"></a><a name="p432316911262"></a>Conv3D计算中卷积核原始kernel W维度大小。</p>
</td>
</tr>
<tr id="row103234918265"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p83234911260"><a name="p83234911260"></a><a name="p83234911260"></a>strideD</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p932317920266"><a name="p932317920266"></a><a name="p932317920266"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p43237920261"><a name="p43237920261"></a><a name="p43237920261"></a>Conv3D计算中Stride D维度大小。</p>
</td>
</tr>
<tr id="row1232439132617"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1832479162619"><a name="p1832479162619"></a><a name="p1832479162619"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1832418912261"><a name="p1832418912261"></a><a name="p1832418912261"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1324179132611"><a name="p1324179132611"></a><a name="p1324179132611"></a>Conv3D计算中Stride H维度大小。</p>
</td>
</tr>
<tr id="row17324293269"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p83243922614"><a name="p83243922614"></a><a name="p83243922614"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p123247916263"><a name="p123247916263"></a><a name="p123247916263"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p16324159142610"><a name="p16324159142610"></a><a name="p16324159142610"></a>Conv3D计算中Stride W维度大小。</p>
</td>
</tr>
<tr id="row19324179102620"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p432449112612"><a name="p432449112612"></a><a name="p432449112612"></a>dilationD</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p183241391264"><a name="p183241391264"></a><a name="p183241391264"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p133246982611"><a name="p133246982611"></a><a name="p133246982611"></a>Conv3D计算中Dilation D维度大小。</p>
</td>
</tr>
<tr id="row153241193262"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p153241298268"><a name="p153241298268"></a><a name="p153241298268"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p163241797269"><a name="p163241797269"></a><a name="p163241797269"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p732415912267"><a name="p732415912267"></a><a name="p732415912267"></a>Conv3D计算中Dilation H维度大小。</p>
</td>
</tr>
<tr id="row163245962613"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p13242912614"><a name="p13242912614"></a><a name="p13242912614"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p143249915269"><a name="p143249915269"></a><a name="p143249915269"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p123241990262"><a name="p123241990262"></a><a name="p123241990262"></a>Conv3D计算中Dilation W维度大小。</p>
</td>
</tr>
<tr id="row532414911264"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1032419914266"><a name="p1032419914266"></a><a name="p1032419914266"></a>padHead</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p432489142617"><a name="p432489142617"></a><a name="p432489142617"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p432416917261"><a name="p432416917261"></a><a name="p432416917261"></a>Conv3D计算中Padding D维度Head方向大小。</p>
</td>
</tr>
<tr id="row33241918268"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1932449172610"><a name="p1932449172610"></a><a name="p1932449172610"></a>padTail</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p18324119192611"><a name="p18324119192611"></a><a name="p18324119192611"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p53243942612"><a name="p53243942612"></a><a name="p53243942612"></a>Conv3D计算中Padding D维度Tail方向大小。</p>
</td>
</tr>
<tr id="row53249942614"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1732409132613"><a name="p1732409132613"></a><a name="p1732409132613"></a>padUp</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p0324594262"><a name="p0324594262"></a><a name="p0324594262"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1332417952610"><a name="p1332417952610"></a><a name="p1332417952610"></a>Conv3D计算中Padding H维度Up方向大小。</p>
</td>
</tr>
<tr id="row732420972611"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p153249916261"><a name="p153249916261"></a><a name="p153249916261"></a>padDown</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p12324294269"><a name="p12324294269"></a><a name="p12324294269"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1232409152613"><a name="p1232409152613"></a><a name="p1232409152613"></a>Conv3D计算中Padding H维度Down方向大小。</p>
</td>
</tr>
<tr id="row143241798260"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p132439162611"><a name="p132439162611"></a><a name="p132439162611"></a>padLeft</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1832410917260"><a name="p1832410917260"></a><a name="p1832410917260"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1132417982611"><a name="p1132417982611"></a><a name="p1132417982611"></a>Conv3D计算中Padding W维度Left方向大小。</p>
</td>
</tr>
<tr id="row16324898260"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p432412910265"><a name="p432412910265"></a><a name="p432412910265"></a>padRight</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p133241992617"><a name="p133241992617"></a><a name="p133241992617"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p43241991261"><a name="p43241991261"></a><a name="p43241991261"></a>Conv3D计算中Padding W维度Right方向大小。</p>
</td>
</tr>
<tr id="row11324898261"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p8324169172613"><a name="p8324169172613"></a><a name="p8324169172613"></a>mL0</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p5324149172614"><a name="p5324149172614"></a><a name="p5324149172614"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p143243910269"><a name="p143243910269"></a><a name="p143243910269"></a>L0上单次处理的M大小。</p>
</td>
</tr>
<tr id="row532419932616"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p103241098266"><a name="p103241098266"></a><a name="p103241098266"></a>kL0</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1332420915266"><a name="p1332420915266"></a><a name="p1332420915266"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p153244942616"><a name="p153244942616"></a><a name="p153244942616"></a>L0上单次处理的K大小。</p>
</td>
</tr>
<tr id="row1232414918268"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p332479172612"><a name="p332479172612"></a><a name="p332479172612"></a>nL0</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p63241992260"><a name="p63241992260"></a><a name="p63241992260"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p133241922616"><a name="p133241922616"></a><a name="p133241922616"></a>L0上单次处理的N大小。</p>
</td>
</tr>
<tr id="row183241798263"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p183245972616"><a name="p183245972616"></a><a name="p183245972616"></a>kAL1</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p173242922611"><a name="p173242922611"></a><a name="p173242922611"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p203241598260"><a name="p203241598260"></a><a name="p203241598260"></a>L1上Input K的实际大小，等于Cin1InL1 * KH *  KW * C0，Cin1InL1是KD * Cin1合轴之后Tiling切分的大小。</p>
</td>
</tr>
<tr id="row103252902617"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p113251972610"><a name="p113251972610"></a><a name="p113251972610"></a>kBL1</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1832516902618"><a name="p1832516902618"></a><a name="p1832516902618"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p332539182617"><a name="p332539182617"></a><a name="p332539182617"></a>L1上Weight K的实际大小，等于Cin1InL1 * KH *  KW * C0，Cin1InL1是KD * Cin1合轴之后Tiling切分的大小。</p>
</td>
</tr>
<tr id="row8325494265"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p13258992619"><a name="p13258992619"></a><a name="p13258992619"></a>nBL1</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p12325195269"><a name="p12325195269"></a><a name="p12325195269"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1132512918262"><a name="p1132512918262"></a><a name="p1132512918262"></a>L1上Weight载入Cout维度的实际数据大小。</p>
</td>
</tr>
<tr id="row43253922616"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p532599162616"><a name="p532599162616"></a><a name="p532599162616"></a>mAL1</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p63251797265"><a name="p63251797265"></a><a name="p63251797265"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p1832519919267"><a name="p1832519919267"></a><a name="p1832519919267"></a>L1上Input载入M的实际数据大小。</p>
</td>
</tr>
<tr id="row732514972615"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p1432519112610"><a name="p1432519112610"></a><a name="p1432519112610"></a>al1FullLoad</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1832519132610"><a name="p1832519132610"></a><a name="p1832519132610"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p23259912266"><a name="p23259912266"></a><a name="p23259912266"></a>Input数据在L1 Buffer是否全载。</p>
<p id="p76122059181313"><a name="p76122059181313"></a><a name="p76122059181313"></a>0：Input数据在L1 Buffer上不全载。</p>
<p id="p96121459111319"><a name="p96121459111319"></a><a name="p96121459111319"></a>1：Input数据在L1 Buffer上全载。</p>
</td>
</tr>
<tr id="row123257992611"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p4325109182611"><a name="p4325109182611"></a><a name="p4325109182611"></a>bl1FullLoad</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p163254919261"><a name="p163254919261"></a><a name="p163254919261"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p18325294264"><a name="p18325294264"></a><a name="p18325294264"></a>Weight数据在L1 Buffer是否全载。</p>
<p id="p92941935201412"><a name="p92941935201412"></a><a name="p92941935201412"></a>0：Weight数据在L1 Buffer上不全载。</p>
<p id="p14294835191410"><a name="p14294835191410"></a><a name="p14294835191410"></a>1：Weight数据在L1 Buffer上全载。</p>
</td>
</tr>
<tr id="row1132519917265"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p63251962614"><a name="p63251962614"></a><a name="p63251962614"></a>iterateMNOrder</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p83251098267"><a name="p83251098267"></a><a name="p83251098267"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p632539102616"><a name="p632539102616"></a><a name="p632539102616"></a>输出结果矩阵Output时，M轴和N轴的输出顺序。</p>
<p id="p732539182610"><a name="p732539182610"></a><a name="p732539182610"></a>0：优先输出M方向。先输出M方向，再输出N方向，<a href="#fig2054162614714">图2</a>。</p>
<p id="p6325796264"><a name="p6325796264"></a><a name="p6325796264"></a>1：优先输出N方向。先输出N方向，再输出M方向，<a href="#fig1788889124818">图3</a>。</p>
<p id="p8325169172619"><a name="p8325169172619"></a><a name="p8325169172619"></a>M由Hout和Wout组成，M方向的输出顺序为，先输出Wout方向，再输出Hout方向。</p>
</td>
</tr>
<tr id="row183257914264"><td class="cellrowborder" valign="top" width="32.65%" headers="mcps1.2.4.1.1 "><p id="p10325119182610"><a name="p10325119182610"></a><a name="p10325119182610"></a>biasFullLoadFlag</p>
</td>
<td class="cellrowborder" valign="top" width="14.29%" headers="mcps1.2.4.1.2 "><p id="p1832539142612"><a name="p1832539142612"></a><a name="p1832539142612"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="53.059999999999995%" headers="mcps1.2.4.1.3 "><p id="p7325209182614"><a name="p7325209182614"></a><a name="p7325209182614"></a>Bias是否全载进L1 Buffer。</p>
<p id="p10325179182617"><a name="p10325179182617"></a><a name="p10325179182617"></a>0：否，单核内单次载入Bias大小等于单次矩阵乘N方向的大小nL0。</p>
<p id="p8325295264"><a name="p8325295264"></a><a name="p8325295264"></a>1：是，单核内的Bias一次全载。</p>
</td>
</tr>
<tr id="row6190142593516"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.4.1.1 mcps1.2.4.1.2 mcps1.2.4.1.3 "><p id="p1499632883513"><a name="p1499632883513"></a><a name="p1499632883513"></a>注：上述的M轴为卷积正向操作过程中的输入Input在img2col展开后的纵轴，数值上等于Hout * Wout；K为输入Input在img2col展开后的横轴，数值上等于KD*C1*KH*KW*C0；KD/KH/KW为Weight的Depth、Height、Width，即kernelD/kernelH/kernelW的简写；N为Weight的Cout，具体请见<a href="#fig1053275794620">图1</a>。</p>
</td>
</tr>
</tbody>
</table>

**图 1**  卷积3D正向MKN示意图<a name="fig1053275794620"></a>  
<!-- img2text -->
```
           K                       N                                 N
    ╭─────────────╮        ╭──────────────────╮               ╭──────────────────────╮
    │             │        │                  │               │                      │
╭───╯ ┌─────────┐ │        │  ┌────────────┐  │               │  ┌────────────────┐  │ ╰───╮
│ M   │         │ │    *   │  │            │  │   K   =       │  │                │  │   M │
╰───╮ │         │ │        ╰──┤            ├──╯               ╰──┤                ├──╯ ╭───╯
    │ │         │ │           │            │                      │                │     │
    │ │         │ │           └────────────┘                      │                │     │
    │ └─────────┘ │                                               │                │     │
    ╰─────────────╯                                               │                │     ╰───
                                                                  └────────────────┘

        Input                         Weight                               Output
```

**图 2**  卷积3D正向MFirst示意图<a name="fig2054162614714"></a>  
<!-- img2text -->
```
          <──── K ────>                           <────────── N ──────────>                            <────────── N ──────────>
        ╭──────────────╮                        ╭────────────────────────╮                          ╭────────────────────────╮
        │              │                        │                        │                          │                        │
        │  ↓           │                        │  ─────────────────→    │                          │  ↗      ↘    ↗      ↘ │
        │  │           │                        │                        │                          │ ↗        ↘  ↗        ↘│
<─ M ─> │  │           │                    <─ K ─>                      │                      <─ M ─>│↓          ↑↓          ↓│
        │  │           │                        │                        │                          │                        │
        │  │           │                        │                        │                          │                        │
        │              │                        │                        │                          │                        │
        ╰──────────────╯                        ╰────────────────────────╯                          ╰────────────────────────╯
           Input                                   Weight                                       Output M方向先输出

                        *                                         =
```

**图 3**  卷积3D正向NFirst示意图<a name="fig1788889124818"></a>  
<!-- img2text -->
```
            K                           N                                         N
      ⎧───────────⎫              ⎧─────────────────⎫                    ⎧─────────────────────⎫
      │           │              │                 │                    │                     │
   ┌─────────┐    │           ┌─────────────────┐  │                    │ ┌─────────────────┐ │
   │         │    │           │                 │  │                    │ │ ──────────────→ │ │
M  │    │    │    │        *  │ ─────────────→  │  K                 =  │ │  ←───────────── │ │ M
   │    │    │    │           │                 │  │                    │ │ ──────────────→ │ │
   │    ↓    │    │           │                 │  │                    │ │  ←───────────── │ │
   │         │    │           └─────────────────┘  │                    │ │ ──────────────→ │ │
   └─────────┘    │              ⎩─────────────────⎭                    │ │  ←───────────── │ │
      ⎩───────────⎭                                                     │ │ ──────────────→ │ │
                                                                         │ └─────────────────┘ │
                                                                         ⎩─────────────────────⎭

      Input                          Weight                                  Output N方向先输出
```

