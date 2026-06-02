# Cast<a name="ZH-CN_TOPIC_0000002554344913"></a>

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

根据源操作数和目的操作数Tensor的数据类型进行精度转换。

在了解精度转换规则之前，需要先了解浮点数的表示方式和二进制的舍入规则：

-   浮点数的表示方式
    -   half共16bit，包括1bit符号位（S），5bit指数位（E）和10bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 15</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-14</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        <!-- img2text -->
```
┌───┬─────────┬──────────────┐
│ S │    E    │      M       │
├───┼─────────┼──────────────┤
│ 0 │  01111  │ 1100000000   │
└───┴─────────┴──────────────┘
```

        上图中S=0，E=15，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，表示的结果为1.75。

    -   float共32bit，包括1bit符号位（S），8bit指数位（E）和23bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        <!-- img2text -->
```
┌───┬──────────┬────────────────────────┐
│ S │    E     │           M            │
├───┼──────────┼────────────────────────┤
│ 0 │ 01111111 │ 11000000000000000000000 │
└───┴──────────┴────────────────────────┘
```

        上图中S = 0，E = 127，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，最终表示的结果为1.75 。

    -   bfloat16\_t共16bit，包括1bit符号位（S），8bit指数位（E）和7bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        <!-- img2text -->
```
┌───┬──────────┬──────────┐
│ S │    E     │    M     │
├───┼──────────┼──────────┤
│ 0 │ 01111111 │ 1100000  │
└───┴──────────┴──────────┘
```

        上图中S = 0，E = 127，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，最终表示的结果为1.75。

-   二进制的舍入规则和十进制类似，具体如下：

    <!-- img2text -->
```text
┌───┬────────────┬──────────────────────────────┬────────────┐
│ S │     E      │              M               │  待舍入部分 │
├───┼────────────┼──────────────────────────────┼────────────┤
│ x │ xxxxxxxx   │ xxxxxxxxxxxxxxxxxxxxxxxxxx   │ xxxxxxx    │
└───┴────────────┴──────────────────────────────┴────────────┘
```

    -   CAST\_RINT模式下，若待舍入部分的第一位为0，则不进位；若第一位为1且后续位不全为0，则进位；若第一位为1且后续位全为0，当M的最后一位为0则不进位，当M的最后一位为1则进位。

    -   CAST\_FLOOR模式下，若S为0，则不进位；若S为1，当待舍入部分全为0则不进位，否则，进位。
    -   CAST\_CEIL模式下，若S为1，则不进位；若S为0，当待舍入部分全为0则不进位；否则，进位。
    -   CAST\_ROUND模式下，若待舍入部分的第一位为0，则不进位；否则，进位。
    -   CAST\_TRUNC模式下，总是不进位。
    -   CAST\_ODD模式下，若待舍入部分全为0，则不进位；若待舍入部分不全为0，当M的最后一位为1则不进位，当M的最后一位为0则进位。
    -   CAST\_HYBRID：随机舍入，目前特指输出结果是hif8数据类型时，会用到的一种随机舍入。

精度转换规则如下表所示（为方便描述下文描述中的src代表源操作数，dst代表目的操作数）：

**表 1**  精度转换规则

<a name="table235404962912"></a>
<table><thead align="left"><tr id="row935554942920"><th class="cellrowborder" valign="top" width="10.431043104310431%" id="mcps1.2.4.1.1"><p id="p13355144922911"><a name="p13355144922911"></a><a name="p13355144922911"></a>src类型</p>
</th>
<th class="cellrowborder" valign="top" width="10.46104610461046%" id="mcps1.2.4.1.2"><p id="p135514913299"><a name="p135514913299"></a><a name="p135514913299"></a>dst类型</p>
</th>
<th class="cellrowborder" valign="top" width="79.1079107910791%" id="mcps1.2.4.1.3"><p id="p7113121774314"><a name="p7113121774314"></a><a name="p7113121774314"></a>精度转换规则介绍</p>
</th>
</tr>
</thead>
<tbody><tr id="row3355849152915"><td class="cellrowborder" rowspan="6" align="left" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p9687163213521"><a name="p9687163213521"></a><a name="p9687163213521"></a>float</p>
</td>
<td class="cellrowborder" align="left" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p113551749202919"><a name="p113551749202919"></a><a name="p113551749202919"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p6999183016349"><a name="p6999183016349"></a><a name="p6999183016349"></a>将src按照roundMode（精度转换处理模式，参见<a href="#section622mcpsimp">参数说明</a>中的roundMode参数）取整，仍以float格式存入dst中。</p>
<p id="p2023513251345"><a name="p2023513251345"></a><a name="p2023513251345"></a>示例：输入0.5，</p>
<p id="p1543914528509"><a name="p1543914528509"></a><a name="p1543914528509"></a>CAST_RINT模式输出0.0，CAST_FLOOR模式输出0.0，CAST_CEIL模式输出1.0，CAST_ROUND模式输出1.0，CAST_TRUNC模式输出0.0。</p>
</td>
</tr>
<tr id="row14355204919294"><td class="cellrowborder" align="left" valign="top" headers="mcps1.2.4.1.1 "><p id="p1935534915297"><a name="p1935534915297"></a><a name="p1935534915297"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p43071952153512"><a name="p43071952153512"></a><a name="p43071952153512"></a>将src按照roundMode取到half所能表示的数，以half格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p162319468358"><a name="p162319468358"></a><a name="p162319468358"></a>示例：输入0.5 + 2<sup id="sup69711611523"><a name="sup69711611523"></a><a name="sup69711611523"></a>-12</sup>，写成float的表示形式：2<sup id="sup61291556145210"><a name="sup61291556145210"></a><a name="sup61291556145210"></a>-1</sup> * (1 + 2<sup id="sup732521620210"><a name="sup732521620210"></a><a name="sup732521620210"></a>-11</sup>)，因此E = -1 + 127 = 126，M = 2<sup id="sup1315617330213"><a name="sup1315617330213"></a><a name="sup1315617330213"></a>-11。</sup></p>
<p id="p673417168511"><a name="p673417168511"></a><a name="p673417168511"></a><a name="image1952310171059"></a><a name="image1952310171059"></a><span><img class="eddx" id="image1952310171059" src="figures/流水任务运行示意图-103.png"></span></p>
<p id="p1862334619354"><a name="p1862334619354"></a><a name="p1862334619354"></a>half的指数位可以表示出2<sup id="sup156939018108"><a name="sup156939018108"></a><a name="sup156939018108"></a>-1</sup>，E = -1 + 15 = 14，但half只有10 bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p1562314462353"><a name="p1562314462353"></a><a name="p1562314462353"></a>CAST_RINT模式舍入得尾数0000000000，E = 14，M = 0，最终表示的结果为0.5；</p>
<p id="p962374623520"><a name="p962374623520"></a><a name="p962374623520"></a>CAST_FLOOR模式舍入得尾数0000000000，E = 14，M = 0，最终表示的结果为0.5；</p>
<p id="p15623646153518"><a name="p15623646153518"></a><a name="p15623646153518"></a>CAST_CEIL模式舍入得尾数0000000001，E = 14，M = 2<sup id="sup897714261414"><a name="sup897714261414"></a><a name="sup897714261414"></a>-10</sup>，最终表示的结果为0.5 + 2<sup id="sup2460401919"><a name="sup2460401919"></a><a name="sup2460401919"></a>-11</sup>；</p>
<p id="p3623746173516"><a name="p3623746173516"></a><a name="p3623746173516"></a>CAST_ROUND模式舍入得尾数0000000001，E = 14，M = 2<sup id="sup51233291039"><a name="sup51233291039"></a><a name="sup51233291039"></a>-10</sup>，最终表示的结果为0.5 + 2<sup id="sup518915411539"><a name="sup518915411539"></a><a name="sup518915411539"></a>-11</sup>；</p>
<p id="p5623174673519"><a name="p5623174673519"></a><a name="p5623174673519"></a>CAST_TRUNC模式舍入得尾数0000000000，E = 14，M = 0，最终表示的结果为0.5；</p>
<p id="p18623184663515"><a name="p18623184663515"></a><a name="p18623184663515"></a>CAST_ODD模式舍入得尾数0000000001，E = 14，M = 2<sup id="sup135246541"><a name="sup135246541"></a><a name="sup135246541"></a>-10</sup>，最终表示的结果为0.5 + 2<sup id="sup389919164419"><a name="sup389919164419"></a><a name="sup389919164419"></a>-11</sup> 。</p>
</td>
</tr>
<tr id="row7394104235320"><td class="cellrowborder" align="left" valign="top" headers="mcps1.2.4.1.1 "><p id="p103941542205314"><a name="p103941542205314"></a><a name="p103941542205314"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5992104645711"><a name="p5992104645711"></a><a name="p5992104645711"></a>将src按照roundMode取整，以int64_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p197626427574"><a name="p197626427574"></a><a name="p197626427574"></a>示例：输入2<sup id="sup9635352887"><a name="sup9635352887"></a><a name="sup9635352887"></a>22</sup> + 0.5，</p>
<p id="p9762134215714"><a name="p9762134215714"></a><a name="p9762134215714"></a>CAST_RINT模式输出2<sup id="sup2129612141616"><a name="sup2129612141616"></a><a name="sup2129612141616"></a>22</sup>，CAST_FLOOR模式输出2<sup id="sup1077211374169"><a name="sup1077211374169"></a><a name="sup1077211374169"></a>22</sup>，CAST_CEIL模式输出2<sup id="sup1797365810166"><a name="sup1797365810166"></a><a name="sup1797365810166"></a>22</sup> + 1，CAST_ROUND模式输出2<sup id="sup67489377179"><a name="sup67489377179"></a><a name="sup67489377179"></a>22</sup> + 1，CAST_TRUNC模式输出2<sup id="sup17685121615182"><a name="sup17685121615182"></a><a name="sup17685121615182"></a>22</sup>。</p>
</td>
</tr>
<tr id="row127281637573"><td class="cellrowborder" align="left" valign="top" headers="mcps1.2.4.1.1 "><p id="p1972843105717"><a name="p1972843105717"></a><a name="p1972843105717"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1644991984013"><a name="p1644991984013"></a><a name="p1644991984013"></a>将src按照roundMode取整，以int32_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p336661414407"><a name="p336661414407"></a><a name="p336661414407"></a>示例：输入2<sup id="sup107091910191917"><a name="sup107091910191917"></a><a name="sup107091910191917"></a>22 </sup> + 0.5，</p>
<p id="p0366914184010"><a name="p0366914184010"></a><a name="p0366914184010"></a>CAST_RINT模式输出2<sup id="sup646984014195"><a name="sup646984014195"></a><a name="sup646984014195"></a>22</sup>，CAST_FLOOR模式输出2<sup id="sup969555315201"><a name="sup969555315201"></a><a name="sup969555315201"></a>22</sup> ，CAST_CEIL模式输出2<sup id="sup942324412117"><a name="sup942324412117"></a><a name="sup942324412117"></a>22</sup> + 1，CAST_ROUND模式输出2<sup id="sup189201223132217"><a name="sup189201223132217"></a><a name="sup189201223132217"></a>22</sup> + 1，CAST_TRUNC模式输出2<sup id="sup104451415132319"><a name="sup104451415132319"></a><a name="sup104451415132319"></a>22</sup>。</p>
</td>
</tr>
<tr id="row178573216486"><td class="cellrowborder" align="left" valign="top" headers="mcps1.2.4.1.1 "><p id="p1778516322487"><a name="p1778516322487"></a><a name="p1778516322487"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17489160174910"><a name="p17489160174910"></a><a name="p17489160174910"></a>将src按照roundMode取整，以int16_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p17904115784820"><a name="p17904115784820"></a><a name="p17904115784820"></a>示例：输入2<sup id="sup1777114542316"><a name="sup1777114542316"></a><a name="sup1777114542316"></a>22</sup> + 0.5，</p>
<p id="p149041357194813"><a name="p149041357194813"></a><a name="p149041357194813"></a>CAST_RINT模式输出2<sup id="sup997595662418"><a name="sup997595662418"></a><a name="sup997595662418"></a>15</sup> - 1（溢出处理），CAST_FLOOR模式输出2<sup id="sup11919141717255"><a name="sup11919141717255"></a><a name="sup11919141717255"></a>15</sup> - 1（溢出处理），CAST_CEIL模式输出2<sup id="sup121521540132519"><a name="sup121521540132519"></a><a name="sup121521540132519"></a>15</sup> - 1（溢出处理），CAST_ROUND模式输出2<sup id="sup2082373312610"><a name="sup2082373312610"></a><a name="sup2082373312610"></a>15</sup> - 1（溢出处理），CAST_TRUNC模式输出2<sup id="sup15439589268"><a name="sup15439589268"></a><a name="sup15439589268"></a>15</sup> - 1（溢出处理）。</p>
</td>
</tr>
<tr id="row494010912450"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1694010974510"><a name="p1694010974510"></a><a name="p1694010974510"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p32441413463"><a name="p32441413463"></a><a name="p32441413463"></a>将src按照roundMode取到bfloat16_t所能表示的数，以bfloat16_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p471962154618"><a name="p471962154618"></a><a name="p471962154618"></a>示例：输入0.5+ 2<sup id="sup374112013460"><a name="sup374112013460"></a><a name="sup374112013460"></a>-9</sup> + 2<sup id="sup3741190204611"><a name="sup3741190204611"></a><a name="sup3741190204611"></a>-11</sup> ，写成float的表示形式：2<sup id="sup17741110104616"><a name="sup17741110104616"></a><a name="sup17741110104616"></a>-1</sup> * (1 + 2<sup id="sup2074170114612"><a name="sup2074170114612"></a><a name="sup2074170114612"></a>-8</sup> + 2<sup id="sup27413094618"><a name="sup27413094618"></a><a name="sup27413094618"></a>-10</sup>)，因此E = -1 + 127 = 126，M = 2<sup id="sup87411054619"><a name="sup87411054619"></a><a name="sup87411054619"></a>-8</sup> + 2<sup id="sup107417094616"><a name="sup107417094616"></a><a name="sup107417094616"></a>-10</sup> 。</p>
<p id="p143031654144710"><a name="p143031654144710"></a><a name="p143031654144710"></a><a name="image4303185416478"></a><a name="image4303185416478"></a><span><img class="eddx" id="image4303185416478" src="figures/流水任务运行示意图-104.png"></span></p>
<p id="p1666754012468"><a name="p1666754012468"></a><a name="p1666754012468"></a>bfloat16_t的指数位位数和float的相同，有E = 126，但bfloat16_t只有7bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p1388819433467"><a name="p1388819433467"></a><a name="p1388819433467"></a>CAST_RINT模式舍入得尾数0000001，E = 126，M = 2<sup id="sup274212064611"><a name="sup274212064611"></a><a name="sup274212064611"></a>-7</sup>，最终表示的结果为0.5 + 2<sup id="sup20742100204612"><a name="sup20742100204612"></a><a name="sup20742100204612"></a>-8</sup>；</p>
<p id="p523514453460"><a name="p523514453460"></a><a name="p523514453460"></a>CAST_FLOOR模式舍入得尾数0000000，E = 126，M = 0，最终表示的结果为0.5；</p>
<p id="p17614194619461"><a name="p17614194619461"></a><a name="p17614194619461"></a>CAST_CEIL模式舍入得尾数0000001，E = 126，M = 2<sup id="sup374217010465"><a name="sup374217010465"></a><a name="sup374217010465"></a>-7</sup>，最终表示的结果为0.5 + 2<sup id="sup3742110144610"><a name="sup3742110144610"></a><a name="sup3742110144610"></a>-8</sup>；</p>
<p id="p66851549174612"><a name="p66851549174612"></a><a name="p66851549174612"></a>CAST_ROUND模式舍入得尾数0000001，E = 126，M = 2<sup id="sup674215014468"><a name="sup674215014468"></a><a name="sup674215014468"></a>-7</sup>，最终表示的结果为0.5 + 2<sup id="sup117421406462"><a name="sup117421406462"></a><a name="sup117421406462"></a>-8</sup>；</p>
<p id="p29409920456"><a name="p29409920456"></a><a name="p29409920456"></a>CAST_TRUNC模式舍入得尾数0000000，E = 126，M = 0，最终表示的结果为0.5。</p>
</td>
</tr>
<tr id="row1471183015520"><td class="cellrowborder" rowspan="3" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p16371141915311"><a name="p16371141915311"></a><a name="p16371141915311"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p5978836185815"><a name="p5978836185815"></a><a name="p5978836185815"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p1583714118115"><a name="p1583714118115"></a><a name="p1583714118115"></a>将src按照roundMode取整，以hifloat8_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p9837194171118"><a name="p9837194171118"></a><a name="p9837194171118"></a>示例：输入1.75，CAST_ROUND模式输出2；CAST_HYBRID参考<a href="Cast-65.md#table1352142520363">表9</a>输出。</p>
</td>
</tr>
<tr id="row125807343529"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p211613474584"><a name="p211613474584"></a><a name="p211613474584"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p113488519114"><a name="p113488519114"></a><a name="p113488519114"></a>将src按照roundMode取整，以fp8_e4m3fn_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p1934835112114"><a name="p1934835112114"></a><a name="p1934835112114"></a>示例：输入2.5，CAST_RINT模式输出2。</p>
</td>
</tr>
<tr id="row765153811529"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5456114945819"><a name="p5456114945819"></a><a name="p5456114945819"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p78483511140"><a name="p78483511140"></a><a name="p78483511140"></a>将src按照roundMode取整，以fp8_e5m2_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p148493513143"><a name="p148493513143"></a><a name="p148493513143"></a>示例：输入2.5，CAST_RINT模式输出2。</p>
</td>
</tr>
<tr id="row126837164286"><td class="cellrowborder" rowspan="6" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p5117735302"><a name="p5117735302"></a><a name="p5117735302"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p56831016192812"><a name="p56831016192812"></a><a name="p56831016192812"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p7616217173110"><a name="p7616217173110"></a><a name="p7616217173110"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p17414713163117"><a name="p17414713163117"></a><a name="p17414713163117"></a>示例：输入1.5 - 2<sup id="sup8513173315319"><a name="sup8513173315319"></a><a name="sup8513173315319"></a>-10</sup>，输出1.5 - 2<sup id="sup1898194612314"><a name="sup1898194612314"></a><a name="sup1898194612314"></a>-10</sup>。</p>
</td>
</tr>
<tr id="row13100102112812"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p13101122102818"><a name="p13101122102818"></a><a name="p13101122102818"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6430013183210"><a name="p6430013183210"></a><a name="p6430013183210"></a>将src按照roundMode取整，以int32_t格式存入dst中。</p>
<p id="p26845910325"><a name="p26845910325"></a><a name="p26845910325"></a>示例：输入-1.5，</p>
<p id="p1968413919321"><a name="p1968413919321"></a><a name="p1968413919321"></a>CAST_RINT模式输出-2，CAST_FLOOR模式输出-2，CAST_CEIL模式输出-1，CAST_ROUND模式输出-2，CAST_TRUNC模式输出-1。</p>
</td>
</tr>
<tr id="row147081324102816"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1570912415289"><a name="p1570912415289"></a><a name="p1570912415289"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p318838153418"><a name="p318838153418"></a><a name="p318838153418"></a>将src按照roundMode取整，以int16_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p119315133414"><a name="p119315133414"></a><a name="p119315133414"></a>示例：输入2<sup id="sup1872532420340"><a name="sup1872532420340"></a><a name="sup1872532420340"></a>7</sup> - 0.5，</p>
<p id="p1393115173412"><a name="p1393115173412"></a><a name="p1393115173412"></a>CAST_RINT模式输出2<sup id="sup1719684453410"><a name="sup1719684453410"></a><a name="sup1719684453410"></a>7</sup>，CAST_FLOOR模式输出2<sup id="sup6293101012368"><a name="sup6293101012368"></a><a name="sup6293101012368"></a>7</sup> - 1，CAST_CEIL模式输出2<sup id="sup10950112913617"><a name="sup10950112913617"></a><a name="sup10950112913617"></a>7</sup>，CAST_ROUND模式输出2<sup id="sup067711568368"><a name="sup067711568368"></a><a name="sup067711568368"></a>7</sup>，CAST_TRUNC模式输出2<sup id="sup115261319113716"><a name="sup115261319113716"></a><a name="sup115261319113716"></a>7</sup> - 1。</p>
</td>
</tr>
<tr id="row1719894510296"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p819834582915"><a name="p819834582915"></a><a name="p819834582915"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p125571958113816"><a name="p125571958113816"></a><a name="p125571958113816"></a>将src按照roundMode取整，以int8_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p495635313811"><a name="p495635313811"></a><a name="p495635313811"></a>示例：输入2<sup id="sup795091043917"><a name="sup795091043917"></a><a name="sup795091043917"></a>7</sup> - 0.5，</p>
<p id="p16956553113810"><a name="p16956553113810"></a><a name="p16956553113810"></a>CAST_RINT模式输出2<sup id="sup1715172815397"><a name="sup1715172815397"></a><a name="sup1715172815397"></a>7</sup> - 1（溢出处理），CAST_FLOOR模式输出2<sup id="sup1620734511399"><a name="sup1620734511399"></a><a name="sup1620734511399"></a>7</sup> - 1，CAST_CEIL模式输出2<sup id="sup156381614403"><a name="sup156381614403"></a><a name="sup156381614403"></a>7</sup> - 1（溢出处理），CAST_ROUND模式输出2<sup id="sup167271828164019"><a name="sup167271828164019"></a><a name="sup167271828164019"></a>7</sup> - 1（溢出处理），CAST_TRUNC模式输出2<sup id="sup1841514480404"><a name="sup1841514480404"></a><a name="sup1841514480404"></a>7</sup> - 1。</p>
</td>
</tr>
<tr id="row393435592914"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9934855112913"><a name="p9934855112913"></a><a name="p9934855112913"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p92056654211"><a name="p92056654211"></a><a name="p92056654211"></a>将src按照roundMode取整，以uint8_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p183531235417"><a name="p183531235417"></a><a name="p183531235417"></a>负数输入会被视为异常。</p>
<p id="p989516320426"><a name="p989516320426"></a><a name="p989516320426"></a>示例：输入1.75，</p>
<p id="p20895163184210"><a name="p20895163184210"></a><a name="p20895163184210"></a>CAST_RINT模式输出2，CAST_FLOOR模式输出1，CAST_CEIL模式输出2，CAST_ROUND模式输出2，CAST_TRUNC模式输出1。</p>
</td>
</tr>
<tr id="row1961614418139"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p961694171312"><a name="p961694171312"></a><a name="p961694171312"></a>int4b_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p111803722615"><a name="p111803722615"></a><a name="p111803722615"></a>将src按照roundMode取整，以int4b_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p51853713268"><a name="p51853713268"></a><a name="p51853713268"></a>示例：输入1.5，</p>
<p id="p12185373262"><a name="p12185373262"></a><a name="p12185373262"></a>CAST_RINT模式输出2，CAST_FLOOR模式输出1，CAST_CEIL模式输出2，CAST_ROUND模式输出2，CAST_TRUNC模式输出1。</p>
</td>
</tr>
<tr id="row1866516319519"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1665133115519"><a name="p1665133115519"></a><a name="p1665133115519"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p9641839752"><a name="p9641839752"></a><a name="p9641839752"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p2641339153"><a name="p2641339153"></a><a name="p2641339153"></a>将src按照roundMode取整，以bfloat16_t格式存入dst中。</p>
<p id="p176493917515"><a name="p176493917515"></a><a name="p176493917515"></a>示例：输入1.75，</p>
<p id="p136417398513"><a name="p136417398513"></a><a name="p136417398513"></a>CAST_RINT模式输出2，CAST_FLOOR模式输出1，CAST_CEIL模式输出2，CAST_ROUND模式输出2，CAST_TRUNC模式输出1。</p>
</td>
</tr>
<tr id="row13301058507"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1435541016115"><a name="p1435541016115"></a><a name="p1435541016115"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p43786577253"><a name="p43786577253"></a><a name="p43786577253"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p81878019399"><a name="p81878019399"></a><a name="p81878019399"></a>将src按照roundMode取整，以hifloat8_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p41874016394"><a name="p41874016394"></a><a name="p41874016394"></a>示例：输入1.75，</p>
<p id="p1918714014392"><a name="p1918714014392"></a><a name="p1918714014392"></a>CAST_ROUND模式输出2，CAST_HYBRID模式参考<a href="Cast-65.md#table1352142520363">表9</a>输出。</p>
</td>
</tr>
<tr id="row119737190554"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p286217619548"><a name="p286217619548"></a><a name="p286217619548"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p29741519115513"><a name="p29741519115513"></a><a name="p29741519115513"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p11318125312559"><a name="p11318125312559"></a><a name="p11318125312559"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p6318135313558"><a name="p6318135313558"></a><a name="p6318135313558"></a>示例：输入1.5 - 2<sup id="sup331895312552"><a name="sup331895312552"></a><a name="sup331895312552"></a>-6</sup>，输出1.5 - 2<sup id="sup173181253165513"><a name="sup173181253165513"></a><a name="sup173181253165513"></a>-6</sup>。</p>
</td>
</tr>
<tr id="row12862146125415"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16862366549"><a name="p16862366549"></a><a name="p16862366549"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17862466541"><a name="p17862466541"></a><a name="p17862466541"></a>将src按照roundMode取整，以int32_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p204322058185418"><a name="p204322058185418"></a><a name="p204322058185418"></a>示例：输入2<sup id="sup5432175855410"><a name="sup5432175855410"></a><a name="sup5432175855410"></a>6&nbsp;</sup>+ 0.5</p>
<p id="p15432558115412"><a name="p15432558115412"></a><a name="p15432558115412"></a>CAST_RINT模式输出2<sup id="sup3432558115411"><a name="sup3432558115411"></a><a name="sup3432558115411"></a>6</sup>，CAST_FLOOR模式输出2<sup id="sup443285815414"><a name="sup443285815414"></a><a name="sup443285815414"></a>6</sup> ，CAST_CEIL模式输出2<sup id="sup18432195895416"><a name="sup18432195895416"></a><a name="sup18432195895416"></a>6</sup> + 1，CAST_ROUND模式输出2<sup id="sup7432558205416"><a name="sup7432558205416"></a><a name="sup7432558205416"></a>6 </sup>+ 1，CAST_TRUNC模式输出2<sup id="sup1243285875414"><a name="sup1243285875414"></a><a name="sup1243285875414"></a>6</sup>。</p>
</td>
</tr>
<tr id="row1802124318366"><td class="cellrowborder" rowspan="3" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1280294393610"><a name="p1280294393610"></a><a name="p1280294393610"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p10802543163619"><a name="p10802543163619"></a><a name="p10802543163619"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p193142030163315"><a name="p193142030163315"></a><a name="p193142030163315"></a>将src按照roundMode取整，以half格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p14407525335"><a name="p14407525335"></a><a name="p14407525335"></a>示例：输入2.90573e-06</p>
<p id="p6901174553511"><a name="p6901174553511"></a><a name="p6901174553511"></a>CAST_RINT模式输出2.9e-06，CAST_FLOOR模式输出2.861e-06 ，CAST_CEIL模式输出2.9e-06，CAST_ROUND模式输出2.9e-06，CAST_TRUNC模式输出2.861e-06。</p>
</td>
</tr>
<tr id="row1461715316120"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6885141765918"><a name="p6885141765918"></a><a name="p6885141765918"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1697920591505"><a name="p1697920591505"></a><a name="p1697920591505"></a>将src按照roundMode取整，以fp4x2_e2m1_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p1952765514597"><a name="p1952765514597"></a><a name="p1952765514597"></a>示例：输入2.5</p>
<p id="p0527155518590"><a name="p0527155518590"></a><a name="p0527155518590"></a>CAST_RINT模式输出2，CAST_FLOOR模式输出2 ，CAST_CEIL模式输出3，CAST_ROUND模式输出3，CAST_TRUNC模式输出2。</p>
</td>
</tr>
<tr id="row89043338114"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p52132005915"><a name="p52132005915"></a><a name="p52132005915"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p177127311119"><a name="p177127311119"></a><a name="p177127311119"></a>将src按照roundMode取整，以fp4x2_e1m2_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p571316311812"><a name="p571316311812"></a><a name="p571316311812"></a>示例：输入2.5</p>
<p id="p871393117115"><a name="p871393117115"></a><a name="p871393117115"></a>CAST_RINT模式输出2，CAST_FLOOR模式输出2 ，CAST_CEIL模式输出3，CAST_ROUND模式输出3，CAST_TRUNC模式输出2。</p>
</td>
</tr>
<tr id="row167621616162720"><td class="cellrowborder" rowspan="3" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p4762111672713"><a name="p4762111672713"></a><a name="p4762111672713"></a>int4b_t</p>
<p id="p119061930163211"><a name="p119061930163211"></a><a name="p119061930163211"></a></p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1976219168278"><a name="p1976219168278"></a><a name="p1976219168278"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p5848165552815"><a name="p5848165552815"></a><a name="p5848165552815"></a>将src以half格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p184925515283"><a name="p184925515283"></a><a name="p184925515283"></a>示例：输入1，输出1.0。</p>
</td>
</tr>
<tr id="row106873810331"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p968758163317"><a name="p968758163317"></a><a name="p968758163317"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2687148203314"><a name="p2687148203314"></a><a name="p2687148203314"></a>将src以int16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p123381242123313"><a name="p123381242123313"></a><a name="p123381242123313"></a>示例：输入1，输出1。</p>
</td>
</tr>
<tr id="row2081273573218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4812133510321"><a name="p4812133510321"></a><a name="p4812133510321"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p881263593217"><a name="p881263593217"></a><a name="p881263593217"></a>将src以bfloat16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p161763593117"><a name="p161763593117"></a><a name="p161763593117"></a>示例：输入1，输出1.0。</p>
</td>
</tr>
<tr id="row125091169442"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p8509616174411"><a name="p8509616174411"></a><a name="p8509616174411"></a>uint8_t</p>
<p id="p3782172533314"><a name="p3782172533314"></a><a name="p3782172533314"></a></p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p115091167442"><a name="p115091167442"></a><a name="p115091167442"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p66001459184518"><a name="p66001459184518"></a><a name="p66001459184518"></a>将src以half格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p84001657174512"><a name="p84001657174512"></a><a name="p84001657174512"></a>示例：输入1，输出1.0。</p>
</td>
</tr>
<tr id="row52221137163810"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p43875163817"><a name="p43875163817"></a><a name="p43875163817"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1148610632420"><a name="p1148610632420"></a><a name="p1148610632420"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p576863013241"><a name="p576863013241"></a><a name="p576863013241"></a>将src以uint16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p6768930132415"><a name="p6768930132415"></a><a name="p6768930132415"></a>示例：输入2<sup id="sup2229141310268"><a name="sup2229141310268"></a><a name="sup2229141310268"></a>8</sup> - 1，输出2<sup id="sup137481182279"><a name="sup137481182279"></a><a name="sup137481182279"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row856117392381"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1979612472720"><a name="p1979612472720"></a><a name="p1979612472720"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p06852038192717"><a name="p06852038192717"></a><a name="p06852038192717"></a>将src以uint32_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p126853388276"><a name="p126853388276"></a><a name="p126853388276"></a>示例：输入2<sup id="sup156851638112714"><a name="sup156851638112714"></a><a name="sup156851638112714"></a>8</sup> - 1，输出2<sup id="sup16851638132717"><a name="sup16851638132717"></a><a name="sup16851638132717"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row75193574612"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p351985174613"><a name="p351985174613"></a><a name="p351985174613"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1451935194620"><a name="p1451935194620"></a><a name="p1451935194620"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p7140192014521"><a name="p7140192014521"></a><a name="p7140192014521"></a>将src以half格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p4618040113011"><a name="p4618040113011"></a><a name="p4618040113011"></a>示例：输入-1，输出-1.0。</p>
</td>
</tr>
<tr id="row72740404397"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p92551950153914"><a name="p92551950153914"></a><a name="p92551950153914"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p117806106289"><a name="p117806106289"></a><a name="p117806106289"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p778031019288"><a name="p778031019288"></a><a name="p778031019288"></a>将src以uint16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p4780710172813"><a name="p4780710172813"></a><a name="p4780710172813"></a>示例：输入2<sup id="sup1122026132918"><a name="sup1122026132918"></a><a name="sup1122026132918"></a>7</sup> - 1，输出2<sup id="sup1957607102919"><a name="sup1957607102919"></a><a name="sup1957607102919"></a>7</sup> - 1。</p>
</td>
</tr>
<tr id="row948064219394"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3780510102818"><a name="p3780510102818"></a><a name="p3780510102818"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7780111018282"><a name="p7780111018282"></a><a name="p7780111018282"></a>将src以uint32_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p16780310152815"><a name="p16780310152815"></a><a name="p16780310152815"></a>示例：输入2<sup id="sup1978019104286"><a name="sup1978019104286"></a><a name="sup1978019104286"></a>7</sup> - 1，输出2<sup id="sup1568719152918"><a name="sup1568719152918"></a><a name="sup1568719152918"></a>7</sup> - 1。</p>
</td>
</tr>
<tr id="row6602202472915"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p76027249298"><a name="p76027249298"></a><a name="p76027249298"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p13602192419298"><a name="p13602192419298"></a><a name="p13602192419298"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p162853308"><a name="p162853308"></a><a name="p162853308"></a>将src以uint8_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p12627513018"><a name="p12627513018"></a><a name="p12627513018"></a>示例：输入2<sup id="sup1462953302"><a name="sup1462953302"></a><a name="sup1462953302"></a>16</sup> - 1，输出2<sup id="sup141653516311"><a name="sup141653516311"></a><a name="sup141653516311"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row614518374294"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p214613762918"><a name="p214613762918"></a><a name="p214613762918"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19621759308"><a name="p19621759308"></a><a name="p19621759308"></a>将src以uint32_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p8627518307"><a name="p8627518307"></a><a name="p8627518307"></a>示例：输入2<sup id="sup1922622213212"><a name="sup1922622213212"></a><a name="sup1922622213212"></a>16</sup> - 1，输出2<sup id="sup1122692233215"><a name="sup1122692233215"></a><a name="sup1122692233215"></a>16</sup> - 1。</p>
</td>
</tr>
<tr id="row77907597515"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p8515163665216"><a name="p8515163665216"></a><a name="p8515163665216"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p12790165915516"><a name="p12790165915516"></a><a name="p12790165915516"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p125201712145516"><a name="p125201712145516"></a><a name="p125201712145516"></a>将src按照roundMode取到half所能表示的数，以half格式存入dst中。</p>
<p id="p444515755515"><a name="p444515755515"></a><a name="p444515755515"></a>示例：输入2<sup id="sup49961326125511"><a name="sup49961326125511"></a><a name="sup49961326125511"></a>12</sup> + 2，写成half的表示形式：2<sup id="sup7672537205516"><a name="sup7672537205516"></a><a name="sup7672537205516"></a>12</sup> * (1 + 2<sup id="sup818555811559"><a name="sup818555811559"></a><a name="sup818555811559"></a>-11</sup>)，要求E = 12 + 15 = 27，M = 2<sup id="sup144541154565"><a name="sup144541154565"></a><a name="sup144541154565"></a>-11</sup>：</p>
<p id="p34593020244"><a name="p34593020244"></a><a name="p34593020244"></a><a name="image25231243174110"></a><a name="image25231243174110"></a><span><img class="eddx" id="image25231243174110" src="figures/绘图1.png" width="339.33653250000003" height="107.74596000000001"></span></p>
<p id="p10445117165520"><a name="p10445117165520"></a><a name="p10445117165520"></a>由于half只有10bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p1544518712559"><a name="p1544518712559"></a><a name="p1544518712559"></a>CAST_RINT模式舍入得尾数0000000000，E = 27，M = 0，最终表示的结果为2<sup id="sup23122675613"><a name="sup23122675613"></a><a name="sup23122675613"></a>12</sup>；</p>
<p id="p1544518715520"><a name="p1544518715520"></a><a name="p1544518715520"></a>CAST_FLOOR模式舍入得尾数0000000000，E = 27，M = 0，最终表示的结果为2<sup id="sup61994343564"><a name="sup61994343564"></a><a name="sup61994343564"></a>12</sup>；</p>
<p id="p444515719558"><a name="p444515719558"></a><a name="p444515719558"></a>CAST_CEIL模式舍入得尾数0000000001，E = 27，M = 2<sup id="sup1454462525718"><a name="sup1454462525718"></a><a name="sup1454462525718"></a>-10</sup>，最终表示的结果为2<sup id="sup1364784245620"><a name="sup1364784245620"></a><a name="sup1364784245620"></a>12</sup> + 4；</p>
<p id="p114456775520"><a name="p114456775520"></a><a name="p114456775520"></a>CAST_ROUND模式舍入得尾数0000000001，E = 27，M = 2<sup id="sup1747161015579"><a name="sup1747161015579"></a><a name="sup1747161015579"></a>-10</sup>，最终表示的结果为2<sup id="sup16630248205618"><a name="sup16630248205618"></a><a name="sup16630248205618"></a>12 </sup>+ 4；</p>
<p id="p344515775519"><a name="p344515775519"></a><a name="p344515775519"></a>CAST_TRUNC模式舍入得尾数0000000000，E = 27，M = 0，最终表示的结果为2<sup id="sup10718135665616"><a name="sup10718135665616"></a><a name="sup10718135665616"></a>12</sup>。</p>
</td>
</tr>
<tr id="row15989522185211"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19897222527"><a name="p19897222527"></a><a name="p19897222527"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2316357105916"><a name="p2316357105916"></a><a name="p2316357105916"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p1236535035912"><a name="p1236535035912"></a><a name="p1236535035912"></a>示例：输入2<sup id="sup6183551106"><a name="sup6183551106"></a><a name="sup6183551106"></a>15</sup> - 1，输出2<sup id="sup126081317"><a name="sup126081317"></a><a name="sup126081317"></a>15</sup> - 1。</p>
</td>
</tr>
<tr id="row938615716408"><td class="cellrowborder" rowspan="4" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p7785101224115"><a name="p7785101224115"></a><a name="p7785101224115"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p18269125216329"><a name="p18269125216329"></a><a name="p18269125216329"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p736781316532"><a name="p736781316532"></a><a name="p736781316532"></a>将src以uint8_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p8278184318418"><a name="p8278184318418"></a><a name="p8278184318418"></a>负数输入会被视为异常。</p>
<p id="p183671137536"><a name="p183671137536"></a><a name="p183671137536"></a>示例：输入2<sup id="sup17367141395311"><a name="sup17367141395311"></a><a name="sup17367141395311"></a>15</sup> - 1，输出2<sup id="sup133671513105312"><a name="sup133671513105312"></a><a name="sup133671513105312"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row426345914400"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p82059119337"><a name="p82059119337"></a><a name="p82059119337"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p121694819537"><a name="p121694819537"></a><a name="p121694819537"></a>将src以uint32_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p83691738116"><a name="p83691738116"></a><a name="p83691738116"></a>负数输入会默认转换成0。</p>
<p id="p1421634865319"><a name="p1421634865319"></a><a name="p1421634865319"></a>示例：输入2<sup id="sup14216174865312"><a name="sup14216174865312"></a><a name="sup14216174865312"></a>15</sup> - 1，输出2<sup id="sup14216148165311"><a name="sup14216148165311"></a><a name="sup14216148165311"></a>15</sup> - 1。</p>
</td>
</tr>
<tr id="row139013144116"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p131497210333"><a name="p131497210333"></a><a name="p131497210333"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p16918519145415"><a name="p16918519145415"></a><a name="p16918519145415"></a>将src以int32_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p29191219135418"><a name="p29191219135418"></a><a name="p29191219135418"></a>示例：输入2<sup id="sup109191319135411"><a name="sup109191319135411"></a><a name="sup109191319135411"></a>15</sup> - 1，输出2<sup id="sup1191912196546"><a name="sup1191912196546"></a><a name="sup1191912196546"></a>15</sup> - 1。</p>
</td>
</tr>
<tr id="row10613833133918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1361313383910"><a name="p1361313383910"></a><a name="p1361313383910"></a>int4b_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p6613183303919"><a name="p6613183303919"></a><a name="p6613183303919"></a>将src以int4b_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p9779122791315"><a name="p9779122791315"></a><a name="p9779122791315"></a>示例：输入2<sup id="sup42771728131316"><a name="sup42771728131316"></a><a name="sup42771728131316"></a>32</sup> - 1，输出2<sup id="sup227852816139"><a name="sup227852816139"></a><a name="sup227852816139"></a>3</sup> - 1。</p>
</td>
</tr>
<tr id="row1181812119551"><td class="cellrowborder" rowspan="3" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p18818711115510"><a name="p18818711115510"></a><a name="p18818711115510"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p198188110556"><a name="p198188110556"></a><a name="p198188110556"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p2516181975618"><a name="p2516181975618"></a><a name="p2516181975618"></a>将src以uint8_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p135162191563"><a name="p135162191563"></a><a name="p135162191563"></a>示例：输入2<sup id="sup10516319155620"><a name="sup10516319155620"></a><a name="sup10516319155620"></a>32</sup> - 1，输出2<sup id="sup1516161985616"><a name="sup1516161985616"></a><a name="sup1516161985616"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row101380371556"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p413843719554"><a name="p413843719554"></a><a name="p413843719554"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p948111121575"><a name="p948111121575"></a><a name="p948111121575"></a>将src以uint16_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p16481181215574"><a name="p16481181215574"></a><a name="p16481181215574"></a>示例：输入2<sup id="sup14481161255717"><a name="sup14481161255717"></a><a name="sup14481161255717"></a>32</sup> - 1，输出2<sup id="sup948121235716"><a name="sup948121235716"></a><a name="sup948121235716"></a>16</sup> - 1。</p>
</td>
</tr>
<tr id="row152431640125515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p162432040135512"><a name="p162432040135512"></a><a name="p162432040135512"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1643521111576"><a name="p1643521111576"></a><a name="p1643521111576"></a>将src以int16_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p1543551110571"><a name="p1543551110571"></a><a name="p1543551110571"></a>示例：输入2<sup id="sup1943551135714"><a name="sup1943551135714"></a><a name="sup1943551135714"></a>32</sup> - 1，输出2<sup id="sup1435171125713"><a name="sup1435171125713"></a><a name="sup1435171125713"></a>15</sup> - 1。</p>
</td>
</tr>
<tr id="row15991928133511"><td class="cellrowborder" rowspan="4" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p83201943133513"><a name="p83201943133513"></a><a name="p83201943133513"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p19100132843510"><a name="p19100132843510"></a><a name="p19100132843510"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p19695047163718"><a name="p19695047163718"></a><a name="p19695047163718"></a>将src按照roundMode取到float所能表示的数，以float格式存入dst中。</p>
<p id="p1997114011372"><a name="p1997114011372"></a><a name="p1997114011372"></a>示例：输入2<sup id="sup69714407371"><a name="sup69714407371"></a><a name="sup69714407371"></a>25</sup> + 3，写成float的表示形式：2<sup id="sup1294995610373"><a name="sup1294995610373"></a><a name="sup1294995610373"></a>25</sup> * (1 + 2<sup id="sup1334610153816"><a name="sup1334610153816"></a><a name="sup1334610153816"></a>-24</sup> + 2<sup id="sup273410198386"><a name="sup273410198386"></a><a name="sup273410198386"></a>-25</sup>)，要求E = 25 + 127 = 152，   M = 2<sup id="sup9565173015402"><a name="sup9565173015402"></a><a name="sup9565173015402"></a>-24</sup> + 2<sup id="sup760021704012"><a name="sup760021704012"></a><a name="sup760021704012"></a>-25。</sup></p>
<p id="p51981334183318"><a name="p51981334183318"></a><a name="p51981334183318"></a><a name="image292993463310"></a><a name="image292993463310"></a><span><img class="eddx" id="image292993463310" src="figures/流水任务运行示意图-105.png"></span></p>
<p id="p397214015374"><a name="p397214015374"></a><a name="p397214015374"></a>由于float只有23bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p18972104083713"><a name="p18972104083713"></a><a name="p18972104083713"></a>CAST_RINT模式舍入得尾数00000000000000000000001，E = 152，M = 2<sup id="sup13448105444217"><a name="sup13448105444217"></a><a name="sup13448105444217"></a>-23</sup>，最终表示的结果为2<sup id="sup284820194314"><a name="sup284820194314"></a><a name="sup284820194314"></a>25</sup> + 4；</p>
<p id="p49721840143717"><a name="p49721840143717"></a><a name="p49721840143717"></a>CAST_FLOOR模式舍入得尾数00000000000000000000000，E = 152，M = 0，最终表示的结果为2<sup id="sup072718128436"><a name="sup072718128436"></a><a name="sup072718128436"></a>25</sup>；</p>
<p id="p199729409376"><a name="p199729409376"></a><a name="p199729409376"></a>CAST_CEIL模式舍入得尾数00000000000000000000001，E = 152，M = 2<sup id="sup635212247437"><a name="sup635212247437"></a><a name="sup635212247437"></a>-23</sup>，最终表示的结果为2<sup id="sup1038512302435"><a name="sup1038512302435"></a><a name="sup1038512302435"></a>25 </sup>+ 4；</p>
<p id="p49721140203720"><a name="p49721140203720"></a><a name="p49721140203720"></a>CAST_ROUND模式舍入得尾数00000000000000000000001，E = 152，M = 2<sup id="sup1133994716435"><a name="sup1133994716435"></a><a name="sup1133994716435"></a>-23</sup>，最终表示的结果为2<sup id="sup41230524447"><a name="sup41230524447"></a><a name="sup41230524447"></a>25</sup> + 4；</p>
<p id="p159721040193714"><a name="p159721040193714"></a><a name="p159721040193714"></a>CAST_TRUNC模式舍入得尾数00000000000000000000000，E = 152，M = 0，最终表示的结果为2<sup id="sup13948634453"><a name="sup13948634453"></a><a name="sup13948634453"></a>25</sup> 。</p>
</td>
</tr>
<tr id="row531412311355"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12176114745517"><a name="p12176114745517"></a><a name="p12176114745517"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1953175495614"><a name="p1953175495614"></a><a name="p1953175495614"></a>将src以int64_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p292254115564"><a name="p292254115564"></a><a name="p292254115564"></a>示例：输入2<sup id="sup636818184578"><a name="sup636818184578"></a><a name="sup636818184578"></a>31</sup> - 1，输出2<sup id="sup51808519575"><a name="sup51808519575"></a><a name="sup51808519575"></a>31</sup> - 1。</p>
</td>
</tr>
<tr id="row28601333133511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1086011335356"><a name="p1086011335356"></a><a name="p1086011335356"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p27661913588"><a name="p27661913588"></a><a name="p27661913588"></a>将src以int16_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p14807195145812"><a name="p14807195145812"></a><a name="p14807195145812"></a>示例：输入2<sup id="sup1890394155817"><a name="sup1890394155817"></a><a name="sup1890394155817"></a>31</sup> - 1，输出2<sup id="sup2911124916585"><a name="sup2911124916585"></a><a name="sup2911124916585"></a>15</sup> - 1。</p>
</td>
</tr>
<tr id="row196317472242"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p10631174742410"><a name="p10631174742410"></a><a name="p10631174742410"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p0586519182516"><a name="p0586519182516"></a><a name="p0586519182516"></a>与<a href="SetDeqScale.md">SetDeqScale(half scale)</a>接口配合使用，输出src / 2<sup id="sup1655219427288"><a name="sup1655219427288"></a><a name="sup1655219427288"></a>17</sup> * scale * 2<sup id="sup1057615072914"><a name="sup1057615072914"></a><a name="sup1057615072914"></a>17</sup>。</p>
</td>
</tr>
<tr id="row9465174420424"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p238715344311"><a name="p238715344311"></a><a name="p238715344311"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p10211716105815"><a name="p10211716105815"></a><a name="p10211716105815"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p6389121315911"><a name="p6389121315911"></a><a name="p6389121315911"></a>将src以uint8_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p718620521411"><a name="p718620521411"></a><a name="p718620521411"></a>负数输入会被视为异常。</p>
<p id="p238941316596"><a name="p238941316596"></a><a name="p238941316596"></a>示例：输入2<sup id="sup13891013105919"><a name="sup13891013105919"></a><a name="sup13891013105919"></a>31</sup> - 1，输出2<sup id="sup43891313175914"><a name="sup43891313175914"></a><a name="sup43891313175914"></a>8</sup> - 1。</p>
</td>
</tr>
<tr id="row10520446114216"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p10674123215582"><a name="p10674123215582"></a><a name="p10674123215582"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1639519019599"><a name="p1639519019599"></a><a name="p1639519019599"></a>将src以uint16_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p817617542048"><a name="p817617542048"></a><a name="p817617542048"></a>负数输入会被视为异常。</p>
<p id="p83952005914"><a name="p83952005914"></a><a name="p83952005914"></a>示例：输入2<sup id="sup193951804597"><a name="sup193951804597"></a><a name="sup193951804597"></a>31</sup> - 1，输出2<sup id="sup639517019592"><a name="sup639517019592"></a><a name="sup639517019592"></a>16</sup> - 1。</p>
</td>
</tr>
<tr id="row0442182465912"><td class="cellrowborder" rowspan="3" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p204201732205914"><a name="p204201732205914"></a><a name="p204201732205914"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1044220242594"><a name="p1044220242594"></a><a name="p1044220242594"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p9625115116017"><a name="p9625115116017"></a><a name="p9625115116017"></a>将src以int32_t格式（溢出默认按照饱和处理）存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p27771645800"><a name="p27771645800"></a><a name="p27771645800"></a>示例：输入2<sup id="sup2458741013"><a name="sup2458741013"></a><a name="sup2458741013"></a>31</sup>，输出2<sup id="sup148020109118"><a name="sup148020109118"></a><a name="sup148020109118"></a>31</sup> - 1。</p>
</td>
</tr>
<tr id="row568412712590"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19685112745913"><a name="p19685112745913"></a><a name="p19685112745913"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18539428311"><a name="p18539428311"></a><a name="p18539428311"></a>将src按照roundMode取到float所能表示的数，以float格式存入dst中。</p>
<p id="p35827220112"><a name="p35827220112"></a><a name="p35827220112"></a>示例：输入2<sup id="sup1522711375111"><a name="sup1522711375111"></a><a name="sup1522711375111"></a>35 </sup>+ 2<sup id="sup96431442216"><a name="sup96431442216"></a><a name="sup96431442216"></a>12</sup> + 2<sup id="sup379611487112"><a name="sup379611487112"></a><a name="sup379611487112"></a>11</sup>，写成float的表示形式：2<sup id="sup105081957114"><a name="sup105081957114"></a><a name="sup105081957114"></a>35</sup> * (1 + 2<sup id="sup175561783219"><a name="sup175561783219"></a><a name="sup175561783219"></a>-23</sup> + 2<sup id="sup12756181026"><a name="sup12756181026"></a><a name="sup12756181026"></a>-24</sup>)，要求E = 35 + 127 = 162，M = 2<sup id="sup10267529624"><a name="sup10267529624"></a><a name="sup10267529624"></a>-23</sup> + 2<sup id="sup158063371229"><a name="sup158063371229"></a><a name="sup158063371229"></a>-24</sup>。</p>
<p id="p16903333433"><a name="p16903333433"></a><a name="p16903333433"></a><a name="image0647154104315"></a><a name="image0647154104315"></a><span><img class="eddx" id="image0647154104315" src="figures/流水任务运行示意图-106.png"></span></p>
<p id="p155825221312"><a name="p155825221312"></a><a name="p155825221312"></a>由于float只有23bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p8582132210117"><a name="p8582132210117"></a><a name="p8582132210117"></a>CAST_RINT模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup74601313311"><a name="sup74601313311"></a><a name="sup74601313311"></a>-22</sup>，最终表示的结果为2<sup id="sup7461220416"><a name="sup7461220416"></a><a name="sup7461220416"></a>35</sup> + 2<sup id="sup0533108045"><a name="sup0533108045"></a><a name="sup0533108045"></a>13</sup>；</p>
<p id="p1158220221518"><a name="p1158220221518"></a><a name="p1158220221518"></a>CAST_FLOOR模式舍入得尾数00000000000000000000001，E = 162，M = 2<sup id="sup8372015637"><a name="sup8372015637"></a><a name="sup8372015637"></a>-23</sup>，最终表示的结果为2<sup id="sup1103112312418"><a name="sup1103112312418"></a><a name="sup1103112312418"></a>25</sup> + 2<sup id="sup20789628443"><a name="sup20789628443"></a><a name="sup20789628443"></a>12</sup>；</p>
<p id="p55829221310"><a name="p55829221310"></a><a name="p55829221310"></a>CAST_CEIL模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup158049263311"><a name="sup158049263311"></a><a name="sup158049263311"></a>-22</sup>，最终表示的结果为2<sup id="sup436514362413"><a name="sup436514362413"></a><a name="sup436514362413"></a>25</sup>  + 2<sup id="sup1669314421043"><a name="sup1669314421043"></a><a name="sup1669314421043"></a>13</sup>；</p>
<p id="p195827221013"><a name="p195827221013"></a><a name="p195827221013"></a>CAST_ROUND模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup197971036435"><a name="sup197971036435"></a><a name="sup197971036435"></a>-22</sup>，最终表示的结果为2<sup id="sup2047713501748"><a name="sup2047713501748"></a><a name="sup2047713501748"></a>25</sup> + 2<sup id="sup15186561642"><a name="sup15186561642"></a><a name="sup15186561642"></a>13</sup>；</p>
<p id="p3582622110"><a name="p3582622110"></a><a name="p3582622110"></a>CAST_TRUNC模式舍入得尾数00000000000000000000001，E = 162，M = 2<sup id="sup368612461739"><a name="sup368612461739"></a><a name="sup368612461739"></a>-23</sup>，最终表示的结果为2<sup id="sup132863181015"><a name="sup132863181015"></a><a name="sup132863181015"></a>25 </sup>+ 2<sup id="sup3727137161014"><a name="sup3727137161014"></a><a name="sup3727137161014"></a>12</sup>。</p>
</td>
</tr>
<tr id="row1164510511086"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p464515112813"><a name="p464515112813"></a><a name="p464515112813"></a>double</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p136451151983"><a name="p136451151983"></a><a name="p136451151983"></a>将src按照roundMode取到double所能表示的数，以double格式存入dst中。</p>
<p id="p1450664214619"><a name="p1450664214619"></a><a name="p1450664214619"></a>示例：输入2<sup id="sup422116539329"><a name="sup422116539329"></a><a name="sup422116539329"></a>61 </sup>+ 2<sup id="sup1722135318328"><a name="sup1722135318328"></a><a name="sup1722135318328"></a>9</sup> + 2<sup id="sup922117538329"><a name="sup922117538329"></a><a name="sup922117538329"></a>8</sup>，写成float的表示形式：2<sup id="sup19221115333214"><a name="sup19221115333214"></a><a name="sup19221115333214"></a>61</sup> * (1 + 2<sup id="sup522135313214"><a name="sup522135313214"></a><a name="sup522135313214"></a>-52</sup> + 2<sup id="sup82211553153216"><a name="sup82211553153216"></a><a name="sup82211553153216"></a>-53</sup>)，要求E = 61 + 1023 = 1084，M =2<sup id="sup2666125317560"><a name="sup2666125317560"></a><a name="sup2666125317560"></a>-52</sup> + 2<sup id="sup1666620539569"><a name="sup1666620539569"></a><a name="sup1666620539569"></a>-53</sup>。</p>
<p id="p1105553141816"><a name="p1105553141816"></a><a name="p1105553141816"></a><a name="image4105115314181"></a><a name="image4105115314181"></a><span><img class="eddx" id="image4105115314181" src="figures/int64.png"></span></p>
<p id="p1065044112219"><a name="p1065044112219"></a><a name="p1065044112219"></a>由于double只有52bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p108241335163912"><a name="p108241335163912"></a><a name="p108241335163912"></a>CAST_RINT模式舍入得尾数0000000000000000000000000000000000000000000000000010，E = 1084，M = 2<sup id="sup188246351399"><a name="sup188246351399"></a><a name="sup188246351399"></a>-51</sup>，最终表示的结果为2<sup id="sup148245359392"><a name="sup148245359392"></a><a name="sup148245359392"></a>61</sup> + 2<sup id="sup1382423503915"><a name="sup1382423503915"></a><a name="sup1382423503915"></a>10</sup>；</p>
<p id="p3824135183911"><a name="p3824135183911"></a><a name="p3824135183911"></a>CAST_FLOOR模式舍入得尾数0000000000000000000000000000000000000000000000000001，E = 1084，M = 2<sup id="sup28241535203915"><a name="sup28241535203915"></a><a name="sup28241535203915"></a>-52</sup>，最终表示的结果为2<sup id="sup7882134915458"><a name="sup7882134915458"></a><a name="sup7882134915458"></a>61</sup> + 2<sup id="sup388224964514"><a name="sup388224964514"></a><a name="sup388224964514"></a>9</sup>；</p>
<p id="p12824183513911"><a name="p12824183513911"></a><a name="p12824183513911"></a>CAST_CEIL模式舍入得尾数0000000000000000000000000000000000000000000000000010，E = 1084，M =2<sup id="sup1852133713579"><a name="sup1852133713579"></a><a name="sup1852133713579"></a>-51</sup>，最终表示的结果为2<sup id="sup4621135184611"><a name="sup4621135184611"></a><a name="sup4621135184611"></a>61</sup> + 2<sup id="sup972715416585"><a name="sup972715416585"></a><a name="sup972715416585"></a>10</sup>；</p>
<p id="p982519358394"><a name="p982519358394"></a><a name="p982519358394"></a>CAST_ROUND模式舍入得尾数0000000000000000000000000000000000000000000000000010，E = 1084，M = 2<sup id="sup051674015573"><a name="sup051674015573"></a><a name="sup051674015573"></a>-51</sup>，最终表示的结果为2<sup id="sup2079891010461"><a name="sup2079891010461"></a><a name="sup2079891010461"></a>61</sup> + 2<sup id="sup89592355585"><a name="sup89592355585"></a><a name="sup89592355585"></a>10</sup>；</p>
<p id="p382510353397"><a name="p382510353397"></a><a name="p382510353397"></a>CAST_TRUNC模式舍入得尾数0000000000000000000000000000000000000000000000000001，E = 1084，M = 2<sup id="sup13469950155714"><a name="sup13469950155714"></a><a name="sup13469950155714"></a>-52</sup>，最终表示的结果为2<sup id="sup26951317164618"><a name="sup26951317164618"></a><a name="sup26951317164618"></a>61</sup> + 2<sup id="sup129991173588"><a name="sup129991173588"></a><a name="sup129991173588"></a>9</sup>。</p>
<p id="p127321816173911"><a name="p127321816173911"></a><a name="p127321816173911"></a>注：仅支持tensor前n个数据计算接口。</p>
</td>
</tr>
<tr id="row17902111113546"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p69021211145411"><a name="p69021211145411"></a><a name="p69021211145411"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1590241115540"><a name="p1590241115540"></a><a name="p1590241115540"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p425371210557"><a name="p425371210557"></a><a name="p425371210557"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p32531412165513"><a name="p32531412165513"></a><a name="p32531412165513"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row94690317546"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1651023565411"><a name="p1651023565411"></a><a name="p1651023565411"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p311141418553"><a name="p311141418553"></a><a name="p311141418553"></a>将src以half格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p811141414554"><a name="p811141414554"></a><a name="p811141414554"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row16342142365611"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1634282335620"><a name="p1634282335620"></a><a name="p1634282335620"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1134262385618"><a name="p1134262385618"></a><a name="p1134262385618"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p1055761635719"><a name="p1055761635719"></a><a name="p1055761635719"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p14557216155718"><a name="p14557216155718"></a><a name="p14557216155718"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row10771174712566"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1350210512566"><a name="p1350210512566"></a><a name="p1350210512566"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p127711847185612"><a name="p127711847185612"></a><a name="p127711847185612"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p13352141755714"><a name="p13352141755714"></a><a name="p13352141755714"></a>将src以float格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p183521117195716"><a name="p183521117195716"></a><a name="p183521117195716"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row18609176125819"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p46099615815"><a name="p46099615815"></a><a name="p46099615815"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p660966125818"><a name="p660966125818"></a><a name="p660966125818"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p1731011222589"><a name="p1731011222589"></a><a name="p1731011222589"></a>将src以bfloat16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p231022220582"><a name="p231022220582"></a><a name="p231022220582"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row1671353165817"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p848475845817"><a name="p848475845817"></a><a name="p848475845817"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p7484195895819"><a name="p7484195895819"></a><a name="p7484195895819"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p17484185845816"><a name="p17484185845816"></a><a name="p17484185845816"></a>将src以bfloat16_t格式存入dst中，不存在精度转换问题，无舍入模式。</p>
<p id="p2484155813588"><a name="p2484155813588"></a><a name="p2484155813588"></a>示例：输入2，输出2。</p>
</td>
</tr>
<tr id="row265501053114"><td class="cellrowborder" rowspan="2" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p196551410183110"><a name="p196551410183110"></a><a name="p196551410183110"></a>complex64</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1265551003119"><a name="p1265551003119"></a><a name="p1265551003119"></a>complex64</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p2385183173512"><a name="p2385183173512"></a><a name="p2385183173512"></a>complex64实部和虚部都是float类型，参考float与float之间的精度转换规则。</p>
</td>
</tr>
<tr id="row152211611163920"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1122111115397"><a name="p1122111115397"></a><a name="p1122111115397"></a>complex32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p122218110399"><a name="p122218110399"></a><a name="p122218110399"></a>complex64实部和虚部都是float类型，complex32实部和虚部都是half类型，参考从float到half的精度转换规则。</p>
</td>
</tr>
<tr id="row590512404117"><td class="cellrowborder" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p13906194194114"><a name="p13906194194114"></a><a name="p13906194194114"></a>complex32</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1190619419415"><a name="p1190619419415"></a><a name="p1190619419415"></a>complex64</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p66810235417"><a name="p66810235417"></a><a name="p66810235417"></a>complex64实部和虚部都是float类型，complex32实部和虚部都是half类型，参考从half到float的精度转换规则。</p>
</td>
</tr>
<tr id="row959125135216"><td class="cellrowborder" rowspan="4" valign="top" width="10.431043104310431%" headers="mcps1.2.4.1.1 "><p id="p1513114225217"><a name="p1513114225217"></a><a name="p1513114225217"></a>double</p>
</td>
<td class="cellrowborder" valign="top" width="10.46104610461046%" headers="mcps1.2.4.1.2 "><p id="p1959142513524"><a name="p1959142513524"></a><a name="p1959142513524"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="79.1079107910791%" headers="mcps1.2.4.1.3 "><p id="p145992515522"><a name="p145992515522"></a><a name="p145992515522"></a>将src按照roundMode取到float所能表示的数，以float格式存入dst中。</p>
<p id="p767831912500"><a name="p767831912500"></a><a name="p767831912500"></a>示例：输入2<sup id="sup14321333175014"><a name="sup14321333175014"></a><a name="sup14321333175014"></a>35 </sup>+ 2<sup id="sup193214332507"><a name="sup193214332507"></a><a name="sup193214332507"></a>12</sup> + 2<sup id="sup143211337505"><a name="sup143211337505"></a><a name="sup143211337505"></a>11</sup>，写成float的表示形式：2<sup id="sup17321133311509"><a name="sup17321133311509"></a><a name="sup17321133311509"></a>35</sup> * (1 + 2<sup id="sup1532183395018"><a name="sup1532183395018"></a><a name="sup1532183395018"></a>-23</sup> + 2<sup id="sup173211233125014"><a name="sup173211233125014"></a><a name="sup173211233125014"></a>-24</sup>)，要求E = 1058 - 1023 + 127 = 162，M = 2<sup id="sup03211533155012"><a name="sup03211533155012"></a><a name="sup03211533155012"></a>-23</sup> + 2<sup id="sup232113311509"><a name="sup232113311509"></a><a name="sup232113311509"></a>-24</sup>。</p>
<p id="p7847111513138"><a name="p7847111513138"></a><a name="p7847111513138"></a><a name="image17847151521311"></a><a name="image17847151521311"></a><span><img class="eddx" id="image17847151521311" src="figures/float.png"></span></p>
<p id="p1913019473556"><a name="p1913019473556"></a><a name="p1913019473556"></a>由于float只有8bit指数位，指数部分需要转换，只有23bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p113004785512"><a name="p113004785512"></a><a name="p113004785512"></a>CAST_RINT模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup141303472555"><a name="sup141303472555"></a><a name="sup141303472555"></a>-22</sup>，最终表示的结果为2<sup id="sup1613010477552"><a name="sup1613010477552"></a><a name="sup1613010477552"></a>35</sup> + 2<sup id="sup10130247195519"><a name="sup10130247195519"></a><a name="sup10130247195519"></a>13</sup>；</p>
<p id="p181301447185517"><a name="p181301447185517"></a><a name="p181301447185517"></a>CAST_FLOOR模式舍入得尾数00000000000000000000001，E = 162，M = 2<sup id="sup21301147125519"><a name="sup21301147125519"></a><a name="sup21301147125519"></a>-23</sup>，最终表示的结果为2<sup id="sup51302479556"><a name="sup51302479556"></a><a name="sup51302479556"></a>35</sup> + 2<sup id="sup6130184719556"><a name="sup6130184719556"></a><a name="sup6130184719556"></a>12</sup>；</p>
<p id="p41301047125515"><a name="p41301047125515"></a><a name="p41301047125515"></a>CAST_CEIL模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup1130847145514"><a name="sup1130847145514"></a><a name="sup1130847145514"></a>-22</sup>，最终表示的结果为2<sup id="sup11130147205511"><a name="sup11130147205511"></a><a name="sup11130147205511"></a>35</sup>  + 2<sup id="sup1113014479552"><a name="sup1113014479552"></a><a name="sup1113014479552"></a>13</sup>；</p>
<p id="p81307474551"><a name="p81307474551"></a><a name="p81307474551"></a>CAST_ROUND模式舍入得尾数00000000000000000000010，E = 162，M = 2<sup id="sup013011470559"><a name="sup013011470559"></a><a name="sup013011470559"></a>-22</sup>，最终表示的结果为2<sup id="sup1913011472551"><a name="sup1913011472551"></a><a name="sup1913011472551"></a>35</sup> + 2<sup id="sup313084735518"><a name="sup313084735518"></a><a name="sup313084735518"></a>13</sup>；</p>
<p id="p5130194765511"><a name="p5130194765511"></a><a name="p5130194765511"></a>CAST_TRUNC模式舍入得尾数00000000000000000000001，E = 162，M = 2<sup id="sup313044755519"><a name="sup313044755519"></a><a name="sup313044755519"></a>-23</sup>，最终表示的结果为2<sup id="sup16131747175512"><a name="sup16131747175512"></a><a name="sup16131747175512"></a>35 </sup>+ 2<sup id="sup91312479553"><a name="sup91312479553"></a><a name="sup91312479553"></a>12</sup>。</p>
<p id="p1813339143911"><a name="p1813339143911"></a><a name="p1813339143911"></a>注：仅支持tensor前n个数据计算接口。</p>
</td>
</tr>
<tr id="row26006433293"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p549413304521"><a name="p549413304521"></a><a name="p549413304521"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p04947308528"><a name="p04947308528"></a><a name="p04947308528"></a>将src按照roundMode取到bfloat_16t所能表示的数，以bfloat_16t格式存入dst中。</p>
<p id="p1264614211145"><a name="p1264614211145"></a><a name="p1264614211145"></a>示例：输入2<sup id="sup064642191414"><a name="sup064642191414"></a><a name="sup064642191414"></a>35 </sup>+ 2<sup id="sup86469217147"><a name="sup86469217147"></a><a name="sup86469217147"></a>28</sup> + 2<sup id="sup1064612131415"><a name="sup1064612131415"></a><a name="sup1064612131415"></a>27</sup>，写成bfloat_16的表示形式：2<sup id="sup86461625143"><a name="sup86461625143"></a><a name="sup86461625143"></a>35</sup> * (1 + 2<sup id="sup064617221412"><a name="sup064617221412"></a><a name="sup064617221412"></a>-7</sup> + 2<sup id="sup106461528148"><a name="sup106461528148"></a><a name="sup106461528148"></a>-8</sup>)，要求E = 1058 - 1023 + 127 = 162，M = 2<sup id="sup12151173915537"><a name="sup12151173915537"></a><a name="sup12151173915537"></a>-7</sup> + 2<sup id="sup151511392538"><a name="sup151511392538"></a><a name="sup151511392538"></a>-8</sup>。</p>
<p id="p636054813104"><a name="p636054813104"></a><a name="p636054813104"></a><a name="image1636014801013"></a><a name="image1636014801013"></a><span><img class="eddx" id="image1636014801013" src="figures/绘图2.png"></span></p>
<p id="p564652111410"><a name="p564652111410"></a><a name="p564652111410"></a>由于bfloat_16t只有8bit指数位，指数部分需要转换，只有7bit尾数位，因此灰色部分要进行舍入。</p>
<p id="p1164615215146"><a name="p1164615215146"></a><a name="p1164615215146"></a>CAST_RINT模式舍入得尾数0000010，E = 162，M = 2<sup id="sup664662181411"><a name="sup664662181411"></a><a name="sup664662181411"></a>-6</sup>，最终表示的结果为2<sup id="sup1364620213141"><a name="sup1364620213141"></a><a name="sup1364620213141"></a>35</sup> + 2<sup id="sup16462027147"><a name="sup16462027147"></a><a name="sup16462027147"></a>29</sup>；</p>
<p id="p116461251410"><a name="p116461251410"></a><a name="p116461251410"></a>CAST_FLOOR模式舍入得尾数0000001，E = 162，M = 2<sup id="sup156462211415"><a name="sup156462211415"></a><a name="sup156462211415"></a>-7</sup>，最终表示的结果为2<sup id="sup36467211144"><a name="sup36467211144"></a><a name="sup36467211144"></a>35</sup> + 2<sup id="sup1764619251412"><a name="sup1764619251412"></a><a name="sup1764619251412"></a>28</sup>；</p>
<p id="p1564614241412"><a name="p1564614241412"></a><a name="p1564614241412"></a>CAST_CEIL模式舍入得尾数0000010，E = 162，M = 2<sup id="sup17646152141411"><a name="sup17646152141411"></a><a name="sup17646152141411"></a>-6</sup>，最终表示的结果为2<sup id="sup166462221419"><a name="sup166462221419"></a><a name="sup166462221419"></a>35</sup>  + 2<sup id="sup1161636155310"><a name="sup1161636155310"></a><a name="sup1161636155310"></a>29</sup>；</p>
<p id="p196469211149"><a name="p196469211149"></a><a name="p196469211149"></a>CAST_ROUND模式舍入得尾数0000010，E = 162，M = 2<sup id="sup186465221415"><a name="sup186465221415"></a><a name="sup186465221415"></a>-6</sup>，最终表示的结果为2<sup id="sup1864662191417"><a name="sup1864662191417"></a><a name="sup1864662191417"></a>35</sup> + 2<sup id="sup3820940135310"><a name="sup3820940135310"></a><a name="sup3820940135310"></a>29</sup>；</p>
<p id="p1164613221416"><a name="p1164613221416"></a><a name="p1164613221416"></a>CAST_TRUNC模式舍入得尾数0000001，E = 162，M = 2<sup id="sup186464213143"><a name="sup186464213143"></a><a name="sup186464213143"></a>-7</sup>，最终表示的结果为2<sup id="sup66465216146"><a name="sup66465216146"></a><a name="sup66465216146"></a>35 </sup>+ 2<sup id="sup843214475538"><a name="sup843214475538"></a><a name="sup843214475538"></a>28</sup>。</p>
<p id="p13489542133917"><a name="p13489542133917"></a><a name="p13489542133917"></a>注：仅支持tensor前n个数据计算接口。</p>
</td>
</tr>
<tr id="row216643992918"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p816611396297"><a name="p816611396297"></a><a name="p816611396297"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2739182619301"><a name="p2739182619301"></a><a name="p2739182619301"></a>将src按照roundMode取整，以int32_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p1473942673011"><a name="p1473942673011"></a><a name="p1473942673011"></a>示例：输入-1.5，</p>
<p id="p373982617304"><a name="p373982617304"></a><a name="p373982617304"></a>CAST_TRUNC模式输出-1。</p>
</td>
</tr>
<tr id="row1649393055210"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p22691947172915"><a name="p22691947172915"></a><a name="p22691947172915"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p17897151363111"><a name="p17897151363111"></a><a name="p17897151363111"></a>将src按照roundMode取整，以int64_t格式（溢出默认按照饱和处理）存入dst中。</p>
<p id="p989716135315"><a name="p989716135315"></a><a name="p989716135315"></a>示例：输入-1.5，</p>
<p id="p28973139317"><a name="p28973139317"></a><a name="p28973139317"></a>CAST_TRUNC模式输出-1。</p>
</td>
</tr>
</tbody>
</table>

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, typename U>
    __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 2**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="16.35%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.65%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11492616168"><td class="cellrowborder" valign="top" width="16.35%" headers="mcps1.2.3.1.1 "><p id="p19933113132715"><a name="p19933113132715"></a><a name="p19933113132715"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.65%" headers="mcps1.2.3.1.2 "><p id="p593343122716"><a name="p593343122716"></a><a name="p593343122716"></a>目的操作数数据类型。</p>
<p id="p0280162816616"><a name="p0280162816616"></a><a name="p0280162816616"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型见<a href="#table75331133863">表4</a></p>
</td>
</tr>
<tr id="row1835857145817"><td class="cellrowborder" valign="top" width="16.35%" headers="mcps1.2.3.1.1 "><p id="p5979215341"><a name="p5979215341"></a><a name="p5979215341"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.65%" headers="mcps1.2.3.1.2 "><p id="p1526974512618"><a name="p1526974512618"></a><a name="p1526974512618"></a>源操作数数据类型。</p>
<p id="p1599417248328"><a name="p1599417248328"></a><a name="p1599417248328"></a><span id="ph2099414246325"><a name="ph2099414246325"></a><a name="ph2099414246325"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型见<a href="#table75331133863">表4</a></p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="16.35%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="83.65%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.92%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.59%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p358015811379"><a name="p358015811379"></a><a name="p358015811379"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p184912374359"><a name="p184912374359"></a><a name="p184912374359"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p195531113161311"><a name="p195531113161311"></a><a name="p195531113161311"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p155310135134"><a name="p155310135134"></a><a name="p155310135134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p1743571233810"><a name="p1743571233810"></a><a name="p1743571233810"></a>源操作数。</p>
<p id="p169251414143819"><a name="p169251414143819"></a><a name="p169251414143819"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p15798144016353"><a name="p15798144016353"></a><a name="p15798144016353"></a><span id="ph13451134110354"><a name="ph13451134110354"></a><a name="ph13451134110354"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1450110360599"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p650118368599"><a name="p650118368599"></a><a name="p650118368599"></a>roundMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p6502936195912"><a name="p6502936195912"></a><a name="p6502936195912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p8502173605915"><a name="p8502173605915"></a><a name="p8502173605915"></a>精度转换处理模式，类型是RoundMode。</p>
<p id="p2402542181412"><a name="p2402542181412"></a><a name="p2402542181412"></a>RoundMode为枚举类型，用以控制精度转换处理模式，具体定义为：</p>
<a name="screen09015195591"></a><a name="screen09015195591"></a><pre class="screen" codetype="Cpp" id="screen09015195591">enum class RoundMode {
    CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
    CAST_RINT,      // rint，四舍六入五成双舍入
    CAST_FLOOR,     // floor，向负无穷舍入
    CAST_CEIL,      // ceil，向正无穷舍入
    CAST_ROUND,     // round，四舍五入舍入
    CAST_TRUNC,     // trunc，向零舍入
    CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入
    CAST_HYBRID,    // hybrid，目前特指输出结果是hif8数据时，会用到的一种随机舍入 
};</pre>
</td>
</tr>
<tr id="row1435424811239"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p497919583239"><a name="p497919583239"></a><a name="p497919583239"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p1197975815231"><a name="p1197975815231"></a><a name="p1197975815231"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p4980125882310"><a name="p4980125882310"></a><a name="p4980125882310"></a>参与计算的元素个数。</p>
</td>
</tr>
<tr id="row16554713131317"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p755431341319"><a name="p755431341319"></a><a name="p755431341319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row185542138131"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p755471321311"><a name="p755471321311"></a><a name="p755471321311"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p135541313101314"><a name="p135541313101314"></a><a name="p135541313101314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p10767143753811"><a name="p10767143753811"></a><a name="p10767143753811"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数，repeatTime∈[0,255]。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row195541813181310"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p15554121320132"><a name="p15554121320132"></a><a name="p15554121320132"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p18554141331317"><a name="p18554141331317"></a><a name="p18554141331317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p455461351319"><a name="p455461351319"></a><a name="p455461351319"></a>控制操作数地址步长的参数，<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>类型。包含操作数相邻迭代间的地址步长，操作数同一迭代内datablock的地址步长等参数。其中dstRepStride/srcRepStride∈[0,255]。</p>
<p id="p1156819418442"><a name="p1156819418442"></a><a name="p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 4** Ascend 950PR/Ascend 950DTCast指令参数说明

<a name="table75331133863"></a>
<table><thead align="left"><tr id="row1753318337613"><th class="cellrowborder" valign="top" width="14.26%" id="mcps1.2.4.1.1"><p id="p1353410334610"><a name="p1353410334610"></a><a name="p1353410334610"></a>src数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="14.67%" id="mcps1.2.4.1.2"><p id="p8534163317610"><a name="p8534163317610"></a><a name="p8534163317610"></a>dst数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="71.07%" id="mcps1.2.4.1.3"><p id="p205344335614"><a name="p205344335614"></a><a name="p205344335614"></a>支持的roundMode</p>
</th>
</tr>
</thead>
<tbody><tr id="row1553463311620"><td class="cellrowborder" rowspan="9" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p557703765612"><a name="p557703765612"></a><a name="p557703765612"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p125779377564"><a name="p125779377564"></a><a name="p125779377564"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p18933501576"><a name="p18933501576"></a><a name="p18933501576"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row55344337613"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8577123775615"><a name="p8577123775615"></a><a name="p8577123775615"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1241315172583"><a name="p1241315172583"></a><a name="p1241315172583"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC/CAST_ODD/CAST_NONE</p>
</td>
</tr>
<tr id="row5534153314611"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p457718375568"><a name="p457718375568"></a><a name="p457718375568"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1657810376565"><a name="p1657810376565"></a><a name="p1657810376565"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row1353403319610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1257853716561"><a name="p1257853716561"></a><a name="p1257853716561"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p295019815598"><a name="p295019815598"></a><a name="p295019815598"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row7534183310614"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1457893765616"><a name="p1457893765616"></a><a name="p1457893765616"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2035121375918"><a name="p2035121375918"></a><a name="p2035121375918"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row185341533667"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p9578937185612"><a name="p9578937185612"></a><a name="p9578937185612"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5276172511590"><a name="p5276172511590"></a><a name="p5276172511590"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row185342332612"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8579193712564"><a name="p8579193712564"></a><a name="p8579193712564"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p165793371568"><a name="p165793371568"></a><a name="p165793371568"></a>CAST_ROUND/CAST_HYBRID</p>
</td>
</tr>
<tr id="row5534203310611"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p145791737185614"><a name="p145791737185614"></a><a name="p145791737185614"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1357963715610"><a name="p1357963715610"></a><a name="p1357963715610"></a>CAST_RINT</p>
</td>
</tr>
<tr id="row55342331068"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19579173705617"><a name="p19579173705617"></a><a name="p19579173705617"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p157963716564"><a name="p157963716564"></a><a name="p157963716564"></a>CAST_RINT</p>
</td>
</tr>
<tr id="row35356331364"><td class="cellrowborder" rowspan="8" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p17579163715562"><a name="p17579163715562"></a><a name="p17579163715562"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p1957913375569"><a name="p1957913375569"></a><a name="p1957913375569"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p17300352124"><a name="p17300352124"></a><a name="p17300352124"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1153512331764"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p557912372568"><a name="p557912372568"></a><a name="p557912372568"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p136321615909"><a name="p136321615909"></a><a name="p136321615909"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row45358331361"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11579103713563"><a name="p11579103713563"></a><a name="p11579103713563"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18580153716564"><a name="p18580153716564"></a><a name="p18580153716564"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row185358334617"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p758016375568"><a name="p758016375568"></a><a name="p758016375568"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p94691024402"><a name="p94691024402"></a><a name="p94691024402"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row75351337616"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p15801379568"><a name="p15801379568"></a><a name="p15801379568"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p658093712561"><a name="p658093712561"></a><a name="p658093712561"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row85355331967"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19580193735613"><a name="p19580193735613"></a><a name="p19580193735613"></a>int4b_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p315453414017"><a name="p315453414017"></a><a name="p315453414017"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row653520331360"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p13580133785619"><a name="p13580133785619"></a><a name="p13580133785619"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p050239008"><a name="p050239008"></a><a name="p050239008"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row205354336611"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p55809378568"><a name="p55809378568"></a><a name="p55809378568"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19580193712563"><a name="p19580193712563"></a><a name="p19580193712563"></a>CAST_ROUND/CAST_HYBRID</p>
</td>
</tr>
<tr id="row3535113311616"><td class="cellrowborder" rowspan="5" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p13580337105611"><a name="p13580337105611"></a><a name="p13580337105611"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p16580537175610"><a name="p16580537175610"></a><a name="p16580537175610"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p1721816541220"><a name="p1721816541220"></a><a name="p1721816541220"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1053673311615"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p95803378569"><a name="p95803378569"></a><a name="p95803378569"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p658003719562"><a name="p658003719562"></a><a name="p658003719562"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row1053614335610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p95801737165610"><a name="p95801737165610"></a><a name="p95801737165610"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p141137277114"><a name="p141137277114"></a><a name="p141137277114"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row1953618331363"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1158019378569"><a name="p1158019378569"></a><a name="p1158019378569"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3189527316"><a name="p3189527316"></a><a name="p3189527316"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row55361133766"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p658053755616"><a name="p658053755616"></a><a name="p658053755616"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p726211278117"><a name="p726211278117"></a><a name="p726211278117"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row55365331561"><td class="cellrowborder" rowspan="3" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p1658193705618"><a name="p1658193705618"></a><a name="p1658193705618"></a>int4b_t</p>
<p id="p41887311235"><a name="p41887311235"></a><a name="p41887311235"></a></p>
<p id="p238216423318"><a name="p238216423318"></a><a name="p238216423318"></a></p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p858173745610"><a name="p858173745610"></a><a name="p858173745610"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p146041103316"><a name="p146041103316"></a><a name="p146041103316"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row51885311639"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4188113115310"><a name="p4188113115310"></a><a name="p4188113115310"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p31881731835"><a name="p31881731835"></a><a name="p31881731835"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row83825421830"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p143823424318"><a name="p143823424318"></a><a name="p143823424318"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10382154211315"><a name="p10382154211315"></a><a name="p10382154211315"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1289474251715"><td class="cellrowborder" rowspan="3" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p12581203719562"><a name="p12581203719562"></a><a name="p12581203719562"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p258113775613"><a name="p258113775613"></a><a name="p258113775613"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p14675120933"><a name="p14675120933"></a><a name="p14675120933"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1344448171618"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6581183718566"><a name="p6581183718566"></a><a name="p6581183718566"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p27441201434"><a name="p27441201434"></a><a name="p27441201434"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row539385718163"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3581163710560"><a name="p3581163710560"></a><a name="p3581163710560"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p118141701311"><a name="p118141701311"></a><a name="p118141701311"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row989315691715"><td class="cellrowborder" rowspan="3" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p9581437185619"><a name="p9581437185619"></a><a name="p9581437185619"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p10581037175612"><a name="p10581037175612"></a><a name="p10581037175612"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p888410018312"><a name="p888410018312"></a><a name="p888410018312"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row16995144181717"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p10581183795611"><a name="p10581183795611"></a><a name="p10581183795611"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4954401431"><a name="p4954401431"></a><a name="p4954401431"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row15291246121713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p0581737105612"><a name="p0581737105612"></a><a name="p0581737105612"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p924814319"><a name="p924814319"></a><a name="p924814319"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row937004910173"><td class="cellrowborder" rowspan="2" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p1159153715566"><a name="p1159153715566"></a><a name="p1159153715566"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p175915375562"><a name="p175915375562"></a><a name="p175915375562"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p18951911435"><a name="p18951911435"></a><a name="p18951911435"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row69047476172"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5591537145617"><a name="p5591537145617"></a><a name="p5591537145617"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p31641511834"><a name="p31641511834"></a><a name="p31641511834"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row192639311712"><td class="cellrowborder" rowspan="6" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p1159163716569"><a name="p1159163716569"></a><a name="p1159163716569"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p20592103725613"><a name="p20592103725613"></a><a name="p20592103725613"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p95921837165618"><a name="p95921837165618"></a><a name="p95921837165618"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row042871161713"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1659223712562"><a name="p1659223712562"></a><a name="p1659223712562"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p05334617615"><a name="p05334617615"></a><a name="p05334617615"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row152015981619"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18592237175614"><a name="p18592237175614"></a><a name="p18592237175614"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p106111561616"><a name="p106111561616"></a><a name="p106111561616"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1964018537161"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p20592133745618"><a name="p20592133745618"></a><a name="p20592133745618"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1069176769"><a name="p1069176769"></a><a name="p1069176769"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row185481151022"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p454845119217"><a name="p454845119217"></a><a name="p454845119217"></a>int4b_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p65483511623"><a name="p65483511623"></a><a name="p65483511623"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row17680105518163"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8592163745614"><a name="p8592163745614"></a><a name="p8592163745614"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p47611061619"><a name="p47611061619"></a><a name="p47611061619"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row92250171610"><td class="cellrowborder" rowspan="3" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p5592193785614"><a name="p5592193785614"></a><a name="p5592193785614"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p859233735615"><a name="p859233735615"></a><a name="p859233735615"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p1983110619612"><a name="p1983110619612"></a><a name="p1983110619612"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row125321050181816"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1592203712566"><a name="p1592203712566"></a><a name="p1592203712566"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p129015620619"><a name="p129015620619"></a><a name="p129015620619"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row4383104918187"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p359216375563"><a name="p359216375563"></a><a name="p359216375563"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4973662618"><a name="p4973662618"></a><a name="p4973662618"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row93261648181810"><td class="cellrowborder" rowspan="6" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p659393735613"><a name="p659393735613"></a><a name="p659393735613"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p1059313720562"><a name="p1059313720562"></a><a name="p1059313720562"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p8593183775620"><a name="p8593183775620"></a><a name="p8593183775620"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row1821914721815"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p12593037135612"><a name="p12593037135612"></a><a name="p12593037135612"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7227816867"><a name="p7227816867"></a><a name="p7227816867"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row6941144521819"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1659333765615"><a name="p1659333765615"></a><a name="p1659333765615"></a>int16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1230216161964"><a name="p1230216161964"></a><a name="p1230216161964"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row157094428189"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p55931937195619"><a name="p55931937195619"></a><a name="p55931937195619"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10284143515396"><a name="p10284143515396"></a><a name="p10284143515396"></a>roundMode不生效，与<a href="SetDeqScale.md">SetDeqScale(half scale)</a>接口配合使用。</p>
</td>
</tr>
<tr id="row33073445180"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p35933372561"><a name="p35933372561"></a><a name="p35933372561"></a>uint8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13659223366"><a name="p13659223366"></a><a name="p13659223366"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row10411941181812"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p15593113715568"><a name="p15593113715568"></a><a name="p15593113715568"></a>uint16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1772912311615"><a name="p1772912311615"></a><a name="p1772912311615"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1016184013184"><td class="cellrowborder" rowspan="3" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p9593537205614"><a name="p9593537205614"></a><a name="p9593537205614"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p13593113718562"><a name="p13593113718562"></a><a name="p13593113718562"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p20797923568"><a name="p20797923568"></a><a name="p20797923568"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row555320383188"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p5594737145617"><a name="p5594737145617"></a><a name="p5594737145617"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p959493765614"><a name="p959493765614"></a><a name="p959493765614"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row571514281519"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p67166281257"><a name="p67166281257"></a><a name="p67166281257"></a>double</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15716182812516"><a name="p15716182812516"></a><a name="p15716182812516"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row619345251614"><td class="cellrowborder" rowspan="2" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p14594153712562"><a name="p14594153712562"></a><a name="p14594153712562"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p15594937155613"><a name="p15594937155613"></a><a name="p15594937155613"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p14762172815611"><a name="p14762172815611"></a><a name="p14762172815611"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1926911131715"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p859415378562"><a name="p859415378562"></a><a name="p859415378562"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p283120281869"><a name="p283120281869"></a><a name="p283120281869"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row115211850141313"><td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p9595203735619"><a name="p9595203735619"></a><a name="p9595203735619"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p95955370565"><a name="p95955370565"></a><a name="p95955370565"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p190062810613"><a name="p190062810613"></a><a name="p190062810613"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row134681038181614"><td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p4595113735613"><a name="p4595113735613"></a><a name="p4595113735613"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p11595937175619"><a name="p11595937175619"></a><a name="p11595937175619"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p1796772816617"><a name="p1796772816617"></a><a name="p1796772816617"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row1968134091612"><td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p11595143713564"><a name="p11595143713564"></a><a name="p11595143713564"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p1259543725620"><a name="p1259543725620"></a><a name="p1259543725620"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p153618291063"><a name="p153618291063"></a><a name="p153618291063"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row49372033201616"><td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p13595637185613"><a name="p13595637185613"></a><a name="p13595637185613"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p18595237125619"><a name="p18595237125619"></a><a name="p18595237125619"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p171068291368"><a name="p171068291368"></a><a name="p171068291368"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row037196104212"><td class="cellrowborder" rowspan="2" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p1375614427"><a name="p1375614427"></a><a name="p1375614427"></a>complex64</p>
<p id="p11928182064214"><a name="p11928182064214"></a><a name="p11928182064214"></a></p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p19371694216"><a name="p19371694216"></a><a name="p19371694216"></a>complex64</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p153706144219"><a name="p153706144219"></a><a name="p153706144219"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row1092882014212"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1492911203427"><a name="p1492911203427"></a><a name="p1492911203427"></a>complex32</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p892915205429"><a name="p892915205429"></a><a name="p892915205429"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC/CAST_ODD/CAST_NONE</p>
</td>
</tr>
<tr id="row8744217104211"><td class="cellrowborder" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p1174451715420"><a name="p1174451715420"></a><a name="p1174451715420"></a>complex32</p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p137442176423"><a name="p137442176423"></a><a name="p137442176423"></a>complex64</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p1474481720422"><a name="p1474481720422"></a><a name="p1474481720422"></a>CAST_NONE</p>
</td>
</tr>
<tr id="row144601341965"><td class="cellrowborder" rowspan="4" valign="top" width="14.26%" headers="mcps1.2.4.1.1 "><p id="p148054451863"><a name="p148054451863"></a><a name="p148054451863"></a>double</p>
<p id="p77624816280"><a name="p77624816280"></a><a name="p77624816280"></a></p>
<p id="p17927101817289"><a name="p17927101817289"></a><a name="p17927101817289"></a></p>
</td>
<td class="cellrowborder" valign="top" width="14.67%" headers="mcps1.2.4.1.2 "><p id="p164619341569"><a name="p164619341569"></a><a name="p164619341569"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="71.07%" headers="mcps1.2.4.1.3 "><p id="p1846153419620"><a name="p1846153419620"></a><a name="p1846153419620"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row39081039362"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p890819391665"><a name="p890819391665"></a><a name="p890819391665"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p49095395614"><a name="p49095395614"></a><a name="p49095395614"></a>CAST_RINT/CAST_FLOOR/CAST_CEIL/CAST_ROUND/CAST_TRUNC</p>
</td>
</tr>
<tr id="row876238152811"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p37621882818"><a name="p37621882818"></a><a name="p37621882818"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13762686282"><a name="p13762686282"></a><a name="p13762686282"></a>CAST_TRUNC</p>
</td>
</tr>
<tr id="row109271118102819"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p17927518102816"><a name="p17927518102816"></a><a name="p17927518102816"></a>int64_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13927151815283"><a name="p13927151815283"></a><a name="p13927151815283"></a>CAST_TRUNC</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。特别地，对于长度较小的数据类型转换为长度较大的数据类型时，地址重叠可能会导致结果错误。

-   每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float-\>half转换每次迭代操作64个源/目的元素。
-   当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32\_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
-   当dst或src为int4b\_t时，由于一个int4b\_t只占半个字节，故申请Tensor空间时，只需申请相同数量的int8\_t数据空间的一半。host侧目前暂不支持int4b\_t，故在申请int4b\_t类型的tensor时，应先申请一个类型为int8\_t的tensor，再用Reinterpretcast转化为int4b\_t并调用Cast指令，详见调用示例。
-   当dst或src为int4b\_t时，tensor高维切分计算接口的连续模式的mask与tensor前n个数据计算接口的count必须为偶数；对于tensor高维切分计算接口的逐bit模式，对应同一字节的相邻两个比特位的数值必须一致，即0-1位数值一致，2-3位数值一致，4-5位数值一致，以此类推。
-   针对Ascend 950PR/Ascend 950DT，complex32/complex64/double数据类型仅支持tensor前n个数据计算接口。

## 调用示例<a name="section642mcpsimp"></a>

本样例中只展示Compute流程中的部分代码。本样例的srcLocal为half类型，dstLocal为int32\_t类型，计算mask时以int32\_t为准。

如果您需要运行样例代码，完整的调用样例请参考[Cast样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/00_math/cast)。

根据不同的RoundMode取值，输出结果会有差异，下面样例以RoundMode::CAST\_CEIL（向正无穷舍入）为例。

-   tensor高维切分计算样例-mask连续模式

    ```
    uint64_t mask = 256 / sizeof(int32_t);
    // repeatTime = 8, 64 elements one repeat, 512 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, srcRepStride = 4, no gap between repeats
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, mask, 8, { 1, 1, 8, 4 });
    ```

-   tensor高维切分计算样例-mask逐bit模式

    ```
    uint64_t mask[2] = { 0, UINT64_MAX };
    // repeatTime = 8, 64 elements one repeat, 512 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 8, srcRepStride = 4, no gap between repeats
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, mask, 8, { 1, 1, 8, 4 });
    ```

-   tensor前n个数据计算样例

    ```
    uint32_t count = 512; // 参与计算的元素个数
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, count);
    ```

结果示例如下：

```
输入数据(srcLocal): 
[1.4, 1.5, 1.6, 2.4, 2.5, 2.6, ... 2.6]
  ]
输出数据(dstLocal): 
[2, 2, 2, 3, 3, 3, ... 3]
```

当RoundMode为RoundMode::CAST\_NONE（half转int32\_t有精度损失，此时同CAST\_RINT模式）或RoundMode::CAST\_RINT（四舍六入五成双舍入）时，结果示例如下：

```
输入数据(srcLocal): 
[1.4, 1.5, 1.6, 2.4, 2.5, 2.6, ... 2.6]
输出数据(dstLocal): 
[1, 2, 2, 2, 2, 3, ... 3]
```

当RoundMode为RoundMode::CAST\_FLOOR（向负无穷舍入）时，结果示例如下：

```
输入数据(srcLocal): 
[1.4, 1.5, 1.6, 2.4, 2.5, 2.6, ... 2.6]
输出数据(dstLocal): 
[1, 1, 1, 2, 2, 2, ... 2]
```

当RoundMode为RoundMode::CAST\_ROUND（四舍五入舍入）时，结果示例如下：

```
输入数据(srcLocal): 
[1.4, 1.5, 1.6, 2.4, 2.5, 2.6, ... 2.6]
输出数据(dstLocal): 
[1, 2, 2, 2, 3, 3, ... 3]
```

当RoundMode为RoundMode::CAST\_TRUNC（向零舍入）时，结果示例如下：

```
输入数据(srcLocal): 
[1.4, 1.5, 1.6, 2.4, 2.5, 2.6, ... 2.6]
输出数据(dstLocal): 
[1, 1, 1, 2, 2, 2, ... 2]
```

-   当Cast涉及int4b\_t时，调用示例如下：

    dstLocal为int8\_t类型，srcLocal为half类型

    ```
    inBufferSize_ = srcSize;  // src buffer size
    outBufferSize_ = srcSize / 2;   //dst buffer size
    uint64_t mask = 128;
    AscendC::LocalTensor<half> srcLocal;
    srcLocal.SetSize(inBufferSize_);
    AscendC::LocalTensor<int8_t> dstLocal;
    dstLocal.SetSize(outBufferSize_);
    AscendC::LocalTensor<AscendC::int4b_t> dstLocalTmp = dstLocal.ReinterpretCast<AscendC::int4b_t>();
    // repeatTime = 1, 128 elements one repeat, 128 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride = 2, srcRepStride = 8, no gap between repeats
    AscendC::Cast<AscendC::int4b_t, half>(dstLocalTmp, srcLocal, AscendC::RoundMode::CAST_CEIL, mask, 1, {1, 1, 2, 8});
    ```

## 更多样例<a name="section149016158516"></a>

更多地，您可以参考以下样例，了解如何使用Cast指令的tensor高维切分计算接口，进行更灵活的操作、实现更高级的功能。

-   通过tensor高维切分计算接口中的mask连续模式，实现数据非连续计算。

    ```
    uint64_t mask = 32;  // 每个迭代内只计算前32个数
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, mask, 8, { 1, 1, 8, 4 });
    ```

    结果示例如下：

    ```
    输入数据(srcLocal): 
    [37.4     7.11   53.5    19.44   22.66   43.     43.16    5.316  74.2
     15.7    87.75   86.94   92.56   25.45   36.06   94.6    73.6    30.48
     48.16   12.55   27.81   14.67    6.58   48.38   67.5    57.5    63.3
     85.2     3.654  68.7    52.53   16.38   13.945  63.84   87.2    82.5
     85.7    27.78   15.41   41.66   31.38   14.65   88.25    0.0332 43.06
     46.88   15.57   87.1    53.16   33.5    91.06   36.5    55.34   60.53
      3.238  23.92   97.5    91.1    78.44   54.47   82.     53.8    72.1
     25.06   32.12   15.88   33.38   36.7    33.3    84.4    19.25    1.743
     46.16   22.06    4.582  71.1    15.94   22.23   53.47   17.05   48.56
     94.44   77.4    90.2    46.56   92.4     9.45   68.44   35.7    31.62
     68.1    63.7    77.     92.06   20.45   27.67   93.4    22.39   17.22
     73.06    7.12   25.34   36.34   13.54   38.12   24.56   86.56   69.7
     68.3    30.38   68.4    86.1    54.44   70.     55.3    48.6    59.03
     64.44   15.45   66.5    92.7    60.7    52.22   47.     99.75   41.94
     43.06   89.5    36.9    62.5     1.306  48.06    9.37   62.25   20.61
     43.8    69.25   27.22   71.44   52.75   11.82   80.6    63.44   53.22
     85.44   25.25    2.309  26.88   84.5    29.83    9.93   81.9    97.75
     75.75   97.7    72.     19.86   26.62   88.7    74.06    9.24   42.5
     14.     39.44   98.56   66.94   89.     57.12   39.     11.57   19.05
     86.56   32.66   19.25   99.3    95.6    58.7    79.6    37.38   65.
     75.7     8.586  77.7     2.68   75.7    77.56   39.1    39.72   64.06
     98.44   30.27   31.9    94.4    85.94    4.965   2.758  92.4    49.53
     50.75    5.7    19.69   87.6    20.08   88.8    87.4    63.6    68.3
     78.9    45.66   10.01   35.25   71.9    37.38   39.7    43.47   11.67
     64.3    35.62   74.3    59.3    28.69   29.56   23.14   36.22    4.88
     70.5    25.05   72.6    71.6    32.28   34.66   80.     96.1    98.7
     12.91   95.4    61.97   87.94   19.1    40.47   89.6    84.     29.72
     17.8    81.44   23.25   33.03   18.67   78.     49.62   63.1    72.75
     77.25    3.74   38.9    17.92   76.     25.62   34.53   84.     32.03
     57.3     9.21    6.836  68.9    35.78   96.75   56.3    96.1    23.45
     78.75   94.25   12.44   56.7    24.55   25.11   90.7    50.94   78.4
      3.576  21.81   53.28   26.2    43.1     7.742  13.4    86.44   86.9
     13.93   16.48   91.06   42.3    95.5    66.8    40.6    98.06   71.9
     67.6    55.9    82.44   93.75   41.53   23.62   40.12   40.53   80.7
     80.25   96.3    51.38   93.6    91.3    32.84   88.     69.7    63.16
     41.75   43.22   43.22   31.73   84.9    91.6    80.     53.34   27.12
     76.6    97.25   44.5    30.28   74.3    76.06   40.     41.28   37.72
     99.56   18.73   16.45   92.75   79.1    40.3    68.     23.98   88.7
     86.6    24.97   59.6    28.25   82.94   46.12   60.12   34.53   79.7
     11.086  20.25   44.88   39.97   42.12   62.7    30.66   42.56   16.69
     85.2    90.8    78.75   26.16   18.14   94.06   40.3    20.16   38.
     12.99   95.44   76.25   26.03   76.     30.06   27.25   84.56   30.45
     66.1    83.25    3.732  39.1    54.22   82.8    43.22   53.03   11.66
     88.1     6.83   66.8    44.4     7.5    24.77   74.4    35.9    79.75
     41.62   37.06   60.12   57.9    96.94   84.25   39.88   22.55   72.7
     58.9    44.75   90.4    46.34   71.3    16.4    26.12   21.45   10.27
     91.     41.53   39.03   80.25    2.11    7.88   72.2    27.83   88.1
     67.56   10.72   52.84   91.2    97.6    51.44   74.7     3.527  79.25
     11.3    19.16   39.53    3.469  98.7    45.72   40.16   47.1    71.8
     11.81   52.97   71.44   37.7    26.81   46.22   26.94    4.805  12.18
     70.4    51.4    24.2    83.9     9.62   12.445  57.6    85.8    55.12
     88.25   32.38   62.88    1.903  47.72   35.9    48.94   86.06   32.44
      1.219  35.56   49.78   49.97   24.45   94.5    99.94   44.72    3.404
     83.6    23.14   76.7    91.7    24.33   20.62   24.72    4.55   88.94
     87.44   95.75   41.56   13.77   34.6    95.94   77.1    24.28   70.06
     10.06   11.38   88.8    57.22   94.56   35.     79.8    58.22   44.06
     26.9    16.25   99.94   51.1    42.38   84.25    0.9604 48.1   ]
    
    输出数据(dstLocal): 
    [        38          8         54         20         23         43
             44          6         75         16         88         87
             93         26         37         95         74         31
             49         13         28         15          7         49
             68         58         64         86          4         69
             53         17 1879993057 1827499998 1823960025 1570990114
     1828150463 1811639312 1794470101 1754296176 1888841335 1715628997
     1839753994 1850888497 1889364175 1891068936 1823369913 1769105534
     1815638091 1808559970 1601662785 1739089473 1863146361 1694785989
     1597138938 1836478181 1888774249 1637707434 1877372650 1796304934
     1887530885 1839295471 1707240971 1873242695         33         16
             34         37         34         85         20          2
             47         23          5         72         16         23
             54         18         49         95         78         91
             47         93         10         69         36         32
             69         64         77         93         21         28
     1753837732 1488743807 1711632378 1799581711 1818783215 1891790695
     1837723802 1752132873 1727950918 1760390205 1866887130 1824876865
     1807839436 1890544910 1889755550 1787129270 1502702106 1841065201
     1820156583 1779396288 1760521448 1844604520 1831039103 1843491014
     1891199259 1839493317 1801349958 1577807434 1811377215 1879404734
     1826057367 1837853054         37         63          2         49
             10         63         21         44         70         28
             72         53         12         81         64         54
             86         26          3         27         85         30
             10         82         98         76         98         72
             20         27         89         75 1890086927 1826517134
     1814783944 1824156809 1875733079 1842114682 1845456975 1830120794
     1787980861 1807380585 1535469972 1883860884 1889167601 1747872128
     1888317235 1720937006 1836806331 1654152236 1695309475 1892773593
     1840737395 1868392748 1833724316 1600153936 1869310159 1883467778
     1892641857 1776248953 1833201514 1886743848 1745972258 1860657622
             95         86          5          3         93         50
             51          6         20         88         21         89
             88         64         69         79         46         11
             36         72         38         40         44         12
             65         36         75         60         29         30
             24         37 1733652589 1756325317 1685744372 1780772214
     1660252348 1629973784 1847815925 1828941229 1683778661 1519480967
     1762160488 1844801381 1832742021 1891724641 1761701480 1695312651
     1429433841 1774275423 1828349211 1779786303 1835953259 1784896595
     1858432988 1413442268 1893363867 1886679050 1872588913 1473866635
     1793158916 1762946052 1719627087 1893231666         76         26
             35         84         33         58         10          7
             69         36         97         57         97         24
             79         95         13         57         25         26
             91         51         79          4         22         54
             27         44          8         14         87         87
     1208960410 1208567888 1215973275 1214859418 1210992732 1208305714
     1165379704 1215252623 1197033625 1212172318 1217415148 1211058280
     1206798479 1215645776 1209223143 1217611809 1212500082 1215383615
     1208567769 1200179260 1216694386 1218070398 1195526187 1211516213
     1213089850 1213941743 1216825148 1212565573 1216694248 1217546087
     1200178778 1215973524         92         80         54         28
             77         98         45         31         75         77
             40         42         38        100         19         17
             93         80         41         68         24         89
             87         25         60         29         83         47
             61         35         80         12 1189759014 1218070662
     1211057953 1216825400 1215121414 1214596952 1216169420 1210075102
     1209157633 1213941279 1195984973 1211648118 1201686666 1212041360
     1216103688 1212500024 1173112887 1194608648 1216825427 1209747582
     1207191587 1214859224 1203980407 1215711379 1213155353 1203259396
     1214859350 1211779156 1217218713 1202473003 1216628529 1196771437
             44         54         12         89          7         67
             45          8         25         75         36         80
             42         38         61         58         97         85
             40         23         73         59         45         91
             47         72         17         27         22         11
             91         42 1211516990 1198213215 1203848735 1217349746
     1212303398 1217808150 1215514752 1209878647 1214138433 1215711277
     1212041273 1215383541 1214728294 1197754500 1169574019 1208371214
     1214269569 1216301092 1216563283 1213548677 1217873970 1203128459
     1209812695 1218136029 1194805359 1204439186 1218005120 1213941626
     1217153046 1208109091 1215055928 1215318166          5         13
             71         52         25         84         10         13
             58         86         56         89         33         63
              2         48         36         49         87         33
              2         36         50         50         25         95
            100         45          4         84         24         77
     1202210969 1213679767 1209288847 1217480263 1184319410 1214072674
     1188382819 1217283870 1200769107 1217939416 1199327294 1213351841
     1206667407 1217153163 1215580283 1214138474 1206798167 1194477696
     1193690499 1214072706 1216825421 1216693888 1217611496 1198540949
     1199654414 1206405188 1214203847 1165183076 1213745302 1208830102
     1209944118 1215121459]
    ```

