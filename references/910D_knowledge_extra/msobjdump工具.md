# msobjdump工具<a name="ZH-CN_TOPIC_0000002554351519"></a>

本工具主要针对[Kernel直调工程](基于样例工程完成Kernel直调.md)（NPU模式）、[标准自定义算子工程](工程化算子开发.md)编译生成的算子ELF文件（Executable and Linkable Format）提供解析和解压功能，并将结果信息以可读形式呈现，方便开发者直观获得kernel文件信息。

> **说明：** 
>-   ELF文件是一种用于二进制文件、可执行文件、目标代码、共享库和核心转储的文件格式，包括常见的\*.a、\*.so文件等。ELF文件常见构成如下：
>    -   ELF头部：描述了整个文件的组织结构，包括文件类型、机器类型、版本号等信息。
>    -   程序头部表：描述了文件中各种段（segments）信息，包括程序如何加载到内存中执行的信息。
>    -   节区头部表：描述了文件中各个节（sections）信息，包括程序的代码、数据、符号表等。
>-   工具使用过程中，若出现如下场景，请根据日志提示信息，分析排查问题。
>    -   ELF文件未找到
>    -   ELF文件权限错误
>    -   ELF文件存在但不支持解析或解压

## 产品支持情况<a name="section15615918103719"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1613032013400"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p11131020174013"><a name="p11131020174013"></a><a name="p11131020174013"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p107841459102"><a name="p107841459102"></a><a name="p107841459102"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 工具安装<a name="section2423188195113"></a>

1.  安装msobjdump工具。

    工具跟随CANN软件包发布（参考[环境准备](环境准备.md)完成CANN安装），其路径默认为“$\{INSTALL\_DIR\}/tools/msobjdump”，其中$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

2.  设置环境变量。

    请以CANN软件包运行用户，执行如下命令使公共环境变量生效：

    ```
    source ${INSTALL_DIR}/bin/setenv.bash
    ```

3.  检查工具是否安装成功。

    执行如下命令，若能正常显示--help或-h信息，则表示工具环境正常，功能可正常使用。

    ```
    msobjdump -h
    ```

## 命令格式<a name="section101202511916"></a>

-   **解析ELF文件的命令**

    ```
    msobjdump --dump-elf <elf_file> [--verbose]
    ```

    **表 1**  参数说明

    <a name="table167911947163519"></a>
    <table><thead align="left"><tr id="row197911147103513"><th class="cellrowborder" valign="top" width="21.55%" id="mcps1.2.4.1.1"><p id="p9791164717352"><a name="p9791164717352"></a><a name="p9791164717352"></a>参数（区分大小写）</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.61%" id="mcps1.2.4.1.2"><p id="p1579144713356"><a name="p1579144713356"></a><a name="p1579144713356"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="67.84%" id="mcps1.2.4.1.3"><p id="p17791174773515"><a name="p17791174773515"></a><a name="p17791174773515"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row27911447183511"><td class="cellrowborder" valign="top" width="21.55%" headers="mcps1.2.4.1.1 "><p id="p5791547143516"><a name="p5791547143516"></a><a name="p5791547143516"></a>--dump-elf &lt;elf_file&gt;，-d</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p11791194718355"><a name="p11791194718355"></a><a name="p11791194718355"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="67.84%" headers="mcps1.2.4.1.3 "><p id="p118129396375"><a name="p118129396375"></a><a name="p118129396375"></a>解析ELF文件中包含的device信息，如文件名、文件类型、文件长度、符号表等，并终端打屏显示。</p>
    <p id="p19791247193518"><a name="p19791247193518"></a><a name="p19791247193518"></a>&lt;elf_file&gt;表示待解析ELF文件路径，如/home/op_api/lib_api.so。支持两种打印模式：</p>
    <p id="p64681643111119"><a name="p64681643111119"></a><a name="p64681643111119"></a>简单打印：默认仅打印部分device信息。</p>
    <p id="p17614154581112"><a name="p17614154581112"></a><a name="p17614154581112"></a>全量打印：与--verbose配套使用，开启全量device信息打屏显示。</p>
    <p id="p579144753511"><a name="p579144753511"></a><a name="p579144753511"></a>不同工程打印字段信息不同，具体参见<a href="#table217334916136">表4</a>和<a href="#table94384560259">表5</a>。</p>
    </td>
    </tr>
    <tr id="row279124793510"><td class="cellrowborder" valign="top" width="21.55%" headers="mcps1.2.4.1.1 "><p id="p67919475352"><a name="p67919475352"></a><a name="p67919475352"></a>--verbose，-V</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p1279144713515"><a name="p1279144713515"></a><a name="p1279144713515"></a>可选</p>
    </td>
    <td class="cellrowborder" valign="top" width="67.84%" headers="mcps1.2.4.1.3 "><p id="p479114723520"><a name="p479114723520"></a><a name="p479114723520"></a>必须与--dump-elf配套使用，用于开启ELF文件中全量打印device信息功能。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   **解压ELF文件的命令**

    ```
    msobjdump --extract-elf <elf_file> [--out-dir <out_path>]
    ```

    **表 2**  参数说明

    <a name="table131531242133819"></a>
    <table><thead align="left"><tr id="row1615374218387"><th class="cellrowborder" valign="top" width="21.55%" id="mcps1.2.4.1.1"><p id="p1915318421386"><a name="p1915318421386"></a><a name="p1915318421386"></a>参数（区分大小写）</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.61%" id="mcps1.2.4.1.2"><p id="p715374203810"><a name="p715374203810"></a><a name="p715374203810"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="67.84%" id="mcps1.2.4.1.3"><p id="p6153144212387"><a name="p6153144212387"></a><a name="p6153144212387"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1015344210387"><td class="cellrowborder" valign="top" width="21.55%" headers="mcps1.2.4.1.1 "><p id="p10913718143914"><a name="p10913718143914"></a><a name="p10913718143914"></a>--extract-elf &lt;elf_file&gt;，-e</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p109131318103919"><a name="p109131318103919"></a><a name="p109131318103919"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="67.84%" headers="mcps1.2.4.1.3 "><p id="p15333744174112"><a name="p15333744174112"></a><a name="p15333744174112"></a>解压ELF文件中包含的device信息，并按原始文件夹规则落盘到输出路径下。</p>
    <p id="p191311188393"><a name="p191311188393"></a><a name="p191311188393"></a>&lt;elf_file&gt;表示待解压ELF文件路径，如/home/op_api/lib_api.so。</p>
    <p id="p149930262154"><a name="p149930262154"></a><a name="p149930262154"></a>默认路径：解压结果文件默认落盘到当前执行路径下。</p>
    <p id="p1732142921514"><a name="p1732142921514"></a><a name="p1732142921514"></a>自定义路径：可与--out-dir配套使用，设置落盘路径。</p>
    </td>
    </tr>
    <tr id="row141530427380"><td class="cellrowborder" valign="top" width="21.55%" headers="mcps1.2.4.1.1 "><p id="p7386182813394"><a name="p7386182813394"></a><a name="p7386182813394"></a>--out-dir &lt;out_path&gt;，-o</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.61%" headers="mcps1.2.4.1.2 "><p id="p183861428143912"><a name="p183861428143912"></a><a name="p183861428143912"></a>可选</p>
    </td>
    <td class="cellrowborder" valign="top" width="67.84%" headers="mcps1.2.4.1.3 "><p id="p12905746174119"><a name="p12905746174119"></a><a name="p12905746174119"></a>必须与--extract-elf配套使用，用于设置解压文件的落盘路径。</p>
    <p id="p738620282392"><a name="p738620282392"></a><a name="p738620282392"></a>&lt;out_path&gt;为落盘文件目录，如/home/extract/。</p>
    <p id="p193582470159"><a name="p193582470159"></a><a name="p193582470159"></a><strong id="b4489207161611"><a name="b4489207161611"></a><a name="b4489207161611"></a>请注意</strong>：<span id="ph238682811391"><a name="ph238682811391"></a><a name="ph238682811391"></a>msobjdump</span>支持多用户并发调用，但需要指定不同的--out-dir，否则可能出现落盘内容被覆盖的问题。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   **获取ELF文件列表的命令**

    ```
    msobjdump --list-elf <elf_file>
    ```

    **表 3**  参数说明

    <a name="table121952819427"></a>
    <table><thead align="left"><tr id="row31956804214"><th class="cellrowborder" valign="top" width="21.55%" id="mcps1.2.4.1.1"><p id="p3195780420"><a name="p3195780420"></a><a name="p3195780420"></a>参数（区分大小写）</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.6%" id="mcps1.2.4.1.2"><p id="p419538154218"><a name="p419538154218"></a><a name="p419538154218"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="67.85%" id="mcps1.2.4.1.3"><p id="p2019548124212"><a name="p2019548124212"></a><a name="p2019548124212"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row121956817426"><td class="cellrowborder" valign="top" width="21.55%" headers="mcps1.2.4.1.1 "><p id="p773202810266"><a name="p773202810266"></a><a name="p773202810266"></a>--list-elf &lt;elf_file&gt;，-l</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.6%" headers="mcps1.2.4.1.2 "><p id="p1924432892716"><a name="p1924432892716"></a><a name="p1924432892716"></a>可选</p>
    </td>
    <td class="cellrowborder" valign="top" width="67.85%" headers="mcps1.2.4.1.3 "><p id="p0575145884218"><a name="p0575145884218"></a><a name="p0575145884218"></a>获取ELF文件中包含的device信息文件列表，并打印显示。</p>
    <p id="p188333261318"><a name="p188333261318"></a><a name="p188333261318"></a>&lt;elf_file&gt;表示待打印的ELF文件路径，如/home/op_api/lib_api.so。</p>
    </td>
    </tr>
    </tbody>
    </table>

**表 4**  ELF解析字段说明（Kernel直调工程）

<a name="table217334916136"></a>
<table><thead align="left"><tr id="row1617317493135"><th class="cellrowborder" valign="top" width="14.81%" id="mcps1.2.5.1.1"><p id="p1173149191319"><a name="p1173149191319"></a><a name="p1173149191319"></a>字段名</p>
</th>
<th class="cellrowborder" valign="top" width="59.650000000000006%" id="mcps1.2.5.1.2"><p id="p17173134921310"><a name="p17173134921310"></a><a name="p17173134921310"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="10.09%" id="mcps1.2.5.1.3"><p id="p46901769546"><a name="p46901769546"></a><a name="p46901769546"></a>是否必选</p>
</th>
<th class="cellrowborder" valign="top" width="15.45%" id="mcps1.2.5.1.4"><p id="p177910207517"><a name="p177910207517"></a><a name="p177910207517"></a>打印说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2017354971315"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p19173124971313"><a name="p19173124971313"></a><a name="p19173124971313"></a>VERSION</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p917313493136"><a name="p917313493136"></a><a name="p917313493136"></a>表示版本号。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p14191216135410"><a name="p14191216135410"></a><a name="p14191216135410"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p179631115438"><a name="p179631115438"></a><a name="p179631115438"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row41733498133"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p4173164916139"><a name="p4173164916139"></a><a name="p4173164916139"></a>TYPE COUNT</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p1717334914135"><a name="p1717334914135"></a><a name="p1717334914135"></a>表示ELF文件中包含的kernel文件个数。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p9689144181715"><a name="p9689144181715"></a><a name="p9689144181715"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p14408133714188"><a name="p14408133714188"></a><a name="p14408133714188"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row11731495131"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p017454931319"><a name="p017454931319"></a><a name="p017454931319"></a>ELF FILE ${id}</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p16364114722515"><a name="p16364114722515"></a><a name="p16364114722515"></a>表示ELF文件中包含的kernel文件名，${id}表示kernel文件序号。</p>
<p id="p1762892235"><a name="p1762892235"></a><a name="p1762892235"></a>kernel文件名的命名规则如下：</p>
<p id="p355919307228"><a name="p355919307228"></a><a name="p355919307228"></a>按${sec_prefix}_${file_index}_${kernel_type}.o拼接，其中${sec_prefix}为section段名（工具根据“.ascend.kernel”关键字搜索获取），${file_index}表示文件编号，${kernel_type}表示kernel类型。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p156891541111714"><a name="p156891541111714"></a><a name="p156891541111714"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p13408163721811"><a name="p13408163721811"></a><a name="p13408163721811"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row19805154016110"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p1174449101315"><a name="p1174449101315"></a><a name="p1174449101315"></a>KERNEL LEN</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p10174949131318"><a name="p10174949131318"></a><a name="p10174949131318"></a>表示kernel文件的长度。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p186897419175"><a name="p186897419175"></a><a name="p186897419175"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p24081337201813"><a name="p24081337201813"></a><a name="p24081337201813"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row21746493138"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p1017410499135"><a name="p1017410499135"></a><a name="p1017410499135"></a>KERNEL TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p13174134917135"><a name="p13174134917135"></a><a name="p13174134917135"></a>表示kernel类型，映射关系为{0 : 'mix', 1: 'aiv', 2: 'aic'}。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p66908655414"><a name="p66908655414"></a><a name="p66908655414"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p1940818375182"><a name="p1940818375182"></a><a name="p1940818375182"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row125721859556"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p157217595516"><a name="p157217595516"></a><a name="p157217595516"></a>ASCEND META</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p103911055132614"><a name="p103911055132614"></a><a name="p103911055132614"></a><span>表</span><span>示算子执行时核间同步、Cube/Vector核占比（task_ration）等信息。</span></p>
<p id="p95729511558"><a name="p95729511558"></a><a name="p95729511558"></a><span>若没有获取到该信息，默认显示None。</span></p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p127833512177"><a name="p127833512177"></a><a name="p127833512177"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p94081376185"><a name="p94081376185"></a><a name="p94081376185"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row11282530101612"><td class="cellrowborder" valign="top" width="14.81%" headers="mcps1.2.5.1.1 "><p id="p628217303160"><a name="p628217303160"></a><a name="p628217303160"></a>elf heard infos</p>
</td>
<td class="cellrowborder" valign="top" width="59.650000000000006%" headers="mcps1.2.5.1.2 "><p id="p10282230201615"><a name="p10282230201615"></a><a name="p10282230201615"></a>包括ELF Header、Section Headers、Key to Flags、Program Headers、Symbol表等信息。</p>
</td>
<td class="cellrowborder" valign="top" width="10.09%" headers="mcps1.2.5.1.3 "><p id="p13690136205410"><a name="p13690136205410"></a><a name="p13690136205410"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.45%" headers="mcps1.2.5.1.4 "><p id="p710931893616"><a name="p710931893616"></a><a name="p710931893616"></a>设置--verbose，开启全量打印。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  ELF解析字段说明（标准/简易自定义算子工程）

<a name="table94384560259"></a>
<table><thead align="left"><tr id="row20438165617250"><th class="cellrowborder" valign="top" width="14.829999999999998%" id="mcps1.2.5.1.1"><p id="p13438756132511"><a name="p13438756132511"></a><a name="p13438756132511"></a>字段名</p>
</th>
<th class="cellrowborder" valign="top" width="59.78%" id="mcps1.2.5.1.2"><p id="p64391956162519"><a name="p64391956162519"></a><a name="p64391956162519"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="10.059999999999999%" id="mcps1.2.5.1.3"><p id="p1843965614253"><a name="p1843965614253"></a><a name="p1843965614253"></a>是否必选</p>
</th>
<th class="cellrowborder" valign="top" width="15.329999999999998%" id="mcps1.2.5.1.4"><p id="p198832518815"><a name="p198832518815"></a><a name="p198832518815"></a>打印说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1779591520273"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p37951515112710"><a name="p37951515112710"></a><a name="p37951515112710"></a>.ascend.meta. ${id}</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p10784139152813"><a name="p10784139152813"></a><a name="p10784139152813"></a>表示算子kernel函数名称，其中${id}表示meta信息的索引值。</p>
<p id="p1881212571287"><a name="p1881212571287"></a><a name="p1881212571287"></a></p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p187951615122715"><a name="p187951615122715"></a><a name="p187951615122715"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1103144911911"><a name="p1103144911911"></a><a name="p1103144911911"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row33021941124717"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p5303141174713"><a name="p5303141174713"></a><a name="p5303141174713"></a>VERSION</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p9303174113471"><a name="p9303174113471"></a><a name="p9303174113471"></a>表示版本号。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p123039415475"><a name="p123039415475"></a><a name="p123039415475"></a>是</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p13520951132011"><a name="p13520951132011"></a><a name="p13520951132011"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row25665496471"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p25664491476"><a name="p25664491476"></a><a name="p25664491476"></a>DEBUG</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p829485716114"><a name="p829485716114"></a><a name="p829485716114"></a>调试相关信息，包含如下两部分内容：</p>
<p id="p2623758131917"><a name="p2623758131917"></a><a name="p2623758131917"></a>debugBufSize：调试信息需要的内存空间。</p>
<p id="p690014142017"><a name="p690014142017"></a><a name="p690014142017"></a>debugOptions：调试开关状态。取值如下：</p>
<p id="p27273138225"><a name="p27273138225"></a><a name="p27273138225"></a>0：调试开关关闭。</p>
<p id="p6424181516223"><a name="p6424181516223"></a><a name="p6424181516223"></a>1：通过DumpTensor、printf打印进行调试。</p>
<p id="p21028197223"><a name="p21028197223"></a><a name="p21028197223"></a>2：通过assert断言进行调试。</p>
<p id="p1951519203229"><a name="p1951519203229"></a><a name="p1951519203229"></a>4：通过时间戳打点功能进行调试。</p>
<p id="p19684112214225"><a name="p19684112214225"></a><a name="p19684112214225"></a>8：通过内存越界检测进行调试。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p556634974719"><a name="p556634974719"></a><a name="p556634974719"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1352113511206"><a name="p1352113511206"></a><a name="p1352113511206"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row99861010174915"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p7986910174915"><a name="p7986910174915"></a><a name="p7986910174915"></a>DYNAMIC_PARAM</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p89862107499"><a name="p89862107499"></a><a name="p89862107499"></a>算子kernel函数是否启用动态参数。取值分别为：</p>
<p id="p183854517206"><a name="p183854517206"></a><a name="p183854517206"></a>0：关闭动态参数模式。</p>
<p id="p18211156192014"><a name="p18211156192014"></a><a name="p18211156192014"></a>1：开启动态参数模式。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p7207193592015"><a name="p7207193592015"></a><a name="p7207193592015"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1652115512201"><a name="p1652115512201"></a><a name="p1652115512201"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row1185219133492"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p12852151374915"><a name="p12852151374915"></a><a name="p12852151374915"></a>OPTIONAL_PARAM</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p124401044141815"><a name="p124401044141815"></a><a name="p124401044141815"></a>可选参数信息，包含如下两部分内容：</p>
<p id="p16984917208"><a name="p16984917208"></a><a name="p16984917208"></a>optionalInputMode：可选输入在算子kernel函数中是否需要占位。取值分别为：</p>
<p id="p2830112792217"><a name="p2830112792217"></a><a name="p2830112792217"></a>0：可选输入不占位。</p>
<p id="p5781122892215"><a name="p5781122892215"></a><a name="p5781122892215"></a>1：可选输入占位。</p>
<p id="p2931710122014"><a name="p2931710122014"></a><a name="p2931710122014"></a>optionalOutputMode：可选输出在算子kernel函数中是否需要占位。取值分别为：</p>
<p id="p6728102992217"><a name="p6728102992217"></a><a name="p6728102992217"></a>0：可选输出不占位。</p>
<p id="p16331230162214"><a name="p16331230162214"></a><a name="p16331230162214"></a>1：可选输出占位。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p220723572014"><a name="p220723572014"></a><a name="p220723572014"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p752145115209"><a name="p752145115209"></a><a name="p752145115209"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row943912561252"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p1658134962616"><a name="p1658134962616"></a><a name="p1658134962616"></a>KERNEL_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p175824962614"><a name="p175824962614"></a><a name="p175824962614"></a>表示kernel函数运行时core类型，取值参见表<a href="#table187419221164">表6</a>。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p1220717357204"><a name="p1220717357204"></a><a name="p1220717357204"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1252155172014"><a name="p1252155172014"></a><a name="p1252155172014"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row3439356122520"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p1958449172618"><a name="p1958449172618"></a><a name="p1958449172618"></a>CROSS_CORE_SYNC</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p195817494264"><a name="p195817494264"></a><a name="p195817494264"></a>表示硬同步syncall类型。</p>
<p id="p151691812205"><a name="p151691812205"></a><a name="p151691812205"></a>USE_SYNC：使用硬同步。</p>
<p id="p19922151802014"><a name="p19922151802014"></a><a name="p19922151802014"></a>NO_USE_SYNC：不使用硬同步。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p152085356206"><a name="p152085356206"></a><a name="p152085356206"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p352135112012"><a name="p352135112012"></a><a name="p352135112012"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row114391456102511"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p19582494261"><a name="p19582494261"></a><a name="p19582494261"></a>MIX_TASK_RATION</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p8581349162620"><a name="p8581349162620"></a><a name="p8581349162620"></a>表示kernel函数运行时的Cube核/Vector核占比分配类型。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p320823582017"><a name="p320823582017"></a><a name="p320823582017"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1752116511207"><a name="p1752116511207"></a><a name="p1752116511207"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row122551449185211"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p172559497524"><a name="p172559497524"></a><a name="p172559497524"></a>DETERMINISTIC_INFO</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p1716512173562"><a name="p1716512173562"></a><a name="p1716512173562"></a>表示算子是否为确定性计算。</p>
<p id="p4133192032017"><a name="p4133192032017"></a><a name="p4133192032017"></a>0：不确定计算。</p>
<p id="p577919204207"><a name="p577919204207"></a><a name="p577919204207"></a>1：确定性计算。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p7208203522016"><a name="p7208203522016"></a><a name="p7208203522016"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p95216517208"><a name="p95216517208"></a><a name="p95216517208"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row19311145495216"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p8312185475217"><a name="p8312185475217"></a><a name="p8312185475217"></a>BLOCK_NUM</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p16312554185215"><a name="p16312554185215"></a><a name="p16312554185215"></a>表示算子执行核数，该字段当前暂不支持，只打印默认值0xFFFFFFFF。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p182081535172016"><a name="p182081535172016"></a><a name="p182081535172016"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p145216515208"><a name="p145216515208"></a><a name="p145216515208"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row14296111534"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p9430414531"><a name="p9430414531"></a><a name="p9430414531"></a>FUNCTION_ENTRY</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p17430019534"><a name="p17430019534"></a><a name="p17430019534"></a>算子TilingKey的值。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p1620833552011"><a name="p1620833552011"></a><a name="p1620833552011"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p1152119515203"><a name="p1152119515203"></a><a name="p1152119515203"></a>不设置--verbose，默认打印。</p>
</td>
</tr>
<tr id="row134396567256"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.2.5.1.1 "><p id="p19879115418264"><a name="p19879115418264"></a><a name="p19879115418264"></a>elf heard infos</p>
</td>
<td class="cellrowborder" valign="top" width="59.78%" headers="mcps1.2.5.1.2 "><p id="p4879154152618"><a name="p4879154152618"></a><a name="p4879154152618"></a>包括ELF Header、Section Headers、Key to Flags、Program Headers、Symbol表等信息。</p>
</td>
<td class="cellrowborder" valign="top" width="10.059999999999999%" headers="mcps1.2.5.1.3 "><p id="p168301927153215"><a name="p168301927153215"></a><a name="p168301927153215"></a>否</p>
</td>
<td class="cellrowborder" valign="top" width="15.329999999999998%" headers="mcps1.2.5.1.4 "><p id="p11211185612914"><a name="p11211185612914"></a><a name="p11211185612914"></a>设置--verbose，开启全量打印。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  kernel type信息

<a name="table187419221164"></a>
<table><thead align="left"><tr id="row1074119226160"><th class="cellrowborder" valign="top" width="25.330000000000002%" id="mcps1.2.3.1.1"><p id="p1741722141612"><a name="p1741722141612"></a><a name="p1741722141612"></a>KERNEL_TYPE</p>
</th>
<th class="cellrowborder" valign="top" width="74.67%" id="mcps1.2.3.1.2"><p id="p274182251617"><a name="p274182251617"></a><a name="p274182251617"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row3741922101618"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p574192218164"><a name="p574192218164"></a><a name="p574192218164"></a>AICORE</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p1990333814243"><a name="p1990333814243"></a><a name="p1990333814243"></a><strong id="b17154104012247"><a name="b17154104012247"></a><a name="b17154104012247"></a>该参数为预留参数，当前版本暂不支持。</strong></p>
<p id="p36351632397"><a name="p36351632397"></a><a name="p36351632397"></a>算子执行时仅会启动AI Core，比如用户在host侧设置blocknum为5，则会启动5个AI Core。</p>
</td>
</tr>
<tr id="row18741162291619"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p1574102212166"><a name="p1574102212166"></a><a name="p1574102212166"></a>AIC</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p17634113218918"><a name="p17634113218918"></a><a name="p17634113218918"></a>算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置blocknum为10，则会启动10个Cube核。</p>
</td>
</tr>
<tr id="row13741202211169"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p107414226166"><a name="p107414226166"></a><a name="p107414226166"></a>AIV</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p14634183212912"><a name="p14634183212912"></a><a name="p14634183212912"></a>算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置blocknum为10，则会启动10个Vector核。</p>
</td>
</tr>
<tr id="row87416227162"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p167421222161611"><a name="p167421222161611"></a><a name="p167421222161611"></a>MIX_AIC_MAIN</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p16635332397"><a name="p16635332397"></a><a name="p16635332397"></a>AIC、AIV混合场景下，设置核函数的类型为MIX ，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置blocknum为10，且设置task_ration为1：2，则会启动10个Cube核和20个Vector核。</p>
</td>
</tr>
<tr id="row9742422181617"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p167420225160"><a name="p167420225160"></a><a name="p167420225160"></a>MIX_AIV_MAIN</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p12883939583"><a name="p12883939583"></a><a name="p12883939583"></a>AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置blocknum为10，且设置task_ration为1：2，则会启动10个Vector核和20个Cube核。</p>
</td>
</tr>
<tr id="row207422022191612"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p5742132211161"><a name="p5742132211161"></a><a name="p5742132211161"></a>AIC_ROLLBACK</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p3742422111616"><a name="p3742422111616"></a><a name="p3742422111616"></a>算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Cube Core使用。</p>
</td>
</tr>
<tr id="row1496174191710"><td class="cellrowborder" valign="top" width="25.330000000000002%" headers="mcps1.2.3.1.1 "><p id="p169611841101711"><a name="p169611841101711"></a><a name="p169611841101711"></a>AIV_ROLLBACK</p>
</td>
<td class="cellrowborder" valign="top" width="74.67%" headers="mcps1.2.3.1.2 "><p id="p196114161714"><a name="p196114161714"></a><a name="p196114161714"></a>算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Vector Core使用。</p>
</td>
</tr>
</tbody>
</table>

## 使用样例（Kernel直调算子工程）<a name="section12189203721319"></a>

以MatMulInvocationNeo算子为例（NPU模式），完整的工程可参考[Matmul多核Kernel直调样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/11_matmul_kernellaunch/MatmulInvocationNeo)。假设$\{cmake\_install\_dir\}为算子Cmake编译产物根目录，目录结构如下（仅为示例，具体以实际算子工程为准），类似[CMake编译配置文件编写](基于样例工程完成Kernel直调.md#section185111259496)。

```
out
├── lib
│   ├── libascendc_kernels_npu.so
├── include
│   ├── ascendc_kernels_npu
│           ├── aclrtlaunch_matmul_custom.h
│           ├── aclrtlaunch_triple_chevrons_func.h
......
```

工具对编译生成的库文件（如\*.so、\*.a等）进行解析和解压，功能实现命令样例如下：

-   **解析包含device信息的库文件**

    支持两种打印方式，请按需选取，解析字段含义参见[表4](#table217334916136)。

    -   简单打印

        ```
        msobjdump --dump-elf ${cmake_install_dir}/out/libascendc_kernels_npu.so
        ```

        执行上述命令，终端打印基础device信息，示例如下：

        ```
        ===========================
        [VERSION]: 1
        [TYPE COUNT]: 1
        ===========================
        [ELF FILE 0]: ascendxxxb1_ascendc_kernels_npu_0_mix.o
        [KERNEL TYPE]: mix
        [KERNEL LEN]: 511560
        [ASCEND META]: None
        ```

    -   全量打印

        ```
        msobjdump --dump-elf ${cmake_install_dir}/out/libascendc_kernels_npu.so --verbose
        ```

        执行上述命令，终端打印所有device信息，示例如下：

        ```
        ===========================
        [VERSION]: 1
        [TYPE COUNT]: 1
        ===========================
        [ELF FILE 0]: ascendxxxb1_ascendc_kernels_npu_0_mix.o
        [KERNEL TYPE]: mix
        [KERNEL LEN]: 511560
        [ASCEND META]: None
        ====== [elf heard infos] ======
        ELF Header:
          Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
          Class:                             ELF64
          Data:                              2's complement, little endian
          Version:                           1 (current)
          OS/ABI:                            UNIX - System V
          ABI Version:                       0
          Type:                              EXEC (Executable file)
          Machine:                           <unknown>: 0x1029
          Version:                           0x1
          Entry point address:               0x0
          Start of program headers:          64 (bytes into file)
          Start of section headers:          510280 (bytes into file)
          Flags:                             0x940000
          Size of this header:               64 (bytes)
          Size of program headers:           56 (bytes)
          Number of program headers:         2
          Size of section headers:           64 (bytes)
          Number of section headers:         20
          Section header string table index: 18
        
        Section Headers:
          [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
          [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
          [ 1] .text             PROGBITS        0000000000000000 0000b0 010a08 00  AX  0   0  4
           .....................................................................................
          [19] .strtab           STRTAB          0000000000000000 071278 00b6cb 00      0   0  1
        Key to Flags:
          W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
          L (link order), O (extra OS processing required), G (group), T (TLS),
          C (compressed), x (unknown), o (OS specific), E (exclude),
          D (mbind), p (processor specific)
        
        There are no section groups in this file.
        
        Program Headers:
          Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
          LOAD           0x0000b0 0x0000000000000000 0x0000000000000000 0x010aa8 0x010aa8 R E 0x1000
          GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0
        ......
        ```

-   **解压包含device信息的库文件并落盘**

    ```
    msobjdump --extract-elf ${cmake_install_dir}/out/libascendc_kernels_npu.so
    ```

    执行上述命令，默认在当前执行路径下落盘ascendxxxb1\_ascendc\_kernels\_npu\_0\_mix.o文件。

-   **获取包含device信息的库文件列表**

    ```
    msobjdump --list-elf ${cmake_install_dir}/out/libascendc_kernels_npu.so
    ```

    执行上述命令，终端会打印所有文件，屏显信息形如：

    ```
    ELF file    0: ascendxxxb1_ascendc_kernels_npu_0_mix.o
    ```

## 使用样例（标准/简易自定义算子工程）<a name="section12835815105114"></a>

以下面的算子工程为例（仅为示例，具体以实际算子工程为准），假设$\{cmake\_install\_dir\}为算子Cmake编译产物根目录，目录结构如下。

```
├── op_api
│   ├── include
│       ├── aclnn_acos_custom.h
│       ├── aclnn_matmul_leakyrelu_custom.h
│       ├── .........
│   ├── lib
│       ├── libcust_opapi.so
```

工具对编译生成的库文件（如\*.so、\*.a等）进行解析和解压，功能实现命令样例如下：

-   **解析包含device信息的库文件**

    支持两种打印方式，请按需选取，解析字段含义参见[表5](#table94384560259)。

    -   简单打印

        ```
        msobjdump --dump-elf ${cmake_install_dir}/op_api/lib/libcust_opapi.so 
        ```

        执行上述命令，终端打印基础device信息，示例如下：

        ```
        .ascend.meta META INFO
        VERSION: 1
        DEBUG: debugBufSize=0, debugOptions=0
        DYNAMIC_PARAM: dynamicParamMode=0
        OPTIONAL_PARAM: optionalInputMode=1, optionalOutputMode=1
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_1
        KERNEL_TYPE: AIV
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 1
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_2_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 2
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_3_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 3
        ....................................
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_1
        KERNEL_TYPE: AIV
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 1
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_2_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 2
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_3_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 3
        ```

    -   全量打印

        ```
        msobjdump --dump-elf ${cmake_install_dir}/op_api/lib/libcust_opapi.so --verbose
        ```

        执行上述命令，终端打印基础device信息，示例如下：

        ```
        .ascend.meta META INFO
        VERSION: 1
        DEBUG: debugBufSize=0, debugOptions=0
        DYNAMIC_PARAM: dynamicParamMode=0
        OPTIONAL_PARAM: optionalInputMode=1, optionalOutputMode=1
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_1
        KERNEL_TYPE: AIV
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 1
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_2_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 2
        .ascend.meta. [0]: AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26_3_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 3
        ....................................
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_1
        KERNEL_TYPE: AIV
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 1
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_2_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 2
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_3_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        DETERMINISTIC_INFO: 1
        BLOCK_NUM: 0xFFFFFFFF
        FUNCTION_ENTRY: 3
        ```

        ```
        ....................................
        ===== [elf heard infos] in ascendxxx_acos_custom_AcosCustom_da824ede53d7e754f85c14b9446ec2fc.o =====:
        ELF Header:
          Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
          Class:                             ELF64
          Data:                              2's complement, little endian
          Version:                           1 (current)
          OS/ABI:                            UNIX - System V
          ................................................
          Size of program headers:           56 (bytes)
          Number of program headers:         3
          Size of section headers:           64 (bytes)
          Number of section headers:         9
          Section header string table index: 7
        Section Headers:
          [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
          [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0 
           .....................................................................................
          [ 8] .strtab           STRTAB          0000000000000000 00529b 000119 00      0   0  1
        Key to Flags:
          W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
          L (link order), O (extra OS processing required), G (group), T (TLS),
          C (compressed), x (unknown), o (OS specific), E (exclude),
          D (mbind), p (processor specific)
        ................................................
        
        ===== [elf heard infos] in ascendxxx_matmul_leakyrelu_custom_MatmulLeakyreluCustom_e052bee3255764ac919095f3bdf83389.o =====:
        ELF Header:
          Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
          Class:                             ELF64
          Data:                              2's complement, little endian
          Version:                           1 (current)
          ................................................
          Section header string table index: 6
        Section Headers:
          [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
          [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
          [ 1] .text             PROGBITS        0000000000000000 0000e8 007ed8 00  AX  0   0  4
          [ 2] .data             PROGBITS        0000000000008000 0080e8 000008 00  WA  0   0 256
          [ 3] .comment          PROGBITS        0000000000000000 0080f0 000043 01  MS  0   0  1
          [ 4] .bl_uninit        NOBITS          0000000000000000 008133 000020 00      0   0  1
          [ 5] .symtab           SYMTAB          0000000000000000 008138 0000c0 18      7   1  8
          [ 6] .shstrtab         STRTAB          0000000000000000 0081f8 00003b 00      0   0  1
          [ 7] .strtab           STRTAB          0000000000000000 008233 0000ec 00      0   0  1
          ................................................
        ```

-   **解压包含device信息的库文件并落盘**

    ```
    msobjdump --extract-elf ${cmake_install_dir}/op_api/lib/libcust_opapi.so 
    ```

    执行上述命令，默认在当前执行路径下保存解压文件，产物目录如下：

    ```
    |-- config                                                               // 算子原型配置文件目录
    |    ├── ${soc_version}   
    |        ├── acos_custom.json                                  
    |        ├── matmul_leakyrelu_custom.json                               
    |        ├── .......                                             
    |-- ${soc_version}                                                    // AI处理器名
    |     ├── acos_custom                                               // 基础单算子编译文件*.o和对应的*.json文件
    |         ├── AcosCustom_da824ede53d7e754f85c14b9446ec2fc.json      // 命名规则：${op_type}_${parm_info}.json或${op_type}_${parm_info}.o，${parm_info}是基于算子输入/输出dtype、shape等信息生成的标识码
    |         ├── AcosCustom_da824ede53d7e754f85c14b9446ec2fc.o
    |         ├── AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26.json
    |         ├── AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26.o
    |     ├── matmul_leakyrelu_custom  
    |         ├── MatmulLeakyreluCustom_e052bee3255764ac919095f3bdf83389.json
    |         ├── MatmulLeakyreluCustom_e052bee3255764ac919095f3bdf83389.o
    |     ├── axpy_custom    
    |         ├── .....
    ```

    以acos\_custom算子编译产物解压为例：

    -   查看算子原型（acos\_custom.json）

        ```
        {
            "binList": [
                {
                    "implMode": "high_performance",
                    "int64Mode": false,
                    "simplifiedKeyMode": 0,
                    "simplifiedKey": [......],
                    "staticKey": "96b2b4bb2e3xxx,ee37ce8796ef139dexxxx",
                    "inputs": [
                        {
                            "name": "x",
                            "index": 0,
                            "dtype": "float32",
                            "format": "ND",
                            "paramType": "required",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "y",
                            "index": 0,
                            "dtype": "float32",
                            "format": "ND",
                            "paramType": "required",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic"
                        }
                    ],
                    "attrs": [
                        {
                            "name": "tmp",
                            "dtype": "int",
                            "value": 0
                        },
                        .........
                    ],
                    "opMode": "dynamic",
                    "optionalInputMode": "gen_placeholder",
                    "deterministic": "ignore",
                    "binInfo": {
                        "jsonFilePath": "ascendxxx/acos_custom/AcosCustom_da824ede53d7e754f85c14b9446ec2fc.json"
                    }
                },
                {
                    "implMode": "high_performance",
                    "int64Mode": false,
                    "simplifiedKeyMode": 0,
                    "simplifiedKey": [
          
                    ],
                    "staticKey": "27d6f997f2f3551axxxx,1385590c47affa578eb429xxx",
                    "inputs": [
                        {
                            "name": "x",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "required",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "y",
                            "index": 0,
                            "dtype": "float16",
                            "format": "ND",
                            "paramType": "required",
                            "shape": [
                                -2
                            ],
                            "format_match_mode": "FormatAgnostic"
                        }
                    ],
                    "attrs": [
                        {
                            "name": "tmp",
                            "dtype": "int",
                            "value": 0
                        },
                        .........
                    ],
                    "opMode": "dynamic",
                    "optionalInputMode": "gen_placeholder",
                    "deterministic": "ignore",
                    "binInfo": {
                        "jsonFilePath": "ascendxxx/acos_custom/AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26.json"
                    }
                }
            ]
        }
        ```

    -   解析$\{op\_type\}\_$\{parm\_info\}.o文件获取.ascend.meta段信息。

        ```
        msobjdump --dump-elf ./AcosCustom_da824ede53d7e754f85c14b9446ec2fc.o
        ```

        执行上述命令，终端屏显如下，字段与库文件解析类似，参见[表5](#table94384560259)。

        ```
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_1
        KERNEL_TYPE: AIV
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_2_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        .ascend.meta. [0]: AcosCustom_da824ede53d7e754f85c14b9446ec2fc_3_mix_aiv
        KERNEL_TYPE: MIX_AIV_MAIN
        MIX_TASK_RATION: [0:1]
        ```

    -   查看$\{op\_type\}\_$\{parm\_info\}.json，直观获取device文件中算子信息。

        ```
        {
            "binFileName": "AcosCustom_da824ede53d7e754f85c14b9446ec2fc",
            "binFileSuffix": ".o",
            "blockDim": -1,
            "coreType": "MIX",
            "intercoreSync": 1,
            "kernelName": "AcosCustom_da824ede53d7e754f85c14b9446ec2fc",
            "magic": "RT_DEV_BINARY_MAGIC_ELF",
            "memoryStamping": [],
            "opParaSize": 24,
            "parameters": [],
            "sha256": "94e32d04fcaf435411xxxxxxxx",
            "workspace": {
                "num": 1,
                "size": [
                    -1
                ],
                "type": [
                    0
                ]
            },
            "kernelList": [
                {
                    "tilingKey": 1,
                    "kernelType": "MIX_AIC",
                    "taskRation": "0:1",
                    "crossCoreSync": 0,
                    "kernelName": "AcosCustom_da824ede53d7e754f85c14b9446ec2fc_1"
                },
                .........
            ],
            "taskRation": "tilingKey",
            "optionalInputMode": "gen_placeholder",
            "debugOptions": "printf",
            "debugBufSize": 78643200,
            "compileInfo": {},
            "supportInfo": {                                                        // 算子原型信息
                "implMode": "high_performance",
                "int64Mode": false,
                "simplifiedKeyMode": 0,
                "simplifiedKey": [......],
                "staticKey": "96b2b4bb2e35fa3dxxx,ee37ce8796ef139dedxxxxxxxx",
                "inputs": [
                    {
                        "name": "x",
                        "index": 0,
                        "dtype": "float32",
                        "format": "ND",
                        "paramType": "required",
                        "shape": [
                            -2
                        ],
                        "format_match_mode": "FormatAgnostic"
                    }
                ],
                "outputs": [
                    {
                        "name": "y",
                        "index": 0,
                        "dtype": "float32",
                        "format": "ND",
                        "paramType": "required",
                        "shape": [
                            -2
                        ],
                        "format_match_mode": "FormatAgnostic"
                    }
                ],
                "attrs": [
                    {
                        "name": "tmp",
                        "dtype": "int",
                        "value": 0
                    },
                    .........
                ],
                "opMode": "dynamic",
                "optionalInputMode": "gen_placeholder",
                "deterministic": "ignore"
            },
            "filePath": "ascendxxx/acos_custom/AcosCustom_da824ede53d7e754f85c14b9446ec2fc.json"
        }
        ```

-   **获取包含device信息的库文件列表**

    ```
    msobjdump --list-elf ${cmake_install_dir}/op_api/lib/libcust_opapi.so 
    ```

    执行上述命令，终端会打印所有文件，屏显信息形如：

    ```
    ELF file    0: ascendxxx_acos_custom_AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26.json
    ELF file    1: ascendxxx_acos_custom_AcosCustom_dad9c8ca8fcbfd789010c8b1c0da8e26.o
    ....................
    ELF file    2: ascendxxx_acos_custom_AcosCustom_da824ede53d7e754f85c14b9446ec2fc.json
    ELF file    3: ascendxxx_acos_custom_AcosCustom_da824ede53d7e754f85c14b9446ec2fc.o
    ```

