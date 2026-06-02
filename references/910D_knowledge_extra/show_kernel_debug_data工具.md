# show\_kernel\_debug\_data工具<a name="ZH-CN_TOPIC_0000002523351542"></a>

静态图场景下，整图算子全部下沉到NPU侧执行，kernel侧单算子调试信息（通过[printf](printf.md)接口）需要在模型执行结束后才能获取。本工具提供了离线解析能力，帮助用户获取并解析调试信息（将bin文件解析成可读格式）。

> **说明：** 
>show\_kernel\_debug\_data支持多用户并发调用，但用户需要指定不同的落盘路径，否则可能出现落盘内容被覆盖等问题。

## 产品支持情况<a name="section15615918103719"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody></tbody>
</table>

## 工具安装<a name="section857412719218"></a>

1.  安装工具。

    工具跟随CANN软件包发布（参考[环境准备](环境准备.md)完成CANN安装），其路径默认为“$\{INSTALL\_DIR\}/tools/show\_kernel\_debug\_data”，其中$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

2.  设置环境变量。

    请以CANN软件包运行用户，执行如下命令使公共环境变量生效：

    ```
    source ${INSTALL_DIR}/bin/setenv.bash
    ```

3.  检查工具是否安装成功。

    执行如下命令，若能正常显示--help或-h信息，则表示工具环境正常，功能可正常使用。

    ```
    show_kernel_debug_data -h
    ```

## 使用方法<a name="section943033882116"></a>

-   **命令行方式**

    ```
    show_kernel_debug_data <bin_file_path> [<output_path>]
    ```

    <a name="table1331233514187"></a>
    <table><thead align="left"><tr id="row1931213353185"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.1.4.1.1"><p id="p031233531816"><a name="p031233531816"></a><a name="p031233531816"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="28.110000000000003%" id="mcps1.1.4.1.2"><p id="p1579144713356"><a name="p1579144713356"></a><a name="p1579144713356"></a>可选/必选</p>
    </th>
    <th class="cellrowborder" valign="top" width="52.5%" id="mcps1.1.4.1.3"><p id="p173127357185"><a name="p173127357185"></a><a name="p173127357185"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1431210350181"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.1.4.1.1 "><p id="p1272774591520"><a name="p1272774591520"></a><a name="p1272774591520"></a>&lt;bin_file_path&gt;</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.110000000000003%" headers="mcps1.1.4.1.2 "><p id="p11791194718355"><a name="p11791194718355"></a><a name="p11791194718355"></a>必选</p>
    </td>
    <td class="cellrowborder" valign="top" width="52.5%" headers="mcps1.1.4.1.3 "><p id="p74714124281"><a name="p74714124281"></a><a name="p74714124281"></a>kernel侧调试信息落盘的bin文件路径，例如“/input/dump_workspace.bin”。</p>
    </td>
    </tr>
    <tr id="row17312193519184"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.1.4.1.1 "><p id="p193121335101811"><a name="p193121335101811"></a><a name="p193121335101811"></a>&lt;output_path&gt;</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.110000000000003%" headers="mcps1.1.4.1.2 "><p id="p1279144713515"><a name="p1279144713515"></a><a name="p1279144713515"></a>可选</p>
    </td>
    <td class="cellrowborder" valign="top" width="52.5%" headers="mcps1.1.4.1.3 "><p id="p1573112228295"><a name="p1573112228295"></a><a name="p1573112228295"></a>解析结果的保存路径，例如“/output_dir”。默认是当前命令行执行目录下。</p>
    </td>
    </tr>
    </tbody>
    </table>

-   **API方式**

    获取kernel侧调试信息并解析成可读文件。函数原型如下。

    ```
    def show_kernel_debug_data(bin_file_path: str, output_path: str = './') -> None
    ```

    其中，输入参数说明如下。函数无输出参数和返回值。

    -   bin\_file\_path：kernel侧调试信息落盘的bin文件路径，字符串类型。
    -   output\_path：解析结果的保存路径，字符串类型，默认是当前接口调用脚本所在目录下。

    调用示例参考如下代码。

    ```
    from show_kernel_debug_data import show_kernel_debug_data
    show_kernel_debug_data(./input/dump_workspace.bin)
    ```

## 产物说明<a name="section7841124516211"></a>

工具解析结果文件目录结构如下：

```
├ ${output_path}
├── PARSER_${timestamp}           // ${timestamp}表示时间戳。
│   ├── parser.log              // 工具解析的日志，包含kernel侧日常流程和printf打印信息。
```

