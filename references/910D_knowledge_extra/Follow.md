# Follow<a name="ZH-CN_TOPIC_0000002554424231"></a>

## 功能说明<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

用于指定当前输入/输出的datatype/format/shape信息与之前定义过的某个输入一致。

## 函数原型<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

-   datatype/format/shape全部信息保持一致

    ```
    OpParamDef &Follow(const char *paramName)
    ```

-   datatype/format/shape指定信息保持一致

    ```
    OpParamDef &Follow(const char *paramName, FollowType ftype)
    ```

## 参数说明<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a>paramName</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002052867297_p15488205233414"><a name="zh-cn_topic_0000002052867297_p15488205233414"></a><a name="zh-cn_topic_0000002052867297_p15488205233414"></a>之前定义过的输入名。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002052867297_row03533135010"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002052867297_p11333395017"><a name="zh-cn_topic_0000002052867297_p11333395017"></a><a name="zh-cn_topic_0000002052867297_p11333395017"></a>ftype</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002052867297_p231133115014"><a name="zh-cn_topic_0000002052867297_p231133115014"></a><a name="zh-cn_topic_0000002052867297_p231133115014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002052867297_p1731033135017"><a name="zh-cn_topic_0000002052867297_p1731033135017"></a><a name="zh-cn_topic_0000002052867297_p1731033135017"></a>ftype类型为枚举类FollowType，表示Follow的模式，取值如下：</p>
<a name="zh-cn_topic_0000002052867297_ul1782019075220"></a><a name="zh-cn_topic_0000002052867297_ul1782019075220"></a><ul id="zh-cn_topic_0000002052867297_ul1782019075220"><li>ALL，datatype/format/shape与paramName均保持一致；</li><li>DTYPE，datatype与paramName保持一致；</li><li>FORMAT，format与paramName保持一致；</li><li>SHAPE，shape与paramName保持一致。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   所有Follow的数据源头仅支持为Input。
-   针对shape推导，仅支持输出参数Follow输入参数，不支持输入参数Follow另一个输入参数。
-   可以支持链式Follow，例如C Follow B，B Follow A，但此时Follow的模式不可中途变更（ftype需要保持一致）。
-   使用Follow接口通常比InferShape函数逻辑更加简单，能用Follow表达的逻辑，建议使用Follow接口，则无需再编写注册InferShape函数。
-   InferShape推导函数和使用Follow接口去Follow shape不能混用，即不支持部分输出采用Infershape推导、部分输出采用Follow推导的情况。若用户同时使用了InferShape函数和Follow接口，以用户的InferShape函数为准，需要保证在InferShape函数中能够推导出所有的输出shape。
-   datatype/format同时支持Follow输入的参数类型为[DataTypeList](DataTypeList.md)/[FormatList](FormatList.md)  ，调用Follow后，当前输入/输出的datatype/format与paramName组合后的datatype/format一致。

## 调用示例<a name="zh-cn_topic_0000002052867297_zh-cn_topic_0000001526594958_zh-cn_topic_0000001575944081_section320753512363"></a>

-   输出“y1”Follow输入“x1”场景，此时“y1”的datatype、format以及shape都将会和“x1”保持一致。

    ```
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y1")
        .ParamType(REQUIRED)
        .Follow("x1");
    ```

-   链式follow场景，y1 -\> x2 -\> x1。此时“y1”的datatype、format以及shape都将会和“x1”保持一致，“x2”的datatype、format和“x1”保持一致。

    ```
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("x2")
        .ParamType(REQUIRED)
        .Follow("x1");
    this->Output("y1")
        .ParamType(REQUIRED)
        .Follow("x2");
    ```

