# DataTypeForBinQuery<a name="ZH-CN_TOPIC_0000002554343815"></a>

## 功能说明<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section36583473819"></a>

设置Input/Output用于运行时算子二进制查找的数据类型，和[DataType](DataType.md)/[DataTypeList](DataTypeList.md)的数量一致，且一一对应。

算子编译过程中，会根据数据类型生成多个.o，并通过这些数据类型在运行时索引算子二进制。某些算子支持多种数据类型，且对数据类型不敏感，这时可以使用该接口，将多种数据类型映射到同一个算子二进制，多个数据类型可以复用一个.o，从而减少二进制文件的生成。

例如，如果一个算子的输入支持多种数据类型（ge::DT\_INT16 和ge::DT\_INT32），并且使用ge::DT\_INT16 输入时可以复用ge::DT\_INT32 的二进制文件而不影响最终结果，那么可以采用如下配置：

```
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT16, ge::DT_INT32})
    .DataTypeForBinQuery({ge::DT_INT32, ge::DT_INT32})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
```

这样，只需生成一个目标文件（.o），就能实现对多种数据类型的支持。

## 函数原型<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &DataTypeForBinQuery(std::vector<ge::DataType> types)
```

## 参数说明<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p318615392613"></a>types</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_p096733515614"><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_p096733515614"></a><a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_p096733515614"></a>算子参数数据类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   DataTypeForBinQuery的参数个数需要和当前算子参数的DataType或者DataTypeList的参数个数保持一致。
-   不支持与[To](To.md)（指定数据类型）的接口同时使用。
-   需要保证使用DataTypeForBinQuery后，产生新的算子参数属性集合（使用DataTypeForBinQuery替换原本DataType序列）存在于原本支持的参数属性集合中。

    参数属性集合的定义为，算子所支持的所有参数的属性的集合，相当于一列参数的集合。

    例如[示例一](#li26512368417)中。算子支持四种原集合，没有重复。

    1、x : DT\_FLOAT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    2、x : DT\_FLOAT, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    3、x : DT\_INT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    4、x : DT\_INT32, FORMAT\_ND    y : DT\_INT16, FORMAT\_NC

    使用DataTypeForBinQuery替换原本DataType序列后，新集合为

    1、x : DT\_INT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    2、x : DT\_FLOAT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    3、x : DT\_FLOAT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_ND

    4、x : DT\_INT16, FORMAT\_ND    y : DT\_INT16, FORMAT\_NC

    此时发现，新集合1与原集合3一致，新集合2、新集合3与原集合1一致。设置生效。新集合4不属于原集合，设置失效。此时会按照原本的集合4进行编译。

## 调用示例<a name="zh-cn_topic_0000001600172389_zh-cn_topic_0000001526115138_zh-cn_topic_0000001575944081_section320753512363"></a>

-   <a name="li26512368417"></a>示例一

    ```
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
                .DataTypeForBinQuery({ge::DT_INT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NC});
    ```

    如下图所示，没有设置DataTypeForBinQuery之前，会生成4个二进制。通过上述代码设置DataTypeForBinQuery后：

    -   替换后第1列使用原来第3列的二进制，第2列和第3列使用原来第1列的二进制。第4列仍使用第4列的二进制。
    -   替换后，第2列和第3列完全一致，达成二进制复用的效果，算子总二进制会由原来的四个（bin1，bin2，bin3，bin4）缩减至现在的三个（bin1，bin3、bin4）。

    <!-- img2text -->
```text
                                  key1            key2            key3            key4
                                  bin1            bin2            bin3            bin4
                                   │               │               │               │
                                   ▼               ▼               ▼               ▼

┌──────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Input   │ DataType    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│    x     │             │  DT_FLOAT16  │  │   DT_FLOAT   │  │   DT_INT16   │  │   DT_INT32   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│          │ Format      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│          │             │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘
                         ┆                ┆                ┆                ┆
                         ┆                ┆                ┆                ┆
┌──────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Output  │ DataType    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│    y     │             │   DT_INT16   │  │   DT_INT16   │  │   DT_INT16   │  │   DT_INT16   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│          │ Format      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│          │             │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_NC   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘


┌──────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Input   │ DataType    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│    x     │             │  DT_FLOAT16  │  │   DT_FLOAT   │  │   DT_INT16   │  │   DT_INT32   │         │
│          │             └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
│          │ DataType    ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────────────┐         │
│          │ ForBinQuery │   DT_INT16   │  │  DT_FLOAT16  │  │  DT_FLOAT16  │  │   DT_INT16   │◄────┐  │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │  │
│          │ Format      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│          │             │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │     │  │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │  │
└──────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘  │
                                                                                                             │
                                                                 找不到对应的key和bin                              │
                                                                 仍使用原有数据类型                               │
                                                                                                             │

┌──────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Output  │ DataType    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│    y     │             │   DT_INT16   │  │   DT_INT16   │  │   DT_INT16   │  │   DT_INT16   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│          │ Format      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│          │             │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_ND   │  │  FORMAT_NC   │         │
│          │             └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘
                                   ▲               ▲               ▲               ▲
                                   │               │               │               │
                                key3            key1            key1            key 4
                                bin3            bin1            bin1            bin 4
```

-   示例二

    ```
    // 简单用例，此时会有两对复用，1、2列->1列，3、4列->4列。总共生成1、4两个二进制。所有支持的DataType会传入这两个二进制运行。
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
                .DataTypeForBinQuery({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    ```

-   示例三

    ```
    // 复杂用例，可以多个Input/Output同时使用DataTypeForBinQuery，此时也会产生两对复用。1、2列->2列，3、4列->1列。总共生成1、2两个二进制。所有支持的DataType会传入这两个二进制运行。
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
                .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
                .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
                .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    ```

