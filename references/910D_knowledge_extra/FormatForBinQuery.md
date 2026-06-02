# FormatForBinQuery<a name="ZH-CN_TOPIC_0000002554424625"></a>

## 功能说明<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section36583473819"></a>

设置Input/Output用于运行时算子二进制查找的数据格式，和[Format](Format.md)/[FormatList](FormatList.md)的数量一致，且一一对应。

算子编译过程中，会根据数据格式生成多个.o，并通过这些数据格式在运行时索引算子二进制。某些算子支持多种数据格式，且对数据格式不敏感，这时可以使用该接口，将多种数据格式映射到同一个算子二进制，多个数据格式可以复用一个.o，从而减少二进制文件的生成。

例如，如果一个算子的输入支持多种数据格式（ge::FORMAT\_NC和ge::FORMAT\_ND），并且使用ge::FORMAT\_NC输入时可以复用ge::FORMAT\_ND的二进制文件而不影响最终结果，那么可以采用如下配置：

```
this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT16})
            .Format({ge::FORMAT_NC, ge::FORMAT_ND})
            .FormatForBinQuery({ge::FORMAT_ND,ge::FORMAT_ND});
```

这样，只需生成一个目标文件（.o），就能实现对多种数据格式的支持。

## 函数原型<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &FormatForBinQuery(std::vector<ge::Format> formats)
```

## 参数说明<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p318615392613"></a>formats</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_p43231148103313"><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_p43231148103313"></a><a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_p43231148103313"></a>算子参数数据格式。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpParamDef算子定义，OpParamDef请参考[OpParamDef](OpParamDef.md)。

## 约束说明<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001525424352_section19165124931511"></a>

-   FormatForBinQuery的参数个数需要和当前算子参数的Format或者FormatList的参数个数保持一致。
-   不支持与[Scalar](Scalar.md)/[ScalarList](ScalarList.md)同时使用。
-   不支持与[ValueDepend](ValueDepend.md)同时使用。
-   设置FormatForBinQuery后，会用FormatForBinQuery的数据格式替换当前Input/Output的数据格式，并检查新组合在替换前是否存在。如果存在，则指向对应的二进制，如果不存在，该参数失效，按照原来的数据格式生成。具体请参考[示例一](#li26512368417)。

## 调用示例<a name="zh-cn_topic_0000001549188228_zh-cn_topic_0000001526275046_zh-cn_topic_0000001575944081_section320753512363"></a>

-   <a name="li26512368417"></a>示例一

    ```
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .FormatForBinQuery({ge::FORMAT_NC, ge::FORMAT_NC, ge::FORMAT_ND, ge::FORMAT_NCHW});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NC, ge::FORMAT_ND});
    ```

    如下图所示，没有设置FormatForBinQuery之前，会生成4个二进制。通过上述代码设置FormatForBinQuery后：

    -   替换后第4列使用原来第2列的二进制，第1列和第2列使用原来第1列的二进制。第3列仍使用第3列的二进制。
    -   替换后，第1列和第2列完全一致，达成二进制复用的效果，算子总二进制会由原来的四个（bin1，bin2，bin3，bin4）缩减至现在的三个（bin1，bin2、bin3）。

    <!-- img2text -->
```text
                         key1            key2            key3            key4
                         bin1            bin2            bin3            bin4
                           │               │               │               │
                           ▼               ▼               ▼               ▼

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│           │                                                                                   │
│  Input    │    DataType        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    x      │                     │ DT_FLOAT16  │  │ DT_FLOAT16  │  │ DT_FLOAT16  │  │ DT_FLOAT16  │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                                                                                   │
│           │    Format          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│           │                     │ FORMAT_NC   │  │ FORMAT_NCHW │  │ FORMAT_NHWC │  │ FORMAT_ND   │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
└───────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│           │                                                                                   │
│  Output   │    DataType        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    y      │                     │  DT_INT16   │  │  DT_INT16   │  │  DT_INT16   │  │  DT_INT16   │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                                                                                   │
│           │    Format          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│           │                     │  FORMAT_ND  │  │  FORMAT_ND  │  │  FORMAT_NC  │  │  FORMAT_ND  │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
└───────────────────────────────────────────────────────────────────────────────────────────────┘


┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│           │                                                                                   │
│  Input    │    DataType        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    x      │                     │ DT_FLOAT16  │  │ DT_FLOAT16  │  │ DT_FLOAT16  │  │ DT_FLOAT16  │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                                                                                   │
│           │    Format          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│           │                     │ FORMAT_NC   │  │ FORMAT_NCHW │  │ FORMAT_NHWC │  │ FORMAT_ND   │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                                                                                   │
│           │    Format          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│           │    ForBinQuery      │ FORMAT_NC   │  │ FORMAT_NC   │  │ FORMAT_ND   │  │ FORMAT_NCHW │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                           │               │               │         ╲
│           │                           ▼               ▼               ▼          ╲
│           │                                                              找不到对应的key和bin
│           │                                                              仍使用原有数据类型
└───────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│           │                                                                                   │
│  Output   │    DataType        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    y      │                     │  DT_INT16   │  │  DT_INT16   │  │  DT_INT16   │  │  DT_INT16   │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│           │                                                                                   │
│           │    Format          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│           │                     │  FORMAT_ND  │  │  FORMAT_ND  │  │  FORMAT_NC  │  │  FORMAT_ND  │
│           │                     └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
└───────────────────────────────────────────────────────────────────────────────────────────────┘

                           ▲               ▲               ▲               ▲
                           │               │               │               │
                        key1            key1            key3            key 2
                        bin1            bin1            bin3            bin 2
```

说明:
- 上半部分表示设置 FormatForBinQuery 之前，共有 4 列，对应 4 组 key/bin：key1 bin1、key2 bin2、key3 bin3、key4 bin4。
- 下半部分表示设置 FormatForBinQuery 之后，Input x 增加了 `Format ForBinQuery` 行：
  - 第1列：FORMAT_NC  → 使用 `key1 bin1`
  - 第2列：FORMAT_NC  → 使用 `key1 bin1`
  - 第3列：FORMAT_ND  → 使用 `key3 bin3`
  - 第4列：FORMAT_NCHW → 使用 `key 2 bin 2`
- 第4列旁注“找不到对应的key和bin，仍使用原有数据类型”表示：虽然 ForBinQuery 被替换，但数据类型仍保持原有的 `DT_FLOAT16 / DT_INT16`。

-   示例二

    ```
            // 简单用例，此时会有两对复用，1、2列->1列，3、4列->4列。总共生成1、4两个二进制。所有支持的Format会传入这两个二进制运行。
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .FormatForBinQuery({ge::FORMAT_NC, ge::FORMAT_NC, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    
    ```

-   示例三

    ```
            // 复杂用例，可以多个Input/Output同时使用FormatBinQuery，此时也会产生两对复用。1、2列->1列，3、4列->3列。总共生成1、3两个二进制。所有支持的Format会传入这两个二进制运行。
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .FormatForBinQuery({ge::FORMAT_NC, ge::FORMAT_NC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .FormatForBinQuery({ge::FORMAT_NC, ge::FORMAT_NC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NC, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .FormatForBinQuery({ge::FORMAT_NC, ge::FORMAT_NC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
    ```

