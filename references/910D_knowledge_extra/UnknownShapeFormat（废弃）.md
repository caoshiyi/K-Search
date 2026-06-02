# UnknownShapeFormat（废弃）<a name="ZH-CN_TOPIC_0000002554344141"></a>

## 功能说明<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section36583473819"></a>

> **须知：** 
>-   该接口废弃，并将在后续版本移除，请不要使用该接口。无需针对动态/静态shape单独设置format，统一使用[Format](Format.md)接口来设置即可。
>-   如果开发者使用了该接口，并开启-Werror -Wall编译选项开启所有警告当做错误处理，会有编译报错。此时可以通过添加-Wno-deprecated编译选项来消除，但是存在后续接口在版本中移除后编译报错的风险，建议不要使用该接口，统一使用[Format](Format.md)接口来设置。
>    编译选项加在自定义算子工程目录下op\_host/CMakeLists.txt中的cust\_optiling、cust\_opproto编译target上，样例如下：
>    ```
>    target_compile_options(cust_optiling PRIVATE
>            -Wno-deprecated
>    )
>    ```

未知Shape情况下的Format的默认值。

## 函数原型<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpParamDef &UnknownShapeFormat(std::vector<ge::Format> formats)
```

## 参数说明<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a>formats</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_p43231148103313"><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_p43231148103313"></a><a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_p43231148103313"></a>算子参数数据格式。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpDef算子定义，OpDef请参考[OpDef](OpDef.md)。

## 约束说明<a name="zh-cn_topic_0000001549347680_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

