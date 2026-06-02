# FormatMatchMode<a name="ZH-CN_TOPIC_0000002554424707"></a>

## 功能说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section36583473819"></a>

设置输入输出tensor的format匹配模式。

## 函数原型<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpDef &FormatMatchMode(FormatCheckOption option)
```

## 参数说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p19712111142511"><a name="p19712111142511"></a><a name="p19712111142511"></a>option</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p265693111279"><a name="p265693111279"></a><a name="p265693111279"></a>匹配模式配置参数，类型为FormatCheckOption枚举类。支持以下几种取值：</p>
<a name="zh-cn_topic_0000002091517061_ul1917131251512"></a><a name="zh-cn_topic_0000002091517061_ul1917131251512"></a><ul id="zh-cn_topic_0000002091517061_ul1917131251512"><li>DEFAULT：对<span>NCHW/NHWC/DHWCN/NCDHW/NCL格式的输入输出转成ND格式进行处理</span>；</li><li>STRICT：对数据格式需要严格区分，<span>针对NCHW/NHWC/DHWCN/NCDHW/NCL格式，aclnn框架侧不做转换处理</span><span>。</span></li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpDef算子定义，OpDef请参考[OpDef](OpDef.md)。

## 约束说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section19165124931511"></a>

不调用该接口的情况下，默认将NCHW/NHWC/DHWCN/NCDHW/NCL格式的输入输出转成ND格式进行处理。

## 调用示例<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_section320753512363"></a>

下面示例中，算子AddCustom输入x只支持format为NCHW，输入y只支持format为NHWC，需要配置FormatMatchMode\(FormatCheckOption::STRICT\)，如果不配置aclnn框架会转成ND格式传给算子tiling。

```
AddCustom(const char* name) : OpDef(name)
{
	this->Input("x")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT})
		.FormatList({ge::FORMAT_NCHW});
	this->Input("y")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT})
		.FormatList({ge::FORMAT_NHWC});
	this->Output("z")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT})
		.FormatList({ge::FORMAT_ND});
	this->AICore().SetTiling(optiling::TilingFunc);
	this->AICore().AddConfig("ascendxxx");
        this->FormatMatchMode(FormatCheckOption::STRICT);
}
```

