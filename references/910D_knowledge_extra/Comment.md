# Comment<a name="ZH-CN_TOPIC_0000002554424093"></a>

## 功能说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section36583473819"></a>

设置算子分组信息以及算子原型注释，包括算子简述，算子约束等内容。用于在自动生成算子原型头文件时，同步生成算子原型注释。

基于OpDef算子原型定义，自定义算子工程可以实现如下自动化能力：自动生成图模式场景使用的算子原型定义REG\_OP（算子原型头文件），开发者可以使用生成的算子原型进行构图、图编译、图执行等操作。

生成的注释有助于辅助理解算子原型，并可以基于这些注释自动生成算子原型的文档说明。通常情况下，内置CANN算子使用较多。开发者可以按需使用。

## 函数原型<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpDef &Comment(CommentSection section, const char *comment)
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
<tbody><tr id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p318615392613"></a>section</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_p096733515614"><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_p096733515614"></a><a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_p096733515614"></a>通过CommentSection类去指定该接口的功能，支持以下几种取值：</p>
<a name="zh-cn_topic_0000002091517061_ul1917131251512"></a><a name="zh-cn_topic_0000002091517061_ul1917131251512"></a><ul id="zh-cn_topic_0000002091517061_ul1917131251512"><li>CATEGORY：表示comment内容设置的是算子分组名称；</li><li>BRIEF：表示comment内容设置的是算子@brief注释内容，即算子功能的简述；</li><li>CONSTRAINTS：表示comment内容设置的是算子@Attention Constraints注释内容，即该算子的约束；</li><li>RESTRICTIONS：表示comment内容设置的是算子@Restrictions注释内容，该选项当前属于试验参数，不推荐使用；</li><li>SEE：表示comment内容设置的是算子@see注释内容，可以表示该算子的相关算子；</li><li>THIRDPARTYFWKCOMPAT：表示comment内容设置的是算子参考的第三方算子。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000002091517061_row1946220499205"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002091517061_p104626492207"><a name="zh-cn_topic_0000002091517061_p104626492207"></a><a name="zh-cn_topic_0000002091517061_p104626492207"></a>comment</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002091517061_p7462104912020"><a name="zh-cn_topic_0000002091517061_p7462104912020"></a><a name="zh-cn_topic_0000002091517061_p7462104912020"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002091517061_p84621049172017"><a name="zh-cn_topic_0000002091517061_p84621049172017"></a><a name="zh-cn_topic_0000002091517061_p84621049172017"></a>增加comment注释。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpDef算子定义，OpDef请参考[OpDef](OpDef.md)。

## 约束说明<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001525424352_section19165124931511"></a>

当用户使用CATEGORY参数设置算子分组名称时，会对应生成同名的代码文件。若文件名过长在编译时可能超过tar的打包文件名称长度限制，导致报错。

具体参考[算子工程编译时出现文件名过长报错](算子工程编译时出现文件名过长报错.md)。

## 调用示例<a name="zh-cn_topic_0000002091517061_zh-cn_topic_0000001526111046_zh-cn_topic_0000001575944081_section320753512363"></a>

```
 AddCustomComment(const char* name) : OpDef(name)
{
	this->Comment(CommentSection::CATEGORY, "catg"); // 算子分组
	this->Comment(CommentSection::BRIEF, "Brief cmt") // BRIEF注释
	        .Comment(CommentSection::CONSTRAINTS, "Constraints cmt1") // CONSTRAINTS注释
	        .Comment(CommentSection::CONSTRAINTS, "Constraints cmt2");
	this->Comment(CommentSection::RESTRICTIONS, "Restrictions cmt1") // RESTRICTIONS注释
		.Comment(CommentSection::RESTRICTIONS, "Restrictions cmt2")
		.Comment(CommentSection::THIRDPARTYFWKCOMPAT, "Third-party framework compatibility cmt1") // THIRDPARTYFWKCOMPAT注释
		.Comment(CommentSection::THIRDPARTYFWKCOMPAT, "Third-party framework compatibility cmt2")
		.Comment(CommentSection::SEE, "See cmt1")// SEE注释
		.Comment(CommentSection::SEE, "See cmt2");
	this->Input("x")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT, ge::DT_INT32})
		.FormatList({ge::FORMAT_ND});
	this->Input("y")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT, ge::DT_INT32})
		.FormatList({ge::FORMAT_ND});

	this->Output("z")
		.ParamType(REQUIRED)
		.DataType({ge::DT_FLOAT, ge::DT_INT32})
		.FormatList({ge::FORMAT_ND});
	this->AICore()
		.SetTiling(optiling::TilingFunc);
	this->AICore().AddConfig("ascendxxx");
}
```

