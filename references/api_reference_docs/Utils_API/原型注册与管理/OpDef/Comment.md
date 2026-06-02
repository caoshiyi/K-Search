# Comment

**页面ID:** atlasascendc_api_07_0956  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0956.html

---

#### 功能说明

设置算子分组信息以及算子原型注释，包括算子简述，算子约束等内容。用于在自动生成算子原型头文件时，同步生成算子原型注释。

#### 函数原型

```
OpDef &Comment(CommentSection section, const char *comment)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| section | 输入 | 通过CommentSection类去指定该接口的功能，支持以下几种取值：                     - CATEGORY：表示comment内容设置的是算子分组名称；           - BRIEF：表示comment内容设置的是算子@brief注释内容，即算子功能的简述；           - CONSTRAINTS：表示comment内容设置的是算子@Attention Constraints注释内容，即该算子的约束；           - RESTRICTIONS：表示comment内容设置的是算子@Restrictions注释内容，该选项当前属于试验参数，不推荐使用；           - SEE：表示comment内容设置的是算子@see注释内容，可以表示该算子的相关算子；           - THIRDPARTYFWKCOMPAT：表示comment内容设置的是算子参考的第三方算子。 |
| comment | 输入 | 增加comment注释。 |

#### 返回值说明

OpDef算子定义，OpDef请参考OpDef。

#### 约束说明

当用户使用CATEGORY参数设置算子分组名称时，会对应生成同名的代码文件。若文件名过长在编译时可能超过tar的打包文件名称长度限制，导致报错。

具体参考算子工程编译时出现文件名过长报错。

#### 调用示例

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
