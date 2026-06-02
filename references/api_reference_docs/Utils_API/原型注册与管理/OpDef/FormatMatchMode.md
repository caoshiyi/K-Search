# FormatMatchMode

**页面ID:** atlasascendc_api_07_00006  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00006.html

---

#### 功能说明

设置输入输出tensor的format匹配模式。

#### 函数原型

```
OpDef &FormatMatchMode(FormatCheckOption option)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| option | 输入 | 匹配模式配置参数，类型为FormatCheckOption枚举类。支持以下几种取值： - DEFAULT：对NCHW/NHWC/DHWCN/NCDHW/NCL格式的输入输出转成ND格式进行处理；- STRICT：对数据格式需要严格区分，针对NCHW/NHWC/DHWCN/NCDHW/NCL格式，aclnn框架侧不做转换处理。 |

#### 返回值说明

OpDef算子定义，OpDef请参考OpDef。

#### 约束说明

不调用该接口的情况下，默认将NCHW/NHWC/DHWCN/NCDHW/NCL格式的输入输出转成ND格式进行处理。

#### 调用示例

下面示例中，算子AddCustom输入x只支持format为NCHW，输入y只支持format为NHWC，需要配置FormatMatchMode(FormatCheckOption::STRICT)，如果不配置aclnn框架会转成ND格式传给算子tiling。

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
