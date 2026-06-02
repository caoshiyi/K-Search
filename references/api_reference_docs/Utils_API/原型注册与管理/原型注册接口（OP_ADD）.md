# 原型注册接口（OP_ADD）

**页面ID:** atlasascendc_api_07_0945  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0945.html

---

#### 功能说明

注册算子的原型定义，从而确保算子能够被框架正确识别、编译和执行****。

算子原型主要描述了算子的输入输出、属性等信息以及算子在AI处理器上相关实现信息，并关联tiling实现等函数。算子原型通过自定义的算子类来承载，该算子类继承自OpDef类。完成算子的原型定义等操作后，需要调用OP_ADD接口，传入算子类型（自定义算子类的类名），进行算子原型注册。详细内容请参考算子原型定义。

#### 函数原型

```
OP_ADD(opType)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| opType | 输入 | 算子类型名称 |

#### 约束说明

无
