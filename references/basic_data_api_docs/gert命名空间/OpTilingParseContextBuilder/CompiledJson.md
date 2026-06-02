# CompiledJson

**页面ID:** atlasopapi_07_00648  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00648.html

---

#### 函数功能

设置算子的CompiledJson指针，json格式文件指针。

#### 函数原型

```
OpTilingParseContextBuilder &CompiledJson(const ge::char_t *compiled_json)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| compiled_json | 输入 | 编译信息json文件指针。 |

#### 返回值说明

OpTilingParseContextBuilder对象本身，用于链式调用。

#### 约束说明

通过指针传入的参数（ge::char_t *），其内存所有权归调用者所有；调用者必须确保指针在ContextHolder对象的生命周期内有效。
