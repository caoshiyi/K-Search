# RegOpLibInit

**页面ID:** atlasopapi_07_00543  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00543.html

---

#### 函数功能

注册自定义算子动态库的初始化函数。

#### 函数原型

```
OpLibRegister &RegOpLibInit(OpLibInitFunc func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 要注册的自定义初始化函数，类型为OpLibInitFunc。 ``` using OpLibInitFunc = uint32_t (*)(ge::AscendString&); ``` |  |  |

#### 返回值说明

返回一个OpLibRegister对象，该对象新增注册了OpLibInitFunc函数func。

#### 约束说明

无。

#### 调用示例

```
uint32_t Init(ge::AscendString&) {
  // init func
  return 0;
}

REGISTER_OP_LIB(vendor_1).RegOpLibInit(Init); // 注册厂商名为vendor_1的初始化函数Init
```
