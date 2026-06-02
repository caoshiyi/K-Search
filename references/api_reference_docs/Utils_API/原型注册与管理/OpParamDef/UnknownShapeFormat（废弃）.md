# UnknownShapeFormat（废弃）

**页面ID:** atlasascendc_api_07_0963  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0963.html

---

#### 功能说明

> **注意:** 

- 该接口废弃，并将在后续版本移除，请不要使用该接口。无需针对动态/静态shape单独设置format，统一使用Format接口来设置即可。
- 如果开发者使用了该接口，并开启-Werror -Wall编译选项开启所有警告当做错误处理，会有编译报错。此时可以通过添加-Wno-deprecated编译选项来消除，但是存在后续接口在版本中移除后编译报错的风险，建议不要使用该接口，统一使用Format接口来设置。
>         

编译选项加在自定义算子工程目录下op_host/CMakeLists.txt中的cust_optiling、cust_opproto编译target上，样例如下：

```
target_compile_options(cust_optiling PRIVATE
        -Wno-deprecated
)
```

未知Shape情况下的Format的默认值。

#### 函数原型

```
OpParamDef &UnknownShapeFormat(std::vector<ge::Format> formats)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| formats | 输入 | 算子参数数据格式，ge::Format请参考Format。 |

#### 返回值说明

OpDef算子定义，OpDef请参考OpDef。

#### 约束说明

无
