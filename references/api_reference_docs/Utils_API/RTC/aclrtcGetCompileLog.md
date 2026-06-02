# aclrtcGetCompileLog

**页面ID:** atlasascendc_api_07_00160  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00160.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取编译日志的内容，以字符串形式保存。

#### 函数原型

```
aclError aclrtcGetCompileLog(aclrtcProg prog, char *log)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| prog | 输入 | 运行时编译程序的句柄。 |
| log | 输出 | 编译日志的内容，以字符串形式保存。 |

#### 返回值说明

aclError为int类型变量，详细说明请参考RTC错误码。

#### 约束说明

无

#### 调用示例

```
aclrtcProg prog;
size_t logSize;
aclrtcGetCompileLogSize(prog, &logSize);
char* log = (char*)malloc(logSize);
aclError result = aclrtcGetCompileLog(prog, log);
```
