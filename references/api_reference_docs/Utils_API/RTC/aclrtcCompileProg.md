# aclrtcCompileProg

**页面ID:** atlasascendc_api_07_00154  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00154.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

编译接口，编译指定的程序。

#### 函数原型

```
aclError aclrtcCompileProg(aclrtcProg prog, int numOptions, const char **options)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| prog | 输入 | 运行时编译程序的句柄。 |
| numOptions | 输入 | 编译选项数量。 |
| options | 输入 | 编译选项数组，保存具体的编译选项（默认添加-std=c++17）。          支持的编译选项可以参考毕昇编译器编译选项。 |

#### 返回值说明

aclError为int类型变量，详细说明请参考RTC错误码。

#### 约束说明

无

#### 调用示例

```
aclrtcProg prog;
const char *options[] = {"--npu-arch=dav-2201"};
int numOptions = sizeof(options) / sizeof(options[0]);
aclError result = aclrtcCompileProg(prog, numOptions, options);
```
