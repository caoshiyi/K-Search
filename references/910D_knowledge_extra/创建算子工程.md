# 创建算子工程<a name="ZH-CN_TOPIC_0000002554431485"></a>

CANN开发套件包中提供了自定义算子工程生成工具msOpGen，可基于算子原型定义输出算子工程：包括**算子host侧代码实现文件**、**算子kernel侧实现文件**以及**工程编译配置文件等**。

> **说明：** 
>使用msOpGen工具创建算子工程之前，需要参考[环境准备](环境准备.md)章节安装驱动固件和CANN软件包，完成开发环境和运行环境的准备。

使用msOpGen工具创建算子开发工程的步骤如下：

1.  编写算子的原型定义json文件，用于生成算子开发工程。

    例如，AddCustom算子的json文件命名为add\_custom.json，文件内容如下：

    ```
    [
        {
            "op": "AddCustom",
            "input_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": [
                        "ND",
                        "ND",
                        "ND"
                    ],
                    "type": [
                        "float16",
                        "float",
                        "int32"
                    ]
                },
                {
                    "name": "y",
                    "param_type": "required",
                    "format": [
                        "ND",
                        "ND",
                        "ND"
                    ],
                    "type": [
                        "float16",
                        "float",
                        "int32"
                    ]
                }
            ],
            "output_desc": [
                {
                    "name": "z",
                    "param_type": "required",
                    "format": [
                        "ND",
                        "ND",
                        "ND"
                    ],
                    "type": [
                        "float16",
                        "float",
                        "int32"
                    ]
                }
            ]
        }
    ]
    ```

    例如，ReduceMaxCustom算子（包含属性）的json文件命名为reduce\_max\_custom.json，文件内容如下：

    ```
    [
        {
            "op": "ReduceMaxCustom",
             "input_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float16"]
                }
            ],
            "output_desc": [
                {
                    "name": "y",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float16"]
                },
                {
                    "name": "idx",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["int32"]
                }
            ],
            "attr": [                                                                   
                {
                    "name": "reduceDim",
                    "param_type": "required",
                    "type": "int"
                },
                {
                    "name": "isKeepDim",
                    "param_type": "optional",
                    "type": "int",
                    "default_value": 1
                }
            ]
        }
    ]
    ```

    > **说明：** 
    >原型定义json文件中的算子类型字段op需要采用**大驼峰**的命名方式，即采用大写字符区分不同的语义。

2.  使用msOpGen工具生成算子的开发工程。以生成AddCustom的算子工程为例，下文仅针对关键参数进行解释，详细参数说明请参见《算子开发工具用户指南》。

    ```
    ${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i $HOME/sample/add_custom.json -c ai_core-<soc_version> -lan cpp -out $HOME/sample/AddCustom
    ```

    -   **$\{INSTALL\_DIR\}**为CANN软件安装后文件存储路径，请根据实际环境进行替换。
    -   -i：指定算子原型定义文件_add\_custom_.json所在路径，请根据实际情况修改。
    -   -c：ai\_core-_<soc\_version\>_代表算子在AI Core上执行，_<soc\_version\>_为AI处理器的型号。

        > **说明：** 
        >基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

    -   -lan： 参数cpp代表算子基于Ascend C编程框架，使用C/C++编程语言开发。
    -   -out：生成文件所在路径，可配置为绝对路径或者相对路径，并且工具执行用户对路径具有可读写权限。若不配置，则默认生成在执行命令的当前路径。

3.  命令执行完后，会在-out指定目录或者默认路径下生成算子工程目录，工程中包含算子实现的模板文件，编译脚本等，以AddCustom算子为例，目录结构如下所示：

    ```
    AddCustom
    ├── build.sh         // 编译入口脚本
    ├── cmake            // 算子工程编译所需脚本及公共编译文件存放目录
    ├── CMakeLists.txt    // 算子工程的CMakeLists.txt
    ├── CMakePresets.json    // 编译配置项
    ├── framework        // AI框架适配时，算子插件实现文件目录
    ├── op_host                      // host侧实现文件
    │   ├── add_custom_tiling.h    // 算子tiling定义文件
    │   ├── add_custom.cpp         // 算子原型注册、shape推导、信息库、tiling实现等内容文件
    │   ├── CMakeLists.txt
    ├── op_kernel                   // kernel侧实现文件
    │   ├── CMakeLists.txt   
    │   ├── add_custom.cpp        // 算子代码实现文件 
    └── scripts                     // 自定义算子工程打包相关脚本所在目录
    ```

    > **说明：** 
    >-   上述目录结构中的粗体文件为后续算子开发过程中需要修改的文件，其他文件无需修改。
    >-   kernel侧实现依赖的所有文件需全部放置在op\_kernel目录下，不遵循该约束会导致[源码发布](算子工程编译.md#li139988411333)场景在线编译失败。因为在后续的算子源码发布时，只会打包发布op\_kernel目录下的文件。

工程目录中的op\_kernel和op\_host包含了算子的核心实现文件。op\_kernel下存放[kernel侧算子实现](Kernel侧算子实现.md)。op\_host下存放host侧代码实现，包括[算子原型定义](算子原型定义.md)、[host侧tiling实现](Host侧Tiling实现.md)。其中kernel侧算子实现和host侧tiling实现在[SIMD算子实现](SIMD算子实现.md)章节已经介绍了其核心的实现方法，在该章节会侧重于介绍接入CANN框架后的编程模式和API的使用。工程目录中的CMakePresets.json，用于开发者完成工程编译相关配置，之后即可进行编译部署。

