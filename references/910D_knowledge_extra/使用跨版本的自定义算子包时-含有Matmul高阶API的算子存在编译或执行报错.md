# 使用跨版本的自定义算子包时，含有Matmul高阶API的算子存在编译或执行报错<a name="ZH-CN_TOPIC_0000002554351471"></a>

## 现象描述<a name="section151611254194612"></a>

1.  基于CANN-7.2及之前版本（<=7.2）的CANN开发套件包，编译含有Matmul高阶API的自定义算子包，将编译后的自定义算子包安装至CANN-7.3及之后版本（\>=7.3）的CANN包环境，然后对该含有Matmul高阶API的算子，执行图模式在线编译时，报如下错误：

    ```
    res = struct.unpack_from(fmt_str, tiling_data, offset + unpack_size) 
    struct.error: unpack_from requires a buffer of at least 52 bytes for unpacking 4 bytes at offset 48
    ```

1.  基于CANN-7.2及之前版本（<=7.2）的CANN开发套件包，编译[sample样例仓](https://gitee.com/ascend/samples/tree/master/operator)中含有Matmul高阶API的算子，例如[MatmulLeakyReluCustomSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/12_matmulleakyrelu_frameworklaunch)，将编译后的自定义算子包安装至CANN-7.3及之后版本（\>=7.3）的CANN包环境，然后对该含有Matmul高阶API的算子，执行单算子API的调用时，报如下错误：

    ```
    ERROR：acl executable run failed! please check your project!
    ```

## 问题根因<a name="section417961104715"></a>

该错误的原因是编译自定义算子包的软件版本过老，可通过更新自定义算子包编译环境上的CANN开发套件包版本，然后重新编译和部署自定义算子包，来避免出现该问题。

## 处理步骤<a name="section166318242419"></a>

1.  查看自定义算子包编译时使用的CANN开发套件包版本号，示例如下：

    ```
    cd ${CANN包安装路径}
    cat version.cfg
    
    # version: 1.0
    runtime_running_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
    runtime_upgrade_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
    runtime_installed_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
    ```

2.  在编译自定义算子包的环境上，下载并安装CANN-7.3及之后版本的CANN开发套件包，详情可参见《CANN软件安装指南》。
3.  基于CANN-7.3及之后版本（\>=7.3）的CANN开发套件包，重新编译该自定义算子包。部署编译生成的自定义算子包后，正常编译或者执行算子，无报错。重新编译和部署自定义算子包的具体方法可参考[算子包编译](算子包编译.md)。

