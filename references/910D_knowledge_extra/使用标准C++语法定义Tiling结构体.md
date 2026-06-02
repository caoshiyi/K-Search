# 使用标准C++语法定义Tiling结构体<a name="ZH-CN_TOPIC_0000002523311598"></a>

## 具体步骤<a name="section17812263817"></a>

在定义Tiling结构体时，可以使用标准C++语法定义一个**POD类型（Plain Old Data）**，即与C语言兼容的数据类型。具体步骤如下。完整样例请参考[标准C++语法定义Tiling结构体样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomMultiCore)。

1.  使用C++语法定义Tiling结构体。

    > **说明：** 
    >该结构体定义所在的头文件应放置在算子工程的op\_kernel目录下。由于只有该目录下的文件会被打包进算子包，供在线编译场景中使用，若将文件放置在其他目录中，可能导致在线编译因找不到相关文件而失败。

    用户在使用高阶API的Tiling结构体时，通过AscendC::tiling命名空间引用"kernel\_tiling/kernel\_tiling.h"中预定义的Tiling结构体，如下代码所示。

    ```
    #ifndef MATMUL_CUSTOM_TILING_H
    #define MATMUL_CUSTOM_TILING_H
    #include <cstdint>
    #include "kernel_tiling/kernel_tiling.h"    // for TCubeTiling
    
    struct MatmulCustomTilingData {
        uint64_t localMemSize;
        AscendC::tiling::TCubeTiling cubeTilingData;
    };
    #endif  // MATMUL_CUSTOM_TILING_H
    ```

2.  Host侧Tiling函数中对Tiling结构体赋值。

    -   需要包含Tiling结构体定义头文件。
    -   通过GetTilingData获取Tiling结构体指针，并对其成员变量进行赋值。

    ```
    #include "../op_kernel/matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件
    ...
    
    namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ...
        MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
        ...
        // 获取Tiling结构体指针
        MatmulCustomTilingData *tiling = context->GetTilingData<MatmulCustomTilingData>();
        // 对tiling的成员变量赋值
        if (cubeTiling.GetTiling(tiling->cubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
        uint64_t localMemSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
        tiling->localMemSize = localMemSize;
        ...
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  Kernel侧注册Tiling结构体，解析Tiling数据至TilingData结构并使用。

    -   需要包含Tiling结构体定义头文件。
    -   通过[REGISTER\_TILING\_DEFAULT](REGISTER_TILING_DEFAULT.md)或者[REGISTER\_TILING\_FOR\_TILINGKEY](REGISTER_TILING_FOR_TILINGKEY.md)注册Tiling结构体；通过[GET\_TILING\_DATA](GET_TILING_DATA.md)解析Tiling数据至TilingData结构并使用。其中[REGISTER\_TILING\_DEFAULT](REGISTER_TILING_DEFAULT.md)同时也用于标识使用标准C++语法定义TilingData结构体。

    ```
    #include "kernel_operator.h"
    #include "matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件
    
    extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(MatmulCustomTilingData);
        GET_TILING_DATA(tilingData, tiling);
        MatmulKernel<half, half, float, float> matmulKernel;
        AscendC::TPipe pipe;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData); // Initialize the matmul object.
        matmulKernel.Init(a, b, bias, c, workspace, tilingData.localMemSize, tilingData.cubeTilingData);
        ...
    }
    ```

## 使用标准C++语法定义Tiling结构体的优势<a name="section92385314106"></a>

相比较使用BEGIN\_TILING\_DATA\_DEF等宏进行定义的方式，该方式不仅更符合C++开发者的开发习惯，并且提供了强大的灵活性。

-   支持bool类型，支持数组、结构体数组及列表初始化。

    ```
    class A {
    public:
        bool xxx;
        uint32_t xxx[2][128] = {0};
    };
    
    class B {
    public:
        bool xxx = false;
        uint8_t xxx[2][2]{0};
        A[10];
    };
    ```

-   不同算子可以支持定义同名但结构不同的Tiling结构体，通过算子引用对应的头文件即可实现区分。这种方式允许每个算子使用符合自身需求的Tiling结构定义，而不会与其他算子产生冲突。

    相比之下，使用BEGIN\_TILING\_DATA\_DEF等宏方式定义同名但结构不同的Tiling结构体时，由于这些结构体会被注册到全局的Tiling结构体管理变量中，可能导致后续通过结构体名称访问时，无法准确获取当前算子实际使用的Tiling结构体，从而引发未定义行为。

    算子A：

    ```
    class TilingData {
    public:
        uint32_t length;
    };
    ```

    算子B：

    ```
    class TilingData {
    public:
        uint32_t length;
        uint16_t coreNum;
    };
    ```

-   支持自定义Tiling赋值，用户可以通过访问Tiling结构体成员变量直接赋值，或自定义Tiling赋值函数（宏定义方式下，用户仅可通过框架生成的set\_xx/get\_xx方法赋值/访问）

    Tiling结构体定义：

    ```
    class TilingData {
    public:
        uint32_t xxx1;
        uint32_t xxx2;
        uint8_t xxx3;
        bool xxx4;
    };
    ```

    Host侧Tiling函数：

    ```
    #include "../op_kernel/xxx_custom_tiling.h"  // 包含Tiling结构体定义头文件
    ...
    
    namespace optiling {
    static void ComputeTiling(TilingData* tiling, ...)
    {
        // 计算Tiling逻辑
        ...
        tiling->xxx1 = xxx;
        tiling->xxx2 = xxx;
        tiling->xxx3 = xxx;
        tiling->bool = xxx;
    }
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {    
        ...
        TilingData *tiling = context->GetTilingData<TilingData>();
        ...
        ComputeTiling(tiling, ...)
        ...
    
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

## 使用约束<a name="section318444831016"></a>

使用标准C++语法定义Tiling结构体时存在如下约束限制：

-   Tiling结构体内不支持定义成员函数，因为成员函数存在Device侧和Host侧的差异（Device侧的函数需要\_\_aicore\_\_修饰符），而Tiling结构体Device侧和Host侧共用，所以会在编译或执行时出现问题：

    ```
    class TilingData {
    public:
        uint32_t xxx;
    
        __aicore__ funcA() { ... }  // 错误，host侧编译时不支持__aicore__修饰符，会出现编译错误
        void func() { ... }         // 错误，device侧缺少__aicore__修饰符，无法执行
    };
    ```

-   Tiling结构体成员变量不支持指针、引用类型，此类数据类型会导致Host侧到Device侧数据解析异常：

    ```
    class TilingData {
    public:
        uint32_t* totalLength; // 指针场景不支持，Host无法传递指针到Device
        uint32_t& tileNum;       // 引用场景不支持，Host无法传递指针到Device
    };
    ```

-   Tiling结构体仅支持POD类型，没有虚函数、虚继承等面向对象特性，也不支持模板类：

    ```
    class A {
    public:
        uint32_t totalLength;
        uint32_t tileNum;
    };
    class B: public A {
    public:
        uint32_t xxx;
        uint32_t xxx;
    };
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        // 错误用法
        B *tiling = context->GetTilingData<A>(); // 不支持，会触发未知问题
        // 正确用法
        B *tiling = context->GetTilingData<B>();
        ......
        return ge::GRAPH_SUCCESS;
    }
    ```

-   GetTilingData获取的Tiling不包含初值，需显式赋值或在Tiling结构体定义并调用Tiling赋值函数；

    ```
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData *tiling = context->GetTilingData<TilingData>(); //获取Tiling结构体，此时totalLength、tileNum为0，并不会带入初始值
        ......
        // 需显式赋值
        tiling->totalLength = totalLength;  // 赋值Tiling结构体成员变量
        tiling->tileNum = TILE_NUM;         // 赋值Tiling结构体成员变量
        ......
        return ge::GRAPH_SUCCESS;
    }
    ```

-   host侧和kernel侧的Tiling结构体支持传入模板参数。由于宏函数中逗号运算符的特殊性，在kernel侧宏函数（REGISTER\_TILING\_DEFAULT或者REGISTER\_TILING\_FOR\_TILINGKEY）使用带逗号的模板类型（如：template<int32\_t sizeA, int32\_t sizeB\>），存在编译异常，因此需要使用别名方式来定义带逗号的模板类型（如：using size = template<int32\_t sizeA, int32\_t sizeB\>）。具体示例如下：

    ```
    // 模板参数个数大于1的场景
    template<int32_t sizeA, int32_t sizeB>
    class A {
    public:
        uint32_t totalLength;
        uint32_t tileNum;
        uint32_t dataArray[sizeA];
    };
    // 模板参数个数等于1的场景
    template<int32_t sizeA>
    class B {
    public:
        uint32_t totalLength;
        uint32_t tileNum;
        uint32_t dataArray[sizeA];
    };
    
    // host侧可以直接传入Tiling结构体以及对应模板参数
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        // 模板参数个数等于1或者大于等于1的时候都可以直接传入
        A<3, 5> *tiling = context->GetTilingData<A<3,5>>();
        B<3> *tiling = context->GetTilingData<B<3>>();
        ......
        return ge::GRAPH_SUCCESS;
    }
    
    // kernel侧代码
    #include "kernel_operator.h"
    #include "add_custom_tiling.h"  // 包含Tiling结构体定义头文件
    extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        using aa = A<3,5>;
        REGISTER_TILING_DEFAULT(aa);                                // 模板参数个数大于1时，一定要用using来指定
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", B<3>);  // 模板参数个数等于1时，可以直接写明模板参数
        ......
    }
    ```

## 如何将宏定义Tiling结构体修改为标准C++语法<a name="section11732131341116"></a>

本节介绍如何将使用BEGIN\_TILING\_DATA\_DEF等宏进行定义的方式改造成使用标准C++语法的方式。

1.  **首先**将之前位于op\_host目录下的Tiling结构体定义头文件移至op\_kernel目录下，内容前后对比如下，**注意此时包含的头文件变化，不需要再包含宏定义相关的头文件**。

    **表 1**  两种方式对比

    <a name="table882614511524"></a>
    <table><thead align="left"><tr id="row78261151165217"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p19826125115523"><a name="p19826125115523"></a><a name="p19826125115523"></a>宏定义方式</p>
    </th>
    <th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p1882605117521"><a name="p1882605117521"></a><a name="p1882605117521"></a>标准C++语法定义方式</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row18266511523"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><pre class="screen" id="screen98261951195215"><a name="screen98261951195215"></a><a name="screen98261951195215"></a>#include "register/tilingdata_base.h"
    #include "tiling/tiling_api.h" // TCubeTiling结构体通过宏定义
    
    namespace optiling {
    BEGIN_TILING_DATA_DEF(MatmulCustomTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, localMemSize);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulCustomTilingData)
    } // namespace optiling</pre>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><pre class="screen" id="screen168271451135220"><a name="screen168271451135220"></a><a name="screen168271451135220"></a>#include &lt;cstdint&gt;
    #include "kernel_tiling/kernel_tiling.h" // TCubeTiling结构体通过C++语法定义
    
    struct MatmulCustomTilingData {
        uint64_t localMemSize;
        AscendC::tiling::TCubeTiling cubeTilingData;
    };</pre>
    </td>
    </tr>
    </tbody>
    </table>

2.  **然后**修改Host侧的Tiling函数实现，此时对Tiling结构体的成员变量赋值无需使用宏定义生成的set方法，而是使用用户熟悉的C++指针赋值方式。

    **表 2**  两种方式对比

    <a name="table3481171754810"></a>
    <table><thead align="left"><tr id="row5482151714815"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1848218179487"><a name="p1848218179487"></a><a name="p1848218179487"></a>宏定义方式</p>
    </th>
    <th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p948201764811"><a name="p948201764811"></a><a name="p948201764811"></a>标准C++语法定义方式</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row148281774813"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><pre class="screen" id="screen1773181331120"><a name="screen1773181331120"></a><a name="screen1773181331120"></a>namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ...
        MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
        ...
        MatmulCustomTilingData tiling;
        if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) { // Get matmul tiling.
            return ge::GRAPH_FAILED;
        }
    
        uint64_t localMemSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
        tiling.set_localMemSize(localMemSize);  // 需要使用宏定义方式生成的set方法
    
        ...
        // 需要将局部变量保存至context上下文
        tiling.SaveToBuffer(context-&gt;GetRawTilingData()-&gt;GetData(), context-&gt;GetRawTilingData()-&gt;GetCapacity());  
        ...
    
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling</pre>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><pre class="screen" id="screen34968325556"><a name="screen34968325556"></a><a name="screen34968325556"></a>#include "../op_kernel/matmul_custom_tiling.h"  // 包含Tiling结构体定义头文件
    ...
    
    namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ...
        MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
        ...
        MatmulCustomTilingData *tiling = context-&gt;GetTilingData&lt;MatmulCustomTilingData&gt;();
        if (cubeTiling.GetTiling(tiling-&gt;cubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
    
        uint64_t localMemSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
        tiling-&gt;localMemSize = localMemSize;  // 使用用户友好的C++指针方式赋值成员变量
    
        ...
    
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling</pre>
    </td>
    </tr>
    </tbody>
    </table>

3.  **最后**，在Kernel 函数入口处新增[REGISTER\_TILING\_DEFAULT](REGISTER_TILING_DEFAULT.md)调用，用于注册Tiling结构体。该注册操作的作用是：告知框架用户已使用标准 C++ 语法定义Tiling结构体，并明确其类型，以便框架在进行Tiling数据解析时能够正确识别和使用该结构体。

    ```
    #include "matmul_custom_tiling.h"
    ...
    
    extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(MatmulCustomTilingData);  // 新增REGISTER_TILING_DEFAULT调用注册Tiling结构体
        ...
    }
    ```

