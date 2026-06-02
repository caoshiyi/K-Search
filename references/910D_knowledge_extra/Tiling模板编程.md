# Tiling模板编程<a name="ZH-CN_TOPIC_0000002554351503"></a>

在[TilingKey编程](基本流程.md#li578045965)章节介绍的TilingKey编程方式中，TilingKey不易于记忆和理解，因为它们通常是较长又没有明确含义的数字。

在涉及多个TilingKey的场景中，开发者依赖TilingKey来管理kernel的实现，无论是在管理还是使用上都会遇到相当大的复杂性。为了简化这一过程，可以采用模板编程的方法来替代传统的TilingKey编程，从而减少对TilingKey数值标识的依赖，使kernel的管理更加直观和高效。使用步骤如下：

1.  <a name="li1949014102516"></a>在[自定义算子工程](工程化算子开发.md)的op\_kernel目录下，新增定义模板参数和模板参数组合的头文件，本示例中头文件命名为tiling\_key\_add\_custom.h。

    -   该头文件中需要包含模板头文件ascendc/host\_api/tiling/template\_argument.h。
    -   定义模板参数ASCENDC\_TPL\_ARGS\_DECL和模板参数组合ASCENDC\_TPL\_ARGS\_SEL（即可使用的模板）。具体API参考见[模板参数定义](模板参数定义.md)。

    ```
    #include "ascendc/host_api/tiling/template_argument.h"
    
    // 模板参数
    ASCENDC_TPL_ARGS_DECL(AddCustomTemplate, // 算子OpType
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),  // DataType类型的模板参数定义：输入参数x的数据类型，取值范围为float16/float32, ASCENDC_TPL_INPUT(0)说明对应Kernel侧第0个输入
    ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(1)),  // DataType类型的模板参数定义：输入参数y的数据类型，取值范围为float16/float32, ASCENDC_TPL_INPUT(1)说明对应Kernel侧第1个输入
    ASCENDC_TPL_DATATYPE_DECL(D_T_Z, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_OUTPUT(0)), // DataType类型的模板参数定义：输入参数z的数据类型，取值范围为float16/float32, ASCENDC_TPL_OUTPUT(0)说明对应Kernel侧第0个输出
    ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5, 10, 12, 13, 9, 8),// 自定义UINT类型（无符号整形）的模板参数定义：模板参数为切分的块数，编码位宽为ASCENDC_TPL_8_BW即8比特，表示该模板参数的个数不超过8比特能表达的范围；ASCENDC_TPL_UI_MIX表示通过混合模式表达取值范围，有2组的数据{0-2}、{3-5}和穷举值10、12、13、9、8，最后结果为{0, 1, 2, 3, 4, 5, 10, 12, 13, 9, 8}
    ASCENDC_TPL_BOOL_DECL(IS_SPLIT, 0, 1), // 自定义bool类型的模板参数定义：模板参数为是否切分标志位，取值范围为0和1，1表示切分，0表示不切分
    );
    
    // 模板参数组合
    // 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
    ASCENDC_TPL_SEL(
        ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), // Kernel类型选择，无需在模板参数声明中定义，在SEL中直接配置，所有ASCENDC_TPL_ARGS_SEL是否配置需要保持统一，如不配置将走自动推导流程
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Z, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
        ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1)
        ),
        ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Z, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
        ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1)
        ),
    );
    ```

2.  host侧调用ASCENDC\_TPL\_SEL\_PARAM接口自动生成并配置TilingKey。

    -   host实现文件中包含[步骤1](#li1949014102516)中定义模板参数和模板参数组合的头文件。
    -   调用ASCENDC\_TPL\_SEL\_PARAM接口自动生成并配置TilingKey，ASCENDC\_TPL\_SEL\_PARAM输入参数为模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。

    ```
    #include "tiling_key_add_custom.h"
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
        ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
        ge::DataType dtype_z = context->GetOutputDesc(1)->GetDataType();
        uint32_t D_T_X = static_cast<int>(dtype_x), D_T_Y = static_cast<int>(dtype_y), D_T_Z = static_cast<int>(dtype_z), TILE_NUM = 1, IS_SPLIT = 0;
        if(totalLength< MIN_LENGTH_FOR_SPLIT){
            IS_SPLIT = 0;
            TILE_NUM = 1;
        }else{
            IS_SPLIT = 1;
            TILE_NUM = DEFAULT_TILE_NUM;
        }
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    ```

3.  kernel侧实现

    -   kernel实现文件中包含[步骤1](#li1949014102516)中定义模板参数和模板参数组合的头文件。
    -   核函数添加template模板，以便支持模板参数的传入，参数顺序需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。
    -   通过对模板参数的分支判断，选择不同的kernel侧实现。

    ```
    #include "tiling_key_add_custom.h"
    ...
    ...
    template<typename D_T_X, typename D_T_Y, typename D_T_Z, int TILE_NUM, int IS_SPLIT>
     __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        KernelAdd<D_T_X, D_T_Y, D_T_Z> op;
        op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
        if constexpr (std::is_same_v<D_T_X, float> && std::is_same_v<D_T_Y, float> && std::is_same_v<D_T_Z, float>) {
            op.Process1();
        } else if constexpr (std::is_same_v<D_T_X, half> && std::is_same_v<D_T_Y, half> && std::is_same_v<D_T_Z, half>){
            if (IS_SPLIT == 0) {
                op.Process1();
            } else if(IS_SPLIT == 1) {
                op.Process2();
            }
        }
    }
    ```

> **说明：** 
>完整样例请参考[Tiling模板编程样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/6_addtemplate_frameworklaunch)。

