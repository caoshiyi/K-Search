# api_reference 文档索引

## AI CPU API

- [DataStoreBarrier](AI_CPU_API/DataStoreBarrier.md)
- [assert](AI_CPU_API/assert.md)
- [printf](AI_CPU_API/printf.md)

## Utils API

### C++标准库

#### 容器函数

- [get](Utils_API/C++标准库/容器函数/get.md)
- [make_tuple](Utils_API/C++标准库/容器函数/make_tuple.md)
- [tuple](Utils_API/C++标准库/容器函数/tuple.md)

#### 算法

- [max](Utils_API/C++标准库/算法/max.md)
- [min](Utils_API/C++标准库/算法/min.md)

#### 类型特性

- [conditional](Utils_API/C++标准库/类型特性/conditional.md)
- [enable_if](Utils_API/C++标准库/类型特性/enable_if.md)
- [integral_constant](Utils_API/C++标准库/类型特性/integral_constant.md)
- [is_base_of](Utils_API/C++标准库/类型特性/is_base_of.md)
- [is_convertible](Utils_API/C++标准库/类型特性/is_convertible.md)
- [is_same](Utils_API/C++标准库/类型特性/is_same.md)

#### 通用工具

- [integer_sequence](Utils_API/C++标准库/通用工具/integer_sequence.md)

### RTC

- [RTC简介](Utils_API/RTC/RTC简介.md)
- [RTC错误码](Utils_API/RTC/RTC错误码.md)
- [aclrtcCompileProg](Utils_API/RTC/aclrtcCompileProg.md)
- [aclrtcCreateProg](Utils_API/RTC/aclrtcCreateProg.md)
- [aclrtcDestroyProg](Utils_API/RTC/aclrtcDestroyProg.md)
- [aclrtcGetBinData](Utils_API/RTC/aclrtcGetBinData.md)
- [aclrtcGetBinDataSize](Utils_API/RTC/aclrtcGetBinDataSize.md)
- [aclrtcGetCompileLog](Utils_API/RTC/aclrtcGetCompileLog.md)
- [aclrtcGetCompileLogSize](Utils_API/RTC/aclrtcGetCompileLogSize.md)

### Tiling下沉

- [DEVICE_IMPL_OP_OPTILING](Utils_API/Tiling下沉/DEVICE_IMPL_OP_OPTILING.md)

### Tiling数据结构注册

- [TilingData结构定义](Utils_API/Tiling数据结构注册/TilingData结构定义.md)
- [TilingData结构注册](Utils_API/Tiling数据结构注册/TilingData结构注册.md)

### Tiling模板编程

- [ASCENDC_TPL_SEL_PARAM](Utils_API/Tiling模板编程/ASCENDC_TPL_SEL_PARAM.md)
- [模板参数定义](Utils_API/Tiling模板编程/模板参数定义.md)

### Tiling调测

#### ContextBuilder

- [AddAttr](Utils_API/Tiling调测/ContextBuilder/AddAttr.md)
- [AddInputTd](Utils_API/Tiling调测/ContextBuilder/AddInputTd.md)
- [AddOutputTd](Utils_API/Tiling调测/ContextBuilder/AddOutputTd.md)
- [AddPlatformInfo](Utils_API/Tiling调测/ContextBuilder/AddPlatformInfo.md)
- [BuildKernelRunContext](Utils_API/Tiling调测/ContextBuilder/BuildKernelRunContext.md)
- [BuildTilingContext](Utils_API/Tiling调测/ContextBuilder/BuildTilingContext.md)
- [CompileInfo](Utils_API/Tiling调测/ContextBuilder/CompileInfo.md)
- [ContextBuilder构造函数](Utils_API/Tiling调测/ContextBuilder/ContextBuilder构造函数.md)
- [Inputs](Utils_API/Tiling调测/ContextBuilder/Inputs.md)
- [IrInstanceNum](Utils_API/Tiling调测/ContextBuilder/IrInstanceNum.md)
- [KernelRunContextHolder结构定义](Utils_API/Tiling调测/ContextBuilder/KernelRunContextHolder结构定义.md)
- [NodeIoNum](Utils_API/Tiling调测/ContextBuilder/NodeIoNum.md)
- [Outputs](Utils_API/Tiling调测/ContextBuilder/Outputs.md)
- [PlatformInfo](Utils_API/Tiling调测/ContextBuilder/PlatformInfo.md)
- [SetOpNameType](Utils_API/Tiling调测/ContextBuilder/SetOpNameType.md)
- [TilingData](Utils_API/Tiling调测/ContextBuilder/TilingData.md)
- [Workspace](Utils_API/Tiling调测/ContextBuilder/Workspace.md)
- [简介](Utils_API/Tiling调测/ContextBuilder/简介.md)

#### OpTilingRegistry

- [GetTilingFunc](Utils_API/Tiling调测/OpTilingRegistry/GetTilingFunc.md)
- [LoadTilingLibrary](Utils_API/Tiling调测/OpTilingRegistry/LoadTilingLibrary.md)
- [构造和析构函数](Utils_API/Tiling调测/OpTilingRegistry/构造和析构函数.md)
- [简介](Utils_API/Tiling调测/OpTilingRegistry/简介.md)

### log

- [ASC_CPU_LOG](Utils_API/log/ASC_CPU_LOG.md)

### 原型注册与管理

#### OpAICoreConfig

- [DynamicCompileStaticFlag](Utils_API/原型注册与管理/OpAICoreConfig/DynamicCompileStaticFlag.md)
- [DynamicFormatFlag](Utils_API/原型注册与管理/OpAICoreConfig/DynamicFormatFlag.md)
- [DynamicRankSupportFlag](Utils_API/原型注册与管理/OpAICoreConfig/DynamicRankSupportFlag.md)
- [DynamicShapeSupportFlag](Utils_API/原型注册与管理/OpAICoreConfig/DynamicShapeSupportFlag.md)
- [ExtendCfgInfo](Utils_API/原型注册与管理/OpAICoreConfig/ExtendCfgInfo.md)
- [Input](Utils_API/原型注册与管理/OpAICoreConfig/Input.md)
- [NeedCheckSupportFlag](Utils_API/原型注册与管理/OpAICoreConfig/NeedCheckSupportFlag.md)
- [OpAICoreConfig构造函数](Utils_API/原型注册与管理/OpAICoreConfig/OpAICoreConfig构造函数.md)
- [Output](Utils_API/原型注册与管理/OpAICoreConfig/Output.md)
- [PrecisionReduceFlag](Utils_API/原型注册与管理/OpAICoreConfig/PrecisionReduceFlag.md)

#### OpAICoreDef

- [AddConfig](Utils_API/原型注册与管理/OpAICoreDef/AddConfig.md)
- [LaunchWithZeroEleOutputTensors](Utils_API/原型注册与管理/OpAICoreDef/LaunchWithZeroEleOutputTensors.md)
- [SetCheckSupport](Utils_API/原型注册与管理/OpAICoreDef/SetCheckSupport.md)
- [SetOpSelectFormat](Utils_API/原型注册与管理/OpAICoreDef/SetOpSelectFormat.md)
- [SetTiling](Utils_API/原型注册与管理/OpAICoreDef/SetTiling.md)

#### OpAttrDef

- [Comment](Utils_API/原型注册与管理/OpAttrDef/Comment.md)
- [OpAttrDef](Utils_API/原型注册与管理/OpAttrDef/OpAttrDef.md)

#### OpDef

- [AICore](Utils_API/原型注册与管理/OpDef/AICore.md)
- [Attr](Utils_API/原型注册与管理/OpDef/Attr.md)
- [Comment](Utils_API/原型注册与管理/OpDef/Comment.md)
- [EnableFallBack](Utils_API/原型注册与管理/OpDef/EnableFallBack.md)
- [FormatMatchMode](Utils_API/原型注册与管理/OpDef/FormatMatchMode.md)
- [Input](Utils_API/原型注册与管理/OpDef/Input.md)
- [MC2](Utils_API/原型注册与管理/OpDef/MC2.md)
- [Output](Utils_API/原型注册与管理/OpDef/Output.md)
- [SetInferDataType](Utils_API/原型注册与管理/OpDef/SetInferDataType.md)
- [SetInferShape](Utils_API/原型注册与管理/OpDef/SetInferShape.md)
- [SetInferShapeRange](Utils_API/原型注册与管理/OpDef/SetInferShapeRange.md)

#### OpMC2Def

- [HcclGroup](Utils_API/原型注册与管理/OpMC2Def/HcclGroup.md)
- [HcclServerType](Utils_API/原型注册与管理/OpMC2Def/HcclServerType.md)
- [OpMC2Def构造函数](Utils_API/原型注册与管理/OpMC2Def/OpMC2Def构造函数.md)
- [operator=](Utils_API/原型注册与管理/OpMC2Def/operator=.md)

#### OpParamDef

- [AutoContiguous](Utils_API/原型注册与管理/OpParamDef/AutoContiguous.md)
- [Comment](Utils_API/原型注册与管理/OpParamDef/Comment.md)
- [DataType](Utils_API/原型注册与管理/OpParamDef/DataType.md)
- [DataTypeForBinQuery](Utils_API/原型注册与管理/OpParamDef/DataTypeForBinQuery.md)
- [DataTypeList](Utils_API/原型注册与管理/OpParamDef/DataTypeList.md)
- [Follow](Utils_API/原型注册与管理/OpParamDef/Follow.md)
- [Format](Utils_API/原型注册与管理/OpParamDef/Format.md)
- [FormatForBinQuery](Utils_API/原型注册与管理/OpParamDef/FormatForBinQuery.md)
- [FormatList](Utils_API/原型注册与管理/OpParamDef/FormatList.md)
- [IgnoreContiguous](Utils_API/原型注册与管理/OpParamDef/IgnoreContiguous.md)
- [InitValue](Utils_API/原型注册与管理/OpParamDef/InitValue.md)
- [OutputShapeDependOnCompute](Utils_API/原型注册与管理/OpParamDef/OutputShapeDependOnCompute.md)
- [ParamType](Utils_API/原型注册与管理/OpParamDef/ParamType.md)
- [Scalar](Utils_API/原型注册与管理/OpParamDef/Scalar.md)
- [ScalarList](Utils_API/原型注册与管理/OpParamDef/ScalarList.md)
- [To](Utils_API/原型注册与管理/OpParamDef/To.md)
- [UnknownShapeFormat（废弃）](Utils_API/原型注册与管理/OpParamDef/UnknownShapeFormat（废弃）.md)
- [ValueDepend](Utils_API/原型注册与管理/OpParamDef/ValueDepend.md)
- [Version](Utils_API/原型注册与管理/OpParamDef/Version.md)

- [OpAICoreConfig注册接口（REGISTER_OP_AICORE_CONFIG）](Utils_API/原型注册与管理/OpAICoreConfig注册接口（REGISTER_OP_AICORE_CONFIG）.md)
- [原型注册接口（OP_ADD）](Utils_API/原型注册与管理/原型注册接口（OP_ADD）.md)

### 平台信息获取

#### PlatformAscendC

- [CalcTschBlockDim](Utils_API/平台信息获取/PlatformAscendC/CalcTschBlockDim.md)
- [GetCoreMemBw](Utils_API/平台信息获取/PlatformAscendC/GetCoreMemBw.md)
- [GetCoreMemSize](Utils_API/平台信息获取/PlatformAscendC/GetCoreMemSize.md)
- [GetCoreNum](Utils_API/平台信息获取/PlatformAscendC/GetCoreNum.md)
- [GetCoreNumAic](Utils_API/平台信息获取/PlatformAscendC/GetCoreNumAic.md)
- [GetCoreNumAiv](Utils_API/平台信息获取/PlatformAscendC/GetCoreNumAiv.md)
- [GetCoreNumVector](Utils_API/平台信息获取/PlatformAscendC/GetCoreNumVector.md)
- [GetLibApiWorkSpaceSize](Utils_API/平台信息获取/PlatformAscendC/GetLibApiWorkSpaceSize.md)
- [GetResCubeGroupWorkSpaceSize](Utils_API/平台信息获取/PlatformAscendC/GetResCubeGroupWorkSpaceSize.md)
- [GetResGroupBarrierWorkSpaceSize](Utils_API/平台信息获取/PlatformAscendC/GetResGroupBarrierWorkSpaceSize.md)
- [GetSocVersion](Utils_API/平台信息获取/PlatformAscendC/GetSocVersion.md)
- [PlatformAscendC简介](Utils_API/平台信息获取/PlatformAscendC/PlatformAscendC简介.md)
- [ReserveLocalMemory](Utils_API/平台信息获取/PlatformAscendC/ReserveLocalMemory.md)
- [构造及析构函数](Utils_API/平台信息获取/PlatformAscendC/构造及析构函数.md)

- [PlatformAscendCManager](Utils_API/平台信息获取/PlatformAscendCManager.md)

## 其他数据类型

### TensorDesc

- [GetDataObj](其他数据类型/TensorDesc/GetDataObj.md)
- [GetDataPtr](其他数据类型/TensorDesc/GetDataPtr.md)
- [GetDim](其他数据类型/TensorDesc/GetDim.md)
- [GetIndex](其他数据类型/TensorDesc/GetIndex.md)
- [GetShape](其他数据类型/TensorDesc/GetShape.md)
- [SetShapeAddr](其他数据类型/TensorDesc/SetShapeAddr.md)
- [TensorDesc简介](其他数据类型/TensorDesc/TensorDesc简介.md)
- [构造和析构函数](其他数据类型/TensorDesc/构造和析构函数.md)

- [BinaryRepeatParams](其他数据类型/BinaryRepeatParams.md)
- [ListTensorDesc](其他数据类型/ListTensorDesc.md)
- [ShapeInfo](其他数据类型/ShapeInfo.md)
- [TPosition](其他数据类型/TPosition.md)
- [UnaryRepeatParams](其他数据类型/UnaryRepeatParams.md)

## 基础API

### Cube分组管理（ISASI）

#### CubeResGroupHandle

- [AllocMessage](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/AllocMessage.md)
- [AssignQueue](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/AssignQueue.md)
- [CreateCubeResGroup](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/CreateCubeResGroup.md)
- [CubeResGroupHandle使用说明](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/CubeResGroupHandle使用说明.md)
- [CubeResGroupHandle构造函数](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/CubeResGroupHandle构造函数.md)
- [FreeMessage](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/FreeMessage.md)
- [PostFakeMsg](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/PostFakeMsg.md)
- [PostMessage](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/PostMessage.md)
- [SetQuit](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/SetQuit.md)
- [SetSkipMsg](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/SetSkipMsg.md)
- [Wait](基础API/Cube分组管理（ISASI）/CubeResGroupHandle/Wait.md)

#### GroupBarrier

- [Arrive](基础API/Cube分组管理（ISASI）/GroupBarrier/Arrive.md)
- [GetWorkspaceLen](基础API/Cube分组管理（ISASI）/GroupBarrier/GetWorkspaceLen.md)
- [GroupBarrier使用说明](基础API/Cube分组管理（ISASI）/GroupBarrier/GroupBarrier使用说明.md)
- [GroupBarrier构造函数](基础API/Cube分组管理（ISASI）/GroupBarrier/GroupBarrier构造函数.md)
- [Wait](基础API/Cube分组管理（ISASI）/GroupBarrier/Wait.md)

#### KfcWorkspace

- [GetKfcWorkspace](基础API/Cube分组管理（ISASI）/KfcWorkspace/GetKfcWorkspace.md)
- [UpdateKfcWorkspace](基础API/Cube分组管理（ISASI）/KfcWorkspace/UpdateKfcWorkspace.md)
- [构造函数与析构函数](基础API/Cube分组管理（ISASI）/KfcWorkspace/构造函数与析构函数.md)

### Kernel Tiling

- [COPY_TILING_WITH_ARRAY](基础API/Kernel_Tiling/COPY_TILING_WITH_ARRAY.md)
- [COPY_TILING_WITH_STRUCT](基础API/Kernel_Tiling/COPY_TILING_WITH_STRUCT.md)
- [GET_TILING_DATA](基础API/Kernel_Tiling/GET_TILING_DATA.md)
- [GET_TILING_DATA_MEMBER](基础API/Kernel_Tiling/GET_TILING_DATA_MEMBER.md)
- [GET_TILING_DATA_PTR_WITH_STRUCT](基础API/Kernel_Tiling/GET_TILING_DATA_PTR_WITH_STRUCT.md)
- [GET_TILING_DATA_WITH_STRUCT](基础API/Kernel_Tiling/GET_TILING_DATA_WITH_STRUCT.md)
- [REGISTER_NONE_TILING](基础API/Kernel_Tiling/REGISTER_NONE_TILING.md)
- [REGISTER_TILING_DEFAULT](基础API/Kernel_Tiling/REGISTER_TILING_DEFAULT.md)
- [REGISTER_TILING_FOR_TILINGKEY](基础API/Kernel_Tiling/REGISTER_TILING_FOR_TILINGKEY.md)
- [TILING_KEY_IS](基础API/Kernel_Tiling/TILING_KEY_IS.md)
- [TILING_KEY_LIST](基础API/Kernel_Tiling/TILING_KEY_LIST.md)
- [设置Kernel类型](基础API/Kernel_Tiling/设置Kernel类型.md)

### 原子操作

- [GetStoreAtomicConfig(ISASI)](基础API/原子操作/GetStoreAtomicConfig（ISASI）.md)
- [SetAtomicAdd](基础API/原子操作/SetAtomicAdd.md)
- [SetAtomicMax(ISASI)](基础API/原子操作/SetAtomicMax（ISASI）.md)
- [SetAtomicMin(ISASI)](基础API/原子操作/SetAtomicMin（ISASI）.md)
- [SetAtomicNone](基础API/原子操作/SetAtomicNone.md)
- [SetAtomicType](基础API/原子操作/SetAtomicType.md)
- [SetStoreAtomicConfig(ISASI)](基础API/原子操作/SetStoreAtomicConfig（ISASI）.md)

### 同步控制

#### 任务间同步

- [SetNextTaskStart](基础API/同步控制/任务间同步/SetNextTaskStart.md)
- [WaitPreTaskEnd](基础API/同步控制/任务间同步/WaitPreTaskEnd.md)

#### 核内同步

##### TQueSync

- [SetFlag/WaitFlag](基础API/同步控制/核内同步/TQueSync/SetFlag_WaitFlag.md)
- [模板参数](基础API/同步控制/核内同步/TQueSync/模板参数.md)

- [DataSyncBarrier(ISASI)](基础API/同步控制/核内同步/DataSyncBarrier（ISASI）.md)
- [PipeBarrier(ISASI)](基础API/同步控制/核内同步/PipeBarrier（ISASI）.md)
- [SetFlag/WaitFlag(ISASI)](基础API/同步控制/核内同步/SetFlag_WaitFlag（ISASI）.md)
- [同步控制简介](基础API/同步控制/核内同步/同步控制简介.md)

#### 核间同步

- [CrossCoreSetFlag(ISASI)](基础API/同步控制/核间同步/CrossCoreSetFlag（ISASI）.md)
- [CrossCoreWaitFlag(ISASI)](基础API/同步控制/核间同步/CrossCoreWaitFlag（ISASI）.md)
- [IBSet](基础API/同步控制/核间同步/IBSet.md)
- [IBWait](基础API/同步控制/核间同步/IBWait.md)
- [InitDetermineComputeWorkspace](基础API/同步控制/核间同步/InitDetermineComputeWorkspace.md)
- [NotifyNextBlock](基础API/同步控制/核间同步/NotifyNextBlock.md)
- [SyncAll](基础API/同步控制/核间同步/SyncAll.md)
- [WaitPreBlock](基础API/同步控制/核间同步/WaitPreBlock.md)

### 工具函数

- [Async](基础API/工具函数/Async.md)
- [GetTaskRatio](基础API/工具函数/GetTaskRatio.md)

### 数据搬运

#### DataCopy

- [DataCopy简介](基础API/数据搬运/DataCopy/DataCopy简介.md)
- [切片数据搬运](基础API/数据搬运/DataCopy/切片数据搬运.md)
- [基础数据搬运](基础API/数据搬运/DataCopy/基础数据搬运.md)
- [增强数据搬运](基础API/数据搬运/DataCopy/增强数据搬运.md)
- [随路转换ND2NZ搬运](基础API/数据搬运/DataCopy/随路转换ND2NZ搬运.md)
- [随路转换NZ2ND搬运](基础API/数据搬运/DataCopy/随路转换NZ2ND搬运.md)
- [随路量化激活搬运](基础API/数据搬运/DataCopy/随路量化激活搬运.md)

- [Copy](基础API/数据搬运/Copy.md)
- [DataCopyPad(ISASI)](基础API/数据搬运/DataCopyPad（ISASI）.md)
- [SetPadValue(ISASI)](基础API/数据搬运/SetPadValue（ISASI）.md)

### 标量计算

- [CountBitsCntSameAsSignBit](基础API/标量计算/CountBitsCntSameAsSignBit.md)
- [ScalarCast](基础API/标量计算/ScalarCast.md)
- [ScalarCountLeadingZero](基础API/标量计算/ScalarCountLeadingZero.md)
- [ScalarGetCountOfValue](基础API/标量计算/ScalarGetCountOfValue.md)
- [ScalarGetSFFValue](基础API/标量计算/ScalarGetSFFValue.md)
- [ToBfloat16](基础API/标量计算/ToBfloat16.md)
- [ToFloat](基础API/标量计算/ToFloat.md)

### 矢量计算

#### 基础算术

- [Abs](基础API/矢量计算/基础算术/Abs.md)
- [Add](基础API/矢量计算/基础算术/Add.md)
- [Adds](基础API/矢量计算/基础算术/Adds.md)
- [BilinearInterpolation(ISASI)](基础API/矢量计算/基础算术/BilinearInterpolation（ISASI）.md)
- [Div](基础API/矢量计算/基础算术/Div.md)
- [Exp](基础API/矢量计算/基础算术/Exp.md)
- [LeakyRelu](基础API/矢量计算/基础算术/LeakyRelu.md)
- [Ln](基础API/矢量计算/基础算术/Ln.md)
- [Max](基础API/矢量计算/基础算术/Max.md)
- [Maxs](基础API/矢量计算/基础算术/Maxs.md)
- [Min](基础API/矢量计算/基础算术/Min.md)
- [Mins](基础API/矢量计算/基础算术/Mins.md)
- [Mul](基础API/矢量计算/基础算术/Mul.md)
- [Muls](基础API/矢量计算/基础算术/Muls.md)
- [Reciprocal](基础API/矢量计算/基础算术/Reciprocal.md)
- [Relu](基础API/矢量计算/基础算术/Relu.md)
- [Rsqrt](基础API/矢量计算/基础算术/Rsqrt.md)
- [Sqrt](基础API/矢量计算/基础算术/Sqrt.md)
- [Sub](基础API/矢量计算/基础算术/Sub.md)
- [更多样例](基础API/矢量计算/基础算术/更多样例.md)

#### 复合计算

- [AddDeqRelu](基础API/矢量计算/复合计算/AddDeqRelu.md)
- [AddRelu](基础API/矢量计算/复合计算/AddRelu.md)
- [AddReluCast](基础API/矢量计算/复合计算/AddReluCast.md)
- [Axpy](基础API/矢量计算/复合计算/Axpy.md)
- [CastDeq](基础API/矢量计算/复合计算/CastDeq.md)
- [FusedMulAdd](基础API/矢量计算/复合计算/FusedMulAdd.md)
- [FusedMulAddRelu](基础API/矢量计算/复合计算/FusedMulAddRelu.md)
- [MulAddDst](基础API/矢量计算/复合计算/MulAddDst.md)
- [MulCast](基础API/矢量计算/复合计算/MulCast.md)
- [SubRelu](基础API/矢量计算/复合计算/SubRelu.md)
- [SubReluCast](基础API/矢量计算/复合计算/SubReluCast.md)

#### 归约计算

- [BlockReduceMax](基础API/矢量计算/归约计算/BlockReduceMax.md)
- [BlockReduceMin](基础API/矢量计算/归约计算/BlockReduceMin.md)
- [BlockReduceSum](基础API/矢量计算/归约计算/BlockReduceSum.md)
- [GetAccVal(ISASI)](基础API/矢量计算/归约计算/GetAccVal（ISASI）.md)
- [GetReduceMaxMinCount(ISASI)](基础API/矢量计算/归约计算/GetReduceMaxMinCount（ISASI）.md)
- [PairReduceSum](基础API/矢量计算/归约计算/PairReduceSum.md)
- [ReduceMax](基础API/矢量计算/归约计算/ReduceMax.md)
- [ReduceMin](基础API/矢量计算/归约计算/ReduceMin.md)
- [ReduceSum](基础API/矢量计算/归约计算/ReduceSum.md)
- [RepeatReduceSum](基础API/矢量计算/归约计算/RepeatReduceSum.md)
- [WholeReduceMax](基础API/矢量计算/归约计算/WholeReduceMax.md)
- [WholeReduceMin](基础API/矢量计算/归约计算/WholeReduceMin.md)
- [WholeReduceSum](基础API/矢量计算/归约计算/WholeReduceSum.md)

#### 排序组合（ISASI）

- [GetMrgSortResult](基础API/矢量计算/排序组合（ISASI）/GetMrgSortResult.md)
- [MrgSort](基础API/矢量计算/排序组合（ISASI）/MrgSort.md)
- [MrgSort4](基础API/矢量计算/排序组合（ISASI）/MrgSort4.md)
- [ProposalConcat](基础API/矢量计算/排序组合（ISASI）/ProposalConcat.md)
- [ProposalExtract](基础API/矢量计算/排序组合（ISASI）/ProposalExtract.md)
- [RpSort16](基础API/矢量计算/排序组合（ISASI）/RpSort16.md)
- [Sort32](基础API/矢量计算/排序组合（ISASI）/Sort32.md)

#### 掩码操作

- [ResetMask](基础API/矢量计算/掩码操作/ResetMask.md)
- [SetMaskCount](基础API/矢量计算/掩码操作/SetMaskCount.md)
- [SetMaskNorm](基础API/矢量计算/掩码操作/SetMaskNorm.md)
- [SetVectorMask](基础API/矢量计算/掩码操作/SetVectorMask.md)

#### 数据填充

- [Brcb](基础API/矢量计算/数据填充/Brcb.md)
- [CreateVecIndex](基础API/矢量计算/数据填充/CreateVecIndex.md)
- [Duplicate](基础API/矢量计算/数据填充/Duplicate.md)
- [VectorPadding(ISASI)](基础API/矢量计算/数据填充/VectorPadding（ISASI）.md)

#### 数据转换

- [TransDataTo5HD](基础API/矢量计算/数据转换/TransDataTo5HD.md)
- [Transpose](基础API/矢量计算/数据转换/Transpose.md)

#### 比较与选择

- [Compare](基础API/矢量计算/比较与选择/Compare.md)
- [CompareScalar](基础API/矢量计算/比较与选择/CompareScalar.md)
- [Compare（结果存入寄存器）](基础API/矢量计算/比较与选择/Compare（结果存入寄存器）.md)
- [GatherMask](基础API/矢量计算/比较与选择/GatherMask.md)
- [GetCmpMask(ISASI)](基础API/矢量计算/比较与选择/GetCmpMask（ISASI）.md)
- [Select](基础API/矢量计算/比较与选择/Select.md)
- [SetCmpMask(ISASI)](基础API/矢量计算/比较与选择/SetCmpMask（ISASI）.md)

#### 离散与聚合

- [Gather](基础API/矢量计算/离散与聚合/Gather.md)
- [Gatherb(ISASI)](基础API/矢量计算/离散与聚合/Gatherb（ISASI）.md)
- [Scatter(ISASI)](基础API/矢量计算/离散与聚合/Scatter（ISASI）.md)

#### 类型转换

- [Cast](基础API/矢量计算/类型转换/Cast.md)

#### 逻辑计算

- [And](基础API/矢量计算/逻辑计算/And.md)
- [Not](基础API/矢量计算/逻辑计算/Not.md)
- [Or](基础API/矢量计算/逻辑计算/Or.md)
- [ShiftLeft](基础API/矢量计算/逻辑计算/ShiftLeft.md)
- [ShiftRight](基础API/矢量计算/逻辑计算/ShiftRight.md)

#### 量化设置

- [SetDeqScale](基础API/矢量计算/量化设置/SetDeqScale.md)

### 矩阵计算（ISASI）

#### 数据搬运

##### LoadData

- [Load2D](基础API/矩阵计算（ISASI）/数据搬运/LoadData/Load2D.md)
- [Load3D](基础API/矩阵计算（ISASI）/数据搬运/LoadData/Load3D.md)

- [Fixpipe](基础API/矩阵计算（ISASI）/数据搬运/Fixpipe.md)
- [InitConstValue](基础API/矩阵计算（ISASI）/数据搬运/InitConstValue.md)
- [LoadDataUnzip](基础API/矩阵计算（ISASI）/数据搬运/LoadDataUnzip.md)
- [LoadDataWithSparse](基础API/矩阵计算（ISASI）/数据搬运/LoadDataWithSparse.md)
- [LoadDataWithTranspose](基础API/矩阵计算（ISASI）/数据搬运/LoadDataWithTranspose.md)
- [LoadImageToLocal](基础API/矩阵计算（ISASI）/数据搬运/LoadImageToLocal.md)
- [LoadUnzipIndex](基础API/矩阵计算（ISASI）/数据搬运/LoadUnzipIndex.md)
- [SetAippFunctions](基础API/矩阵计算（ISASI）/数据搬运/SetAippFunctions.md)
- [SetFixPipeAddr](基础API/矩阵计算（ISASI）/数据搬运/SetFixPipeAddr.md)
- [SetFixPipeClipRelu](基础API/矩阵计算（ISASI）/数据搬运/SetFixPipeClipRelu.md)
- [SetFixPipeConfig](基础API/矩阵计算（ISASI）/数据搬运/SetFixPipeConfig.md)
- [SetFixpipeNz2ndFlag](基础API/矩阵计算（ISASI）/数据搬运/SetFixpipeNz2ndFlag.md)
- [SetFixpipePreQuantFlag](基础API/矩阵计算（ISASI）/数据搬运/SetFixpipePreQuantFlag.md)
- [SetFmatrix](基础API/矩阵计算（ISASI）/数据搬运/SetFmatrix.md)
- [SetLoadDataBoundary](基础API/矩阵计算（ISASI）/数据搬运/SetLoadDataBoundary.md)
- [SetLoadDataPaddingValue](基础API/矩阵计算（ISASI）/数据搬运/SetLoadDataPaddingValue.md)
- [SetLoadDataRepeat](基础API/矩阵计算（ISASI）/数据搬运/SetLoadDataRepeat.md)

#### 矩阵计算

- [Conv2D（废弃）](基础API/矩阵计算（ISASI）/矩阵计算/Conv2D（废弃）.md)
- [Gemm（废弃）](基础API/矩阵计算（ISASI）/矩阵计算/Gemm（废弃）.md)
- [Mmad](基础API/矩阵计算（ISASI）/矩阵计算/Mmad.md)
- [MmadWithSparse](基础API/矩阵计算（ISASI）/矩阵计算/MmadWithSparse.md)
- [SetHF32Mode](基础API/矩阵计算（ISASI）/矩阵计算/SetHF32Mode.md)
- [SetHF32TransMode](基础API/矩阵计算（ISASI）/矩阵计算/SetHF32TransMode.md)
- [SetMMLayoutTransform](基础API/矩阵计算（ISASI）/矩阵计算/SetMMLayoutTransform.md)

### 系统变量访问

- [GetArchVersion](基础API/系统变量访问/GetArchVersion.md)
- [GetBlockIdx](基础API/系统变量访问/GetBlockIdx.md)
- [GetBlockNum](基础API/系统变量访问/GetBlockNum.md)
- [GetDataBlockSizeInBytes](基础API/系统变量访问/GetDataBlockSizeInBytes.md)
- [GetProgramCounter(ISASI)](基础API/系统变量访问/GetProgramCounter（ISASI）.md)
- [GetSubBlockIdx(ISASI)](基础API/系统变量访问/GetSubBlockIdx（ISASI）.md)
- [GetSubBlockNum(ISASI)](基础API/系统变量访问/GetSubBlockNum（ISASI）.md)
- [GetSystemCycle(ISASI)](基础API/系统变量访问/GetSystemCycle（ISASI）.md)
- [InitSocState](基础API/系统变量访问/InitSocState.md)

### 缓存控制

- [DataCacheCleanAndInvalid](基础API/缓存控制/DataCacheCleanAndInvalid.md)
- [DataCachePreload](基础API/缓存控制/DataCachePreload.md)
- [GetICachePreloadStatus(ISASI)](基础API/缓存控制/GetICachePreloadStatus（ISASI）.md)
- [ICachePreLoad(ISASI)](基础API/缓存控制/ICachePreLoad（ISASI）.md)

### 调试接口

#### CPU孪生调试

- [GmAlloc](基础API/调试接口/CPU孪生调试/GmAlloc.md)
- [GmFree](基础API/调试接口/CPU孪生调试/GmFree.md)
- [ICPU_RUN_KF](基础API/调试接口/CPU孪生调试/ICPU_RUN_KF.md)
- [ICPU_SET_TILING_KEY](基础API/调试接口/CPU孪生调试/ICPU_SET_TILING_KEY.md)
- [SetKernelMode](基础API/调试接口/CPU孪生调试/SetKernelMode.md)

#### 上板打印

- [DumpAccChkPoint](基础API/调试接口/上板打印/DumpAccChkPoint.md)
- [DumpTensor](基础API/调试接口/上板打印/DumpTensor.md)
- [PrintTimeStamp](基础API/调试接口/上板打印/PrintTimeStamp.md)
- [printf](基础API/调试接口/上板打印/printf.md)

#### 异常检测

- [CheckLocalMemoryIA(ISASI)](基础API/调试接口/异常检测/CheckLocalMemoryIA（ISASI）.md)
- [Trap](基础API/调试接口/异常检测/Trap.md)
- [ascendc_assert](基础API/调试接口/异常检测/ascendc_assert.md)
- [assert](基础API/调试接口/异常检测/assert.md)

#### 性能统计

- [MetricsProfStart](基础API/调试接口/性能统计/MetricsProfStart.md)
- [MetricsProfStop](基础API/调试接口/性能统计/MetricsProfStop.md)
- [TRACE_START](基础API/调试接口/性能统计/TRACE_START.md)
- [TRACE_STOP](基础API/调试接口/性能统计/TRACE_STOP.md)

### 资源管理

#### Pipe和Que框架

##### TBuf

- [Get](基础API/资源管理/Pipe和Que框架/TBuf/Get.md)
- [GetWithOffset](基础API/资源管理/Pipe和Que框架/TBuf/GetWithOffset.md)
- [TBuf构造函数](基础API/资源管理/Pipe和Que框架/TBuf/TBuf构造函数.md)
- [TBuf简介](基础API/资源管理/Pipe和Que框架/TBuf/TBuf简介.md)

##### TBufPool

- [InitBufPool](基础API/资源管理/Pipe和Que框架/TBufPool/InitBufPool.md)
- [InitBuffer](基础API/资源管理/Pipe和Que框架/TBufPool/InitBuffer.md)
- [Reset](基础API/资源管理/Pipe和Que框架/TBufPool/Reset.md)
- [TBufPool构造函数](基础API/资源管理/Pipe和Que框架/TBufPool/TBufPool构造函数.md)
- [TBufPool简介](基础API/资源管理/Pipe和Que框架/TBufPool/TBufPool简介.md)

##### TPipe

- [AllocEventID](基础API/资源管理/Pipe和Que框架/TPipe/AllocEventID.md)
- [Destroy](基础API/资源管理/Pipe和Que框架/TPipe/Destroy.md)
- [FetchEventID](基础API/资源管理/Pipe和Que框架/TPipe/FetchEventID.md)
- [GetBaseAddr](基础API/资源管理/Pipe和Que框架/TPipe/GetBaseAddr.md)
- [Init](基础API/资源管理/Pipe和Que框架/TPipe/Init.md)
- [InitBufPool](基础API/资源管理/Pipe和Que框架/TPipe/InitBufPool.md)
- [InitBuffer](基础API/资源管理/Pipe和Que框架/TPipe/InitBuffer.md)
- [InitSpmBuffer](基础API/资源管理/Pipe和Que框架/TPipe/InitSpmBuffer.md)
- [ReadSpmBuffer](基础API/资源管理/Pipe和Que框架/TPipe/ReadSpmBuffer.md)
- [ReleaseEventID](基础API/资源管理/Pipe和Que框架/TPipe/ReleaseEventID.md)
- [Reset](基础API/资源管理/Pipe和Que框架/TPipe/Reset.md)
- [TPipe构造函数](基础API/资源管理/Pipe和Que框架/TPipe/TPipe构造函数.md)
- [WriteSpmBuffer](基础API/资源管理/Pipe和Que框架/TPipe/WriteSpmBuffer.md)

##### TQue

- [AllocTensor](基础API/资源管理/Pipe和Que框架/TQue/AllocTensor.md)
- [DeQue](基础API/资源管理/Pipe和Que框架/TQue/DeQue.md)
- [EnQue](基础API/资源管理/Pipe和Que框架/TQue/EnQue.md)
- [FreeTensor](基础API/资源管理/Pipe和Que框架/TQue/FreeTensor.md)
- [GetTensorCountInQue](基础API/资源管理/Pipe和Que框架/TQue/GetTensorCountInQue.md)
- [HasIdleBuffer](基础API/资源管理/Pipe和Que框架/TQue/HasIdleBuffer.md)
- [HasTensorInQue](基础API/资源管理/Pipe和Que框架/TQue/HasTensorInQue.md)
- [TQue简介](基础API/资源管理/Pipe和Que框架/TQue/TQue简介.md)
- [VacantInQue](基础API/资源管理/Pipe和Que框架/TQue/VacantInQue.md)

##### TQueBind

- [AllocTensor](基础API/资源管理/Pipe和Que框架/TQueBind/AllocTensor.md)
- [DeQue](基础API/资源管理/Pipe和Que框架/TQueBind/DeQue.md)
- [EnQue](基础API/资源管理/Pipe和Que框架/TQueBind/EnQue.md)
- [FreeAllEvent](基础API/资源管理/Pipe和Que框架/TQueBind/FreeAllEvent.md)
- [FreeTensor](基础API/资源管理/Pipe和Que框架/TQueBind/FreeTensor.md)
- [GetTensorCountInQue](基础API/资源管理/Pipe和Que框架/TQueBind/GetTensorCountInQue.md)
- [HasIdleBuffer](基础API/资源管理/Pipe和Que框架/TQueBind/HasIdleBuffer.md)
- [HasTensorInQue](基础API/资源管理/Pipe和Que框架/TQueBind/HasTensorInQue.md)
- [InitBufHandle](基础API/资源管理/Pipe和Que框架/TQueBind/InitBufHandle.md)
- [InitStartBufHandle](基础API/资源管理/Pipe和Que框架/TQueBind/InitStartBufHandle.md)
- [TQueBind构造函数](基础API/资源管理/Pipe和Que框架/TQueBind/TQueBind构造函数.md)
- [TQueBind简介](基础API/资源管理/Pipe和Que框架/TQueBind/TQueBind简介.md)
- [VacantInQue](基础API/资源管理/Pipe和Que框架/TQueBind/VacantInQue.md)

##### 自定义TBufPool

- [EXTERN_IMPL_BUFPOOL宏](基础API/资源管理/Pipe和Que框架/自定义TBufPool/EXTERN_IMPL_BUFPOOL宏.md)
- [GetBufHandle](基础API/资源管理/Pipe和Que框架/自定义TBufPool/GetBufHandle.md)
- [GetCurAddr](基础API/资源管理/Pipe和Que框架/自定义TBufPool/GetCurAddr.md)
- [GetCurBufSize](基础API/资源管理/Pipe和Que框架/自定义TBufPool/GetCurBufSize.md)
- [Init](基础API/资源管理/Pipe和Que框架/自定义TBufPool/Init.md)
- [Reset](基础API/资源管理/Pipe和Que框架/自定义TBufPool/Reset.md)
- [SetCurAddr](基础API/资源管理/Pipe和Que框架/自定义TBufPool/SetCurAddr.md)
- [SetCurBufSize](基础API/资源管理/Pipe和Que框架/自定义TBufPool/SetCurBufSize.md)

- [GetTPipePtr](基础API/资源管理/Pipe和Que框架/GetTPipePtr.md)

#### 临时空间管理

##### workspace

- [GetSysWorkSpacePtr](基础API/资源管理/临时空间管理/workspace/GetSysWorkSpacePtr.md)
- [GetUserWorkspace](基础API/资源管理/临时空间管理/workspace/GetUserWorkspace.md)
- [SetSysWorkSpace](基础API/资源管理/临时空间管理/workspace/SetSysWorkSpace.md)

#### 内存管理

##### LocalMemAllocator

- [Alloc](基础API/资源管理/内存管理/LocalMemAllocator/Alloc.md)
- [GetCurAddr](基础API/资源管理/内存管理/LocalMemAllocator/GetCurAddr.md)
- [LocalMemAllocator构造函数](基础API/资源管理/内存管理/LocalMemAllocator/LocalMemAllocator构造函数.md)
- [LocalMemAllocator简介](基础API/资源管理/内存管理/LocalMemAllocator/LocalMemAllocator简介.md)

## 基础数据结构

### Coordinate

- [Coordinate简介](基础数据结构/Coordinate/Coordinate简介.md)
- [Crd2Idx](基础数据结构/Coordinate/Crd2Idx.md)
- [MakeCoord](基础数据结构/Coordinate/MakeCoord.md)

### GlobalTensor

- [GetPhyAddr](基础数据结构/GlobalTensor/GetPhyAddr.md)
- [GetShapeInfo](基础数据结构/GlobalTensor/GetShapeInfo.md)
- [GetSize](基础数据结构/GlobalTensor/GetSize.md)
- [GetValue](基础数据结构/GlobalTensor/GetValue.md)
- [GlobalTensor构造函数](基础数据结构/GlobalTensor/GlobalTensor构造函数.md)
- [GlobalTensor简介](基础数据结构/GlobalTensor/GlobalTensor简介.md)
- [SetGlobalBuffer](基础数据结构/GlobalTensor/SetGlobalBuffer.md)
- [SetL2CacheHint](基础数据结构/GlobalTensor/SetL2CacheHint.md)
- [SetShapeInfo](基础数据结构/GlobalTensor/SetShapeInfo.md)
- [SetValue](基础数据结构/GlobalTensor/SetValue.md)
- [operator()](基础数据结构/GlobalTensor/operator（）.md)
- [operator[]](基础数据结构/GlobalTensor/operator[].md)

### Layout

- [GetShape](基础数据结构/Layout/GetShape.md)
- [GetStride](基础数据结构/Layout/GetStride.md)
- [Layout构造函数](基础数据结构/Layout/Layout构造函数.md)
- [Layout简介](基础数据结构/Layout/Layout简介.md)
- [MakeLayout](基础数据结构/Layout/MakeLayout.md)
- [MakeShape](基础数据结构/Layout/MakeShape.md)
- [MakeStride](基础数据结构/Layout/MakeStride.md)
- [is_layout](基础数据结构/Layout/is_layout.md)
- [layout](基础数据结构/Layout/layout.md)
- [运算符重载](基础数据结构/Layout/运算符重载.md)

### LocalTensor

- [GetLength](基础数据结构/LocalTensor/GetLength.md)
- [GetPhyAddr](基础数据结构/LocalTensor/GetPhyAddr.md)
- [GetPosition](基础数据结构/LocalTensor/GetPosition.md)
- [GetShapeInfo](基础数据结构/LocalTensor/GetShapeInfo.md)
- [GetSize](基础数据结构/LocalTensor/GetSize.md)
- [GetUserTag](基础数据结构/LocalTensor/GetUserTag.md)
- [GetValue](基础数据结构/LocalTensor/GetValue.md)
- [LocalTensor构造函数](基础数据结构/LocalTensor/LocalTensor构造函数.md)
- [LocalTensor简介](基础数据结构/LocalTensor/LocalTensor简介.md)
- [Print](基础数据结构/LocalTensor/Print.md)
- [ReinterpretCast](基础数据结构/LocalTensor/ReinterpretCast.md)
- [SetAddrWithOffset](基础数据结构/LocalTensor/SetAddrWithOffset.md)
- [SetBufferLen](基础数据结构/LocalTensor/SetBufferLen.md)
- [SetShapeInfo](基础数据结构/LocalTensor/SetShapeInfo.md)
- [SetSize](基础数据结构/LocalTensor/SetSize.md)
- [SetUserTag](基础数据结构/LocalTensor/SetUserTag.md)
- [SetValue](基础数据结构/LocalTensor/SetValue.md)
- [ToFile](基础数据结构/LocalTensor/ToFile.md)
- [operator()](基础数据结构/LocalTensor/operator（）.md)
- [operator[]](基础数据结构/LocalTensor/operator[].md)

### TensorTrait

- [GetLayout](基础数据结构/TensorTrait/GetLayout.md)
- [MakeTensorTrait](基础数据结构/TensorTrait/MakeTensorTrait.md)
- [SetLayout](基础数据结构/TensorTrait/SetLayout.md)
- [TensorTrait构造函数](基础数据结构/TensorTrait/TensorTrait构造函数.md)
- [TensorTrait简介](基础数据结构/TensorTrait/TensorTrait简介.md)
- [is_tensorTrait](基础数据结构/TensorTrait/is_tensorTrait.md)
- [更多样例](基础数据结构/TensorTrait/更多样例.md)

## 高阶API

### HCCL通信类

#### HCCL Context

- [GetHcclContext](高阶API/HCCL通信类/HCCL_Context/GetHcclContext.md)
- [SetHcclContext](高阶API/HCCL通信类/HCCL_Context/SetHcclContext.md)

#### HCCL Kernel侧接口

- [AllGather](高阶API/HCCL通信类/HCCL_Kernel侧接口/AllGather.md)
- [AllReduce](高阶API/HCCL通信类/HCCL_Kernel侧接口/AllReduce.md)
- [AlltoAll](高阶API/HCCL通信类/HCCL_Kernel侧接口/AlltoAll.md)
- [AlltoAllV](高阶API/HCCL通信类/HCCL_Kernel侧接口/AlltoAllV.md)
- [BatchWrite](高阶API/HCCL通信类/HCCL_Kernel侧接口/BatchWrite.md)
- [Commit](高阶API/HCCL通信类/HCCL_Kernel侧接口/Commit.md)
- [Finalize](高阶API/HCCL通信类/HCCL_Kernel侧接口/Finalize.md)
- [GetQueueNum](高阶API/HCCL通信类/HCCL_Kernel侧接口/GetQueueNum.md)
- [GetRankDim](高阶API/HCCL通信类/HCCL_Kernel侧接口/GetRankDim.md)
- [GetRankId](高阶API/HCCL通信类/HCCL_Kernel侧接口/GetRankId.md)
- [GetWindowsInAddr](高阶API/HCCL通信类/HCCL_Kernel侧接口/GetWindowsInAddr.md)
- [GetWindowsOutAddr](高阶API/HCCL通信类/HCCL_Kernel侧接口/GetWindowsOutAddr.md)
- [HCCL使用说明](高阶API/HCCL通信类/HCCL_Kernel侧接口/HCCL使用说明.md)
- [HCCL模板参数](高阶API/HCCL通信类/HCCL_Kernel侧接口/HCCL模板参数.md)
- [InitV2](高阶API/HCCL通信类/HCCL_Kernel侧接口/InitV2.md)
- [Init（废弃）](高阶API/HCCL通信类/HCCL_Kernel侧接口/Init（废弃）.md)
- [InterHcclGroupSync](高阶API/HCCL通信类/HCCL_Kernel侧接口/InterHcclGroupSync.md)
- [Iterate](高阶API/HCCL通信类/HCCL_Kernel侧接口/Iterate.md)
- [Query](高阶API/HCCL通信类/HCCL_Kernel侧接口/Query.md)
- [QueueBarrier](高阶API/HCCL通信类/HCCL_Kernel侧接口/QueueBarrier.md)
- [ReduceScatter](高阶API/HCCL通信类/HCCL_Kernel侧接口/ReduceScatter.md)
- [SetCcTilingV2](高阶API/HCCL通信类/HCCL_Kernel侧接口/SetCcTilingV2.md)
- [SetCcTiling（废弃）](高阶API/HCCL通信类/HCCL_Kernel侧接口/SetCcTiling（废弃）.md)
- [Wait](高阶API/HCCL通信类/HCCL_Kernel侧接口/Wait.md)

#### HCCL Tiling侧接口

- [GetTiling](高阶API/HCCL通信类/HCCL_Tiling侧接口/GetTiling.md)
- [HCCL Tiling使用说明](高阶API/HCCL通信类/HCCL_Tiling侧接口/HCCL_Tiling使用说明.md)
- [HCCL Tiling构造函数](高阶API/HCCL通信类/HCCL_Tiling侧接口/HCCL_Tiling构造函数.md)
- [SetAlgConfig](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetAlgConfig.md)
- [SetCommBlockNum](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetCommBlockNum.md)
- [SetCommEngine](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetCommEngine.md)
- [SetDebugMode](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetDebugMode.md)
- [SetGroupName](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetGroupName.md)
- [SetOpType](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetOpType.md)
- [SetQueueNum](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetQueueNum.md)
- [SetReduceType](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetReduceType.md)
- [SetSkipBufferWindowCopy](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetSkipBufferWindowCopy.md)
- [SetSkipLocalRankCopy](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetSkipLocalRankCopy.md)
- [SetStepSize](高阶API/HCCL通信类/HCCL_Tiling侧接口/SetStepSize.md)
- [TilingData结构体](高阶API/HCCL通信类/HCCL_Tiling侧接口/TilingData结构体.md)
- [v1版本TilingData（废弃）](高阶API/HCCL通信类/HCCL_Tiling侧接口/v1版本TilingData（废弃）.md)
- [v2版本TilingData（废弃）](高阶API/HCCL通信类/HCCL_Tiling侧接口/v2版本TilingData（废弃）.md)

### 卷积计算

#### Conv3D

##### Conv3D Kernel侧接口

- [Conv3D使用说明](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/Conv3D使用说明.md)
- [Conv3D模板参数](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/Conv3D模板参数.md)
- [End](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/End.md)
- [Init](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/Init.md)
- [IterateAll](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/IterateAll.md)
- [SetBias](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/SetBias.md)
- [SetInput](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/SetInput.md)
- [SetInputStartPosition](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/SetInputStartPosition.md)
- [SetSingleOutputShape](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/SetSingleOutputShape.md)
- [SetWeight](高阶API/卷积计算/Conv3D/Conv3D_Kernel侧接口/SetWeight.md)

##### Conv3D Tiling侧接口

- [Conv3D Tiling使用说明](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/Conv3D_Tiling使用说明.md)
- [Conv3D Tiling构造函数](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/Conv3D_Tiling构造函数.md)
- [GetTiling](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/GetTiling.md)
- [SetBiasType](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetBiasType.md)
- [SetDilation](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetDilation.md)
- [SetGroups](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetGroups.md)
- [SetInputType](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetInputType.md)
- [SetOrgInputShape](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetOrgInputShape.md)
- [SetOrgWeightShape](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetOrgWeightShape.md)
- [SetOutputType](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetOutputType.md)
- [SetPadding](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetPadding.md)
- [SetSingleOutputShape](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetSingleOutputShape.md)
- [SetSingleWeightShape](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetSingleWeightShape.md)
- [SetStride](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetStride.md)
- [SetWeightType](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/SetWeightType.md)
- [TConv3DApiTiling结构体](高阶API/卷积计算/Conv3D/Conv3D_Tiling侧接口/TConv3DApiTiling结构体.md)

#### Conv3DBackpropFilter

##### Conv3DBackpropFilter Kernel侧接口

- [Conv3DBackpropFilter使用说明](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/Conv3DBackpropFilter使用说明.md)
- [End](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/End.md)
- [GetTensorC](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/GetTensorC.md)
- [Init](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/Init.md)
- [Iterate](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/Iterate.md)
- [SetGradOutput](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/SetGradOutput.md)
- [SetInput](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/SetInput.md)
- [SetSingleShape](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/SetSingleShape.md)
- [SetStartPosition](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Kernel侧接口/SetStartPosition.md)

##### Conv3DBackpropFilter Tiling侧接口

- [Conv3DBackpropFilter Tiling使用说明](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/Conv3DBackpropFilter_Tiling使用说明.md)
- [Conv3DBackpropFilter Tiling构造函数](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/Conv3DBackpropFilter_Tiling构造函数.md)
- [GetTiling](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/GetTiling.md)
- [SetDilation](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetDilation.md)
- [SetGradOutputShape](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetGradOutputShape.md)
- [SetGradOutputType](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetGradOutputType.md)
- [SetInputShape](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetInputShape.md)
- [SetInputType](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetInputType.md)
- [SetPadding](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetPadding.md)
- [SetStride](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetStride.md)
- [SetWeightShape](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetWeightShape.md)
- [SetWeightType](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/SetWeightType.md)
- [TConv3DBpFilterTiling结构体](高阶API/卷积计算/Conv3DBackpropFilter/Conv3DBackpropFilter_Tiling侧接口/TConv3DBpFilterTiling结构体.md)

#### Conv3DBackpropInput

##### Conv3DBackpropInput Kernel侧接口

- [Conv3DBackpropInput使用说明](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/Conv3DBackpropInput使用说明.md)
- [End](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/End.md)
- [GetTensorC](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/GetTensorC.md)
- [Init](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/Init.md)
- [Iterate](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/Iterate.md)
- [SetGradOutput](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/SetGradOutput.md)
- [SetSingleShape](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/SetSingleShape.md)
- [SetStartPosition](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/SetStartPosition.md)
- [SetWeight](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Kernel侧接口/SetWeight.md)

##### Conv3DBackpropInput Tiling侧接口

- [Conv3DBackpropInput Tiling使用说明](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/Conv3DBackpropInput_Tiling使用说明.md)
- [Conv3DBackpropInput Tiling构造函数](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/Conv3DBackpropInput_Tiling构造函数.md)
- [GetTiling](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/GetTiling.md)
- [SetDilation](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetDilation.md)
- [SetGradOutputShape](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetGradOutputShape.md)
- [SetGradOutputType](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetGradOutputType.md)
- [SetInputShape](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetInputShape.md)
- [SetInputType](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetInputType.md)
- [SetOutputPadding](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetOutputPadding.md)
- [SetPadding](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetPadding.md)
- [SetStride](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetStride.md)
- [SetWeightShape](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetWeightShape.md)
- [SetWeightType](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/SetWeightType.md)
- [TConv3DBackpropInputTiling结构体](高阶API/卷积计算/Conv3DBackpropInput/Conv3DBackpropInput_Tiling侧接口/TConv3DBackpropInputTiling结构体.md)

### 张量变换

- [Broadcast](高阶API/张量变换/Broadcast.md)
- [Fill](高阶API/张量变换/Fill.md)
- [GetBroadCastMaxMinTmpSize](高阶API/张量变换/GetBroadCastMaxMinTmpSize.md)
- [GetTransDataMaxMinTmpSize](高阶API/张量变换/GetTransDataMaxMinTmpSize.md)
- [Pad](高阶API/张量变换/Pad.md)
- [Pad Tiling](高阶API/张量变换/Pad_Tiling.md)
- [TransData](高阶API/张量变换/TransData.md)
- [Transpose](高阶API/张量变换/Transpose.md)
- [Transpose Tiling](高阶API/张量变换/Transpose_Tiling.md)
- [UnPad](高阶API/张量变换/UnPad.md)
- [UnPad Tiling](高阶API/张量变换/UnPad_Tiling.md)

### 归一化操作

- [BatchNorm](高阶API/归一化操作/BatchNorm.md)
- [BatchNorm Tiling](高阶API/归一化操作/BatchNorm_Tiling.md)
- [DeepNorm](高阶API/归一化操作/DeepNorm.md)
- [DeepNorm Tiling](高阶API/归一化操作/DeepNorm_Tiling.md)
- [GroupNorm](高阶API/归一化操作/GroupNorm.md)
- [GroupNorm Tiling](高阶API/归一化操作/GroupNorm_Tiling.md)
- [LayerNorm](高阶API/归一化操作/LayerNorm.md)
- [LayerNorm Tiling](高阶API/归一化操作/LayerNorm_Tiling.md)
- [LayerNormGrad](高阶API/归一化操作/LayerNormGrad.md)
- [LayerNormGrad Tiling](高阶API/归一化操作/LayerNormGrad_Tiling.md)
- [LayerNormGradBeta](高阶API/归一化操作/LayerNormGradBeta.md)
- [LayerNormGradBeta Tiling](高阶API/归一化操作/LayerNormGradBeta_Tiling.md)
- [Normalize](高阶API/归一化操作/Normalize.md)
- [Normalize Tiling](高阶API/归一化操作/Normalize_Tiling.md)
- [RmsNorm](高阶API/归一化操作/RmsNorm.md)
- [RmsNorm Tiling](高阶API/归一化操作/RmsNorm_Tiling.md)
- [WelfordFinalize](高阶API/归一化操作/WelfordFinalize.md)
- [WelfordFinalize Tiling](高阶API/归一化操作/WelfordFinalize_Tiling.md)
- [WelfordUpdate](高阶API/归一化操作/WelfordUpdate.md)
- [WelfordUpdate Tiling](高阶API/归一化操作/WelfordUpdate_Tiling.md)

### 归约操作

#### Mean接口

- [GetMeanMaxMinTmpSize](高阶API/归约操作/Mean接口/GetMeanMaxMinTmpSize.md)
- [GetMeanTmpBufferFactorSize](高阶API/归约操作/Mean接口/GetMeanTmpBufferFactorSize.md)
- [Mean](高阶API/归约操作/Mean接口/Mean.md)

#### ReduceAll接口

- [GetReduceAllMaxMinTmpSize](高阶API/归约操作/ReduceAll接口/GetReduceAllMaxMinTmpSize.md)
- [ReduceAll](高阶API/归约操作/ReduceAll接口/ReduceAll.md)

#### ReduceAny接口

- [GetReduceAnyMaxMinTmpSize](高阶API/归约操作/ReduceAny接口/GetReduceAnyMaxMinTmpSize.md)
- [ReduceAny](高阶API/归约操作/ReduceAny接口/ReduceAny.md)

#### ReduceMax接口

- [GetReduceMaxMaxMinTmpSize](高阶API/归约操作/ReduceMax接口/GetReduceMaxMaxMinTmpSize.md)
- [ReduceMax](高阶API/归约操作/ReduceMax接口/ReduceMax.md)

#### ReduceMean接口

- [GetReduceMeanMaxMinTmpSize](高阶API/归约操作/ReduceMean接口/GetReduceMeanMaxMinTmpSize.md)
- [ReduceMean](高阶API/归约操作/ReduceMean接口/ReduceMean.md)

#### ReduceMin接口

- [GetReduceMinMaxMinTmpSize](高阶API/归约操作/ReduceMin接口/GetReduceMinMaxMinTmpSize.md)
- [ReduceMin](高阶API/归约操作/ReduceMin接口/ReduceMin.md)

#### ReduceProd接口

- [GetReduceProdMaxMinTmpSize](高阶API/归约操作/ReduceProd接口/GetReduceProdMaxMinTmpSize.md)
- [ReduceProd](高阶API/归约操作/ReduceProd接口/ReduceProd.md)

#### ReduceSum接口

- [GetReduceSumMaxMinTmpSize](高阶API/归约操作/ReduceSum接口/GetReduceSumMaxMinTmpSize.md)
- [ReduceSum](高阶API/归约操作/ReduceSum接口/ReduceSum.md)

#### ReduceXorSum接口

- [GetReduceXorSumMaxMinTmpSize](高阶API/归约操作/ReduceXorSum接口/GetReduceXorSumMaxMinTmpSize.md)
- [ReduceXorSum](高阶API/归约操作/ReduceXorSum接口/ReduceXorSum.md)

#### Sum接口

- [GetSumMaxMinTmpSize](高阶API/归约操作/Sum接口/GetSumMaxMinTmpSize.md)
- [Sum](高阶API/归约操作/Sum接口/Sum.md)

### 排序操作

- [Concat](高阶API/排序操作/Concat.md)
- [Extract](高阶API/排序操作/Extract.md)
- [GetConcatTmpSize](高阶API/排序操作/GetConcatTmpSize.md)
- [GetSortLen](高阶API/排序操作/GetSortLen.md)
- [GetSortOffset](高阶API/排序操作/GetSortOffset.md)
- [GetSortTmpSize](高阶API/排序操作/GetSortTmpSize.md)
- [MrgSort](高阶API/排序操作/MrgSort.md)
- [Sort](高阶API/排序操作/Sort.md)
- [TopK](高阶API/排序操作/TopK.md)
- [TopK Tiling](高阶API/排序操作/TopK_Tiling.md)

### 数学计算

#### Acosh接口

- [Acosh](高阶API/数学计算/Acosh接口/Acosh.md)
- [GetAcoshMaxMinTmpSize](高阶API/数学计算/Acosh接口/GetAcoshMaxMinTmpSize.md)
- [GetAcoshTmpBufferFactorSize](高阶API/数学计算/Acosh接口/GetAcoshTmpBufferFactorSize.md)

#### Acos接口

- [Acos](高阶API/数学计算/Acos接口/Acos.md)
- [GetAcosMaxMinTmpSize](高阶API/数学计算/Acos接口/GetAcosMaxMinTmpSize.md)
- [GetAcosTmpBufferFactorSize](高阶API/数学计算/Acos接口/GetAcosTmpBufferFactorSize.md)

#### Asinh接口

- [Asinh](高阶API/数学计算/Asinh接口/Asinh.md)
- [GetAsinhMaxMinTmpSize](高阶API/数学计算/Asinh接口/GetAsinhMaxMinTmpSize.md)
- [GetAsinhTmpBufferFactorSize](高阶API/数学计算/Asinh接口/GetAsinhTmpBufferFactorSize.md)

#### Asin接口

- [Asin](高阶API/数学计算/Asin接口/Asin.md)
- [GetAsinMaxMinTmpSize](高阶API/数学计算/Asin接口/GetAsinMaxMinTmpSize.md)
- [GetAsinTmpBufferFactorSize](高阶API/数学计算/Asin接口/GetAsinTmpBufferFactorSize.md)

#### Atanh接口

- [Atanh](高阶API/数学计算/Atanh接口/Atanh.md)
- [GetAtanhMaxMinTmpSize](高阶API/数学计算/Atanh接口/GetAtanhMaxMinTmpSize.md)
- [GetAtanhTmpBufferFactorSize](高阶API/数学计算/Atanh接口/GetAtanhTmpBufferFactorSize.md)

#### Atan接口

- [Atan](高阶API/数学计算/Atan接口/Atan.md)
- [GetAtanMaxMinTmpSize](高阶API/数学计算/Atan接口/GetAtanMaxMinTmpSize.md)
- [GetAtanTmpBufferFactorSize](高阶API/数学计算/Atan接口/GetAtanTmpBufferFactorSize.md)

#### Axpy接口

- [Axpy](高阶API/数学计算/Axpy接口/Axpy.md)
- [GetAxpyMaxMinTmpSize](高阶API/数学计算/Axpy接口/GetAxpyMaxMinTmpSize.md)
- [GetAxpyTmpBufferFactorSize](高阶API/数学计算/Axpy接口/GetAxpyTmpBufferFactorSize.md)

#### Ceil接口

- [Ceil](高阶API/数学计算/Ceil接口/Ceil.md)
- [GetCeilMaxMinTmpSize](高阶API/数学计算/Ceil接口/GetCeilMaxMinTmpSize.md)
- [GetCeilTmpBufferFactorSize](高阶API/数学计算/Ceil接口/GetCeilTmpBufferFactorSize.md)

#### Clamp接口

- [ClampMax](高阶API/数学计算/Clamp接口/ClampMax.md)
- [ClampMin](高阶API/数学计算/Clamp接口/ClampMin.md)
- [GetClampMaxMinTmpSize](高阶API/数学计算/Clamp接口/GetClampMaxMinTmpSize.md)
- [GetClampTmpBufferFactorSize](高阶API/数学计算/Clamp接口/GetClampTmpBufferFactorSize.md)

#### Cosh接口

- [Cosh](高阶API/数学计算/Cosh接口/Cosh.md)
- [GetCoshMaxMinTmpSize](高阶API/数学计算/Cosh接口/GetCoshMaxMinTmpSize.md)
- [GetCoshTmpBufferFactorSize](高阶API/数学计算/Cosh接口/GetCoshTmpBufferFactorSize.md)

#### Cos接口

- [Cos](高阶API/数学计算/Cos接口/Cos.md)
- [GetCosMaxMinTmpSize](高阶API/数学计算/Cos接口/GetCosMaxMinTmpSize.md)
- [GetCosTmpBufferFactorSize](高阶API/数学计算/Cos接口/GetCosTmpBufferFactorSize.md)

#### CumSum接口

- [CumSum](高阶API/数学计算/CumSum接口/CumSum.md)
- [GetCumSumMaxMinTmpSize](高阶API/数学计算/CumSum接口/GetCumSumMaxMinTmpSize.md)

#### Digamma接口

- [Digamma](高阶API/数学计算/Digamma接口/Digamma.md)
- [GetDigammaMaxMinTmpSize](高阶API/数学计算/Digamma接口/GetDigammaMaxMinTmpSize.md)
- [GetDigammaTmpBufferFactorSize](高阶API/数学计算/Digamma接口/GetDigammaTmpBufferFactorSize.md)

#### Erfc接口

- [Erfc](高阶API/数学计算/Erfc接口/Erfc.md)
- [GetErfcMaxMinTmpSize](高阶API/数学计算/Erfc接口/GetErfcMaxMinTmpSize.md)
- [GetErfcTmpBufferFactorSize](高阶API/数学计算/Erfc接口/GetErfcTmpBufferFactorSize.md)

#### Erf接口

- [Erf](高阶API/数学计算/Erf接口/Erf.md)
- [GetErfMaxMinTmpSize](高阶API/数学计算/Erf接口/GetErfMaxMinTmpSize.md)
- [GetErfTmpBufferFactorSize](高阶API/数学计算/Erf接口/GetErfTmpBufferFactorSize.md)

#### Exp接口

- [Exp](高阶API/数学计算/Exp接口/Exp.md)
- [GetExpMaxMinTmpSize](高阶API/数学计算/Exp接口/GetExpMaxMinTmpSize.md)
- [GetExpTmpBufferFactorSize](高阶API/数学计算/Exp接口/GetExpTmpBufferFactorSize.md)

#### Floor接口

- [Floor](高阶API/数学计算/Floor接口/Floor.md)
- [GetFloorMaxMinTmpSize](高阶API/数学计算/Floor接口/GetFloorMaxMinTmpSize.md)
- [GetFloorTmpBufferFactorSize](高阶API/数学计算/Floor接口/GetFloorTmpBufferFactorSize.md)

#### Fmod接口

- [Fmod](高阶API/数学计算/Fmod接口/Fmod.md)
- [GetFmodMaxMinTmpSize](高阶API/数学计算/Fmod接口/GetFmodMaxMinTmpSize.md)
- [GetFmodTmpBufferFactorSize](高阶API/数学计算/Fmod接口/GetFmodTmpBufferFactorSize.md)

#### Frac接口

- [Frac](高阶API/数学计算/Frac接口/Frac.md)
- [GetFracMaxMinTmpSize](高阶API/数学计算/Frac接口/GetFracMaxMinTmpSize.md)
- [GetFracTmpBufferFactorSize](高阶API/数学计算/Frac接口/GetFracTmpBufferFactorSize.md)

#### Lgamma接口

- [GetLgammaMaxMinTmpSize](高阶API/数学计算/Lgamma接口/GetLgammaMaxMinTmpSize.md)
- [GetLgammaTmpBufferFactorSize](高阶API/数学计算/Lgamma接口/GetLgammaTmpBufferFactorSize.md)
- [Lgamma](高阶API/数学计算/Lgamma接口/Lgamma.md)

#### Log接口

- [GetLogMaxMinTmpSize](高阶API/数学计算/Log接口/GetLogMaxMinTmpSize.md)
- [GetLogTmpBufferFactorSize](高阶API/数学计算/Log接口/GetLogTmpBufferFactorSize.md)
- [Log](高阶API/数学计算/Log接口/Log.md)

#### Power接口

- [GetPowerMaxMinTmpSize](高阶API/数学计算/Power接口/GetPowerMaxMinTmpSize.md)
- [GetPowerTmpBufferFactorSize](高阶API/数学计算/Power接口/GetPowerTmpBufferFactorSize.md)
- [Power](高阶API/数学计算/Power接口/Power.md)

#### Round接口

- [GetRoundMaxMinTmpSize](高阶API/数学计算/Round接口/GetRoundMaxMinTmpSize.md)
- [GetRoundTmpBufferFactorSize](高阶API/数学计算/Round接口/GetRoundTmpBufferFactorSize.md)
- [Round](高阶API/数学计算/Round接口/Round.md)

#### Sign接口

- [GetSignMaxMinTmpSize](高阶API/数学计算/Sign接口/GetSignMaxMinTmpSize.md)
- [GetSignTmpBufferFactorSize](高阶API/数学计算/Sign接口/GetSignTmpBufferFactorSize.md)
- [Sign](高阶API/数学计算/Sign接口/Sign.md)

#### Sinh接口

- [GetSinhMaxMinTmpSize](高阶API/数学计算/Sinh接口/GetSinhMaxMinTmpSize.md)
- [GetSinhTmpBufferFactorSize](高阶API/数学计算/Sinh接口/GetSinhTmpBufferFactorSize.md)
- [Sinh](高阶API/数学计算/Sinh接口/Sinh.md)

#### Sin接口

- [GetSinMaxMinTmpSize](高阶API/数学计算/Sin接口/GetSinMaxMinTmpSize.md)
- [GetSinTmpBufferFactorSize](高阶API/数学计算/Sin接口/GetSinTmpBufferFactorSize.md)
- [Sin](高阶API/数学计算/Sin接口/Sin.md)

#### Tanh接口

- [GetTanhMaxMinTmpSize](高阶API/数学计算/Tanh接口/GetTanhMaxMinTmpSize.md)
- [GetTanhTmpBufferFactorSize](高阶API/数学计算/Tanh接口/GetTanhTmpBufferFactorSize.md)
- [Tanh](高阶API/数学计算/Tanh接口/Tanh.md)

#### Tan接口

- [GetTanMaxMinTmpSize](高阶API/数学计算/Tan接口/GetTanMaxMinTmpSize.md)
- [GetTanTmpBufferFactorSize](高阶API/数学计算/Tan接口/GetTanTmpBufferFactorSize.md)
- [Tan](高阶API/数学计算/Tan接口/Tan.md)

#### Trunc接口

- [GetTruncMaxMinTmpSize](高阶API/数学计算/Trunc接口/GetTruncMaxMinTmpSize.md)
- [GetTruncTmpBufferFactorSize](高阶API/数学计算/Trunc接口/GetTruncTmpBufferFactorSize.md)
- [Trunc](高阶API/数学计算/Trunc接口/Trunc.md)

#### Xor接口

- [GetXorMaxMinTmpSize](高阶API/数学计算/Xor接口/GetXorMaxMinTmpSize.md)
- [GetXorTmpBufferFactorSize](高阶API/数学计算/Xor接口/GetXorTmpBufferFactorSize.md)
- [Xor](高阶API/数学计算/Xor接口/Xor.md)

- [更多样例](高阶API/数学计算/更多样例.md)

### 数据过滤

- [DropOut](高阶API/数据过滤/DropOut.md)
- [GetDropOutMaxMinTmpSize](高阶API/数据过滤/GetDropOutMaxMinTmpSize.md)
- [GetSelectMaxMinTmpSize](高阶API/数据过滤/GetSelectMaxMinTmpSize.md)
- [Select](高阶API/数据过滤/Select.md)

### 激活函数

#### GeGLU接口

- [GeGLU](高阶API/激活函数/GeGLU接口/GeGLU.md)
- [GetGeGLUMaxMinTmpSize](高阶API/激活函数/GeGLU接口/GetGeGLUMaxMinTmpSize.md)
- [GetGeGLUTmpBufferFactorSize](高阶API/激活函数/GeGLU接口/GetGeGLUTmpBufferFactorSize.md)

#### Gelu接口

- [FasterGelu](高阶API/激活函数/Gelu接口/FasterGelu.md)
- [FasterGeluV2](高阶API/激活函数/Gelu接口/FasterGeluV2.md)
- [Gelu](高阶API/激活函数/Gelu接口/Gelu.md)
- [GetGeluMaxMinTmpSize](高阶API/激活函数/Gelu接口/GetGeluMaxMinTmpSize.md)

#### LogSoftMax接口

- [LogSoftMax](高阶API/激活函数/LogSoftMax接口/LogSoftMax.md)
- [LogSoftMax Tiling](高阶API/激活函数/LogSoftMax接口/LogSoftMax_Tiling.md)

#### ReGlu接口

- [GetReGluMaxMinTmpSize](高阶API/激活函数/ReGlu接口/GetReGluMaxMinTmpSize.md)
- [ReGlu](高阶API/激活函数/ReGlu接口/ReGlu.md)

#### Sigmoid接口

- [GetSigmoidMaxMinTmpSize](高阶API/激活函数/Sigmoid接口/GetSigmoidMaxMinTmpSize.md)
- [Sigmoid](高阶API/激活函数/Sigmoid接口/Sigmoid.md)

#### Silu接口

- [GetSiluTmpSize](高阶API/激活函数/Silu接口/GetSiluTmpSize.md)
- [Silu](高阶API/激活函数/Silu接口/Silu.md)

#### SoftMax接口

- [AdjustSoftMaxRes](高阶API/激活函数/SoftMax接口/AdjustSoftMaxRes.md)
- [IsBasicBlockInSoftMax](高阶API/激活函数/SoftMax接口/IsBasicBlockInSoftMax.md)
- [SimpleSoftMax](高阶API/激活函数/SoftMax接口/SimpleSoftMax.md)
- [SoftMax](高阶API/激活函数/SoftMax接口/SoftMax.md)
- [SoftMax Tiling使用说明](高阶API/激活函数/SoftMax接口/SoftMax_Tiling使用说明.md)
- [SoftMax/SimpleSoftMax Tiling](高阶API/激活函数/SoftMax接口/SoftMax_SimpleSoftMax_Tiling.md)
- [SoftmaxFlash](高阶API/激活函数/SoftMax接口/SoftmaxFlash.md)
- [SoftmaxFlash Tiling接口](高阶API/激活函数/SoftMax接口/SoftmaxFlash_Tiling接口.md)
- [SoftmaxFlashV2](高阶API/激活函数/SoftMax接口/SoftmaxFlashV2.md)
- [SoftmaxFlashV2 Tiling接口](高阶API/激活函数/SoftMax接口/SoftmaxFlashV2_Tiling接口.md)
- [SoftmaxFlashV3](高阶API/激活函数/SoftMax接口/SoftmaxFlashV3.md)
- [SoftmaxFlashV3 Tiling接口](高阶API/激活函数/SoftMax接口/SoftmaxFlashV3_Tiling接口.md)
- [SoftmaxGrad](高阶API/激活函数/SoftMax接口/SoftmaxGrad.md)
- [SoftmaxGrad Tiling接口](高阶API/激活函数/SoftMax接口/SoftmaxGrad_Tiling接口.md)
- [SoftmaxGradFront](高阶API/激活函数/SoftMax接口/SoftmaxGradFront.md)

#### SwiGLU接口

- [GetSwiGLUMaxMinTmpSize](高阶API/激活函数/SwiGLU接口/GetSwiGLUMaxMinTmpSize.md)
- [GetSwiGLUTmpBufferFactorSize](高阶API/激活函数/SwiGLU接口/GetSwiGLUTmpBufferFactorSize.md)
- [SwiGLU](高阶API/激活函数/SwiGLU接口/SwiGLU.md)

#### Swish接口

- [GetSwishTmpSize](高阶API/激活函数/Swish接口/GetSwishTmpSize.md)
- [Swish](高阶API/激活函数/Swish接口/Swish.md)

### 矩阵计算

#### Matmul Kernel侧接口

- [AsyncGetTensorC](高阶API/矩阵计算/Matmul_Kernel侧接口/AsyncGetTensorC.md)
- [ClearBias](高阶API/矩阵计算/Matmul_Kernel侧接口/ClearBias.md)
- [DisableBias](高阶API/矩阵计算/Matmul_Kernel侧接口/DisableBias.md)
- [End](高阶API/矩阵计算/Matmul_Kernel侧接口/End.md)
- [GetBasicConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetBasicConfig.md)
- [GetBatchC](高阶API/矩阵计算/Matmul_Kernel侧接口/GetBatchC.md)
- [GetBatchTensorC](高阶API/矩阵计算/Matmul_Kernel侧接口/GetBatchTensorC.md)
- [GetIBShareNormConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetIBShareNormConfig.md)
- [GetMDLConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetMDLConfig.md)
- [GetMMConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetMMConfig.md)
- [GetMatmulApiTiling](高阶API/矩阵计算/Matmul_Kernel侧接口/GetMatmulApiTiling.md)
- [GetNormalConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetNormalConfig.md)
- [GetOffsetC](高阶API/矩阵计算/Matmul_Kernel侧接口/GetOffsetC.md)
- [GetSpecialBasicConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetSpecialBasicConfig.md)
- [GetSpecialMDLConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/GetSpecialMDLConfig.md)
- [GetTensorC](高阶API/矩阵计算/Matmul_Kernel侧接口/GetTensorC.md)
- [Init](高阶API/矩阵计算/Matmul_Kernel侧接口/Init.md)
- [Iterate](高阶API/矩阵计算/Matmul_Kernel侧接口/Iterate.md)
- [IterateAll](高阶API/矩阵计算/Matmul_Kernel侧接口/IterateAll.md)
- [IterateBatch](高阶API/矩阵计算/Matmul_Kernel侧接口/IterateBatch.md)
- [IterateNBatch](高阶API/矩阵计算/Matmul_Kernel侧接口/IterateNBatch.md)
- [MatmulCallBackFunc](高阶API/矩阵计算/Matmul_Kernel侧接口/MatmulCallBackFunc.md)
- [MatmulConfig](高阶API/矩阵计算/Matmul_Kernel侧接口/MatmulConfig.md)
- [MatmulPolicy](高阶API/矩阵计算/Matmul_Kernel侧接口/MatmulPolicy.md)
- [Matmul使用说明](高阶API/矩阵计算/Matmul_Kernel侧接口/Matmul使用说明.md)
- [Matmul模板参数](高阶API/矩阵计算/Matmul_Kernel侧接口/Matmul模板参数.md)
- [Matmul特性场景](高阶API/矩阵计算/Matmul_Kernel侧接口/Matmul特性场景.md)
- [REGIST_MATMUL_OBJ](高阶API/矩阵计算/Matmul_Kernel侧接口/REGIST_MATMUL_OBJ.md)
- [SetAntiQuantScalar](高阶API/矩阵计算/Matmul_Kernel侧接口/SetAntiQuantScalar.md)
- [SetAntiQuantVector](高阶API/矩阵计算/Matmul_Kernel侧接口/SetAntiQuantVector.md)
- [SetBatchNum](高阶API/矩阵计算/Matmul_Kernel侧接口/SetBatchNum.md)
- [SetBias](高阶API/矩阵计算/Matmul_Kernel侧接口/SetBias.md)
- [SetHF32](高阶API/矩阵计算/Matmul_Kernel侧接口/SetHF32.md)
- [SetLocalWorkspace](高阶API/矩阵计算/Matmul_Kernel侧接口/SetLocalWorkspace.md)
- [SetOrgShape](高阶API/矩阵计算/Matmul_Kernel侧接口/SetOrgShape.md)
- [SetQuantScalar](高阶API/矩阵计算/Matmul_Kernel侧接口/SetQuantScalar.md)
- [SetQuantVector](高阶API/矩阵计算/Matmul_Kernel侧接口/SetQuantVector.md)
- [SetSelfDefineData](高阶API/矩阵计算/Matmul_Kernel侧接口/SetSelfDefineData.md)
- [SetSingleShape](高阶API/矩阵计算/Matmul_Kernel侧接口/SetSingleShape.md)
- [SetSparseIndex](高阶API/矩阵计算/Matmul_Kernel侧接口/SetSparseIndex.md)
- [SetTail](高阶API/矩阵计算/Matmul_Kernel侧接口/SetTail.md)
- [SetTensorA](高阶API/矩阵计算/Matmul_Kernel侧接口/SetTensorA.md)
- [SetTensorB](高阶API/矩阵计算/Matmul_Kernel侧接口/SetTensorB.md)
- [SetUserDefInfo](高阶API/矩阵计算/Matmul_Kernel侧接口/SetUserDefInfo.md)
- [SetWorkspace](高阶API/矩阵计算/Matmul_Kernel侧接口/SetWorkspace.md)
- [WaitGetTensorC](高阶API/矩阵计算/Matmul_Kernel侧接口/WaitGetTensorC.md)
- [WaitIterateAll](高阶API/矩阵计算/Matmul_Kernel侧接口/WaitIterateAll.md)
- [WaitIterateBatch](高阶API/矩阵计算/Matmul_Kernel侧接口/WaitIterateBatch.md)

#### Matmul Tiling侧接口

##### Matmul Tiling类

- [EnableBias](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/EnableBias.md)
- [EnableL1BankConflictOptimise](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/EnableL1BankConflictOptimise.md)
- [EnableMultiCoreSplitK](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/EnableMultiCoreSplitK.md)
- [GetBaseK](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetBaseK.md)
- [GetBaseM](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetBaseM.md)
- [GetBaseN](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetBaseN.md)
- [GetCoreNum](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetCoreNum.md)
- [GetSingleShape](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetSingleShape.md)
- [GetTiling](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/GetTiling.md)
- [Matmul Tiling类使用说明](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/Matmul_Tiling类使用说明.md)
- [Matmul Tiling类构造函数](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/Matmul_Tiling类构造函数.md)
- [SetALayout](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetALayout.md)
- [SetAType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetAType.md)
- [SetAlignSplit](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetAlignSplit.md)
- [SetBLayout](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBLayout.md)
- [SetBType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBType.md)
- [SetBatchInfoForNormal](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBatchInfoForNormal.md)
- [SetBatchNum](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBatchNum.md)
- [SetBias](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBias.md)
- [SetBiasType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBiasType.md)
- [SetBufferSpace](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetBufferSpace.md)
- [SetCLayout](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetCLayout.md)
- [SetCType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetCType.md)
- [SetDequantType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetDequantType.md)
- [SetDim](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetDim.md)
- [SetDoubleBuffer](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetDoubleBuffer.md)
- [SetFixSplit](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetFixSplit.md)
- [SetMadType](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetMadType.md)
- [SetMatmulConfigParams](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetMatmulConfigParams.md)
- [SetOrgShape](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetOrgShape.md)
- [SetShape](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetShape.md)
- [SetSingleRange](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetSingleRange.md)
- [SetSingleShape](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetSingleShape.md)
- [SetSparse](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetSparse.md)
- [SetSplitK](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetSplitK.md)
- [SetSplitRange](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetSplitRange.md)
- [SetTraverse](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/SetTraverse.md)
- [TCubeTiling结构体](高阶API/矩阵计算/Matmul_Tiling侧接口/Matmul_Tiling类/TCubeTiling结构体.md)

##### 获取Matmul计算所需空间

- [BatchMatmulGetTmpBufSize](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/BatchMatmulGetTmpBufSize.md)
- [BatchMatmulGetTmpBufSizeV2](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/BatchMatmulGetTmpBufSizeV2.md)
- [MatmulGetTmpBufSize](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/MatmulGetTmpBufSize.md)
- [MatmulGetTmpBufSizeV2](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/MatmulGetTmpBufSizeV2.md)
- [MultiCoreMatmulGetTmpBufSize](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/MultiCoreMatmulGetTmpBufSize.md)
- [MultiCoreMatmulGetTmpBufSizeV2](高阶API/矩阵计算/Matmul_Tiling侧接口/获取Matmul计算所需空间/MultiCoreMatmulGetTmpBufSizeV2.md)

### 索引计算

- [Arange](高阶API/索引计算/Arange.md)
- [GetArangeMaxMinTmpSize](高阶API/索引计算/GetArangeMaxMinTmpSize.md)

### 量化操作

- [AscendAntiQuant](高阶API/量化操作/AscendAntiQuant.md)
- [AscendDequant](高阶API/量化操作/AscendDequant.md)
- [AscendQuant](高阶API/量化操作/AscendQuant.md)
- [GetAscendAntiQuantMaxMinTmpSize](高阶API/量化操作/GetAscendAntiQuantMaxMinTmpSize.md)
- [GetAscendAntiQuantTmpBufferFactorSize](高阶API/量化操作/GetAscendAntiQuantTmpBufferFactorSize.md)
- [GetAscendDequantMaxMinTmpSize](高阶API/量化操作/GetAscendDequantMaxMinTmpSize.md)
- [GetAscendDequantTmpBufferFactorSize](高阶API/量化操作/GetAscendDequantTmpBufferFactorSize.md)
- [GetAscendQuantMaxMinTmpSize](高阶API/量化操作/GetAscendQuantMaxMinTmpSize.md)
- [GetAscendQuantTmpBufferFactorSize](高阶API/量化操作/GetAscendQuantTmpBufferFactorSize.md)

- [Ascend C API列表](Ascend_C_API列表.md)
- [通用说明和约束](通用说明和约束.md)

