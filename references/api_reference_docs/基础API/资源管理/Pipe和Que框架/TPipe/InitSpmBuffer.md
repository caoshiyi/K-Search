# InitSpmBuffer

**页面ID:** atlasascendc_api_07_0166  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0166.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

初始化SPM Buffer。

#### 函数原型

- 暂存到workspace初始化，需要指定GM地址为SPM Buffer：

```
template <typename T>
__aicore__ inline void InitSpmBuffer(const GlobalTensor<T>& workspace, const int32_t bufferSize)
```

- 暂存到L1 Buffer初始化，不需要指定地址，会默认暂存到L1 Buffer，只需要传入需要的SPM Buffer大小：

```
__aicore__ inline void InitSpmBuffer(const int32_t  bufferSize)
```

Atlas A2 训练系列产品/Atlas A2 推理系列产品，不支持暂存到L1 Buffer初始化接口。

Atlas A3 训练系列产品/Atlas A3 推理系列产品，不支持暂存到L1 Buffer初始化接口。

#### 参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| workspace | 输入 | workspace地址。 |
| bufferSize | 输入 | SPM Buffer的大小，单位是字节。 |

#### 约束说明

无

#### 调用示例

- 暂存到workspace初始化

```
AscendC::TPipe pipe;
int len = 1024; // 设置spm buffer为1024个类型为T的数据
workspace_gm.SetGlobalBuffer((__gm__ T *)usrWorkspace, len);  // 此处的usrWorkspace为用户自定义的workspace
auto gm = workspace_gm[AscendC::GetBlockIdx() * len];
pipe.InitSpmBuffer(gm, len * sizeof(T));
```

- 暂存到L1 Buffer初始化

```
AscendC::TPipe pipe;
int len = 1024; // 设置spm buffer为1024个类型为T的数据
pipe.InitSpmBuffer(len * sizeof(T));
```
