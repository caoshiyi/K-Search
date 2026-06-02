# InitBuffer

**页面ID:** atlasascendc_api_07_0110  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0110.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

用于为TQue等队列和TBuf分配内存。

#### 函数原型

- 为TQue等队列分配内存

```
template <class T>
__aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)

// 为TQue等队列分配内存，且开发者可以自定义内存的地址信息，地址信息包含起始地址和长度。
template <class T, class U, class V, class... Addrs>
__aicore__ inline bool InitBuffer(T& que, const Std::tuple<U, V>& addr0, const Addrs&... addrs)
```

- 为TBuf分配内存

```
template <TPosition bufPos>
__aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len)
```

#### 参数说明

**表1 **bool InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 队列的类型，支持取值TQue、TQueBind。 |

**表2 **bool InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| que | 输入 | 需要分配内存的TQue等对象。 |
| num | 输入 | 分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer。 |
| len | 输入 | 每个内存块的大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。 |

**表3 **bool InitBuffer(T& que, const Std::tuple<U, V>& addr0, const Addrs&... addrs) 原型定义模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 队列的类型，支持取值TQue、TQueBind。 |
| U | 起始地址的类型。类型为整型。 |
| V | 长度的类型。类型为整型。 |
| Addrs... | tuple形式的地址信息，包含起始地址和长度。 |

**表4 **bool InitBuffer(T& que, const Std::tuple<U, V>& addr0, const Addrs&... addrs) 原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| que | 输入 | 需要分配内存的TQue等对象。 |
| addr0 | 输入 | 分配内存块的地址信息，结构为起始地址和长度。 |
| addrs | 输入 | 格式为tuple的地址信息列表，tuple内的元素个数须为2，即起始地址和长度。 |

**表5 **InitBuffer(TBuf<bufPos>& buf, uint32_t len)原型定义模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| bufPos | TBuf所在的逻辑位置，TPosition类型。 |

**表6 **InitBuffer(TBuf<bufPos>& buf, uint32_t len)原型定义参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| buf | 输入 | 需要分配内存的TBuf对象。 |
| len | 输入 | 为TBuf分配的内存大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考非对齐场景。 |

#### 约束说明

- InitBuffer申请的内存会在TPipe对象销毁时通过析构函数自动释放，无需手动释放。
- 如果需要重新分配InitBuffer申请的内存，可以调用Reset，再调用InitBuffer接口。
- 一个kernel中所有使用的Buffer数量之和不能超过64。
- 自定义地址InitBuffer方式分配不建议与不指定地址的混用，可能会导致内存冲突。

#### 返回值说明

返回Buffer初始化的结果。

#### 调用示例

```
// 为TQue分配内存，分配内存块数为2，每块大小为128字节
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; // 输出数据队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);

// 为TQue分配内存，开发者自定义内存的地址信息，分别为[0, 1024], [2048, 4096], [8192, 12288]
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 1> que; // 输出数据队列管理对象，TPosition为VECOUT
auto addr0 = Std::make_tuple(0, 1024);
auto addr1 = Std::make_tuple(2048, 2048);
auto addr2 = Std::make_tuple(8192, 4096);
pipe.InitBuffer(que, addr0, addr1, addr2);

// 为TBuf分配内存，分配长度为128字节
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::A1> buf; // 输出数据管理对象，TPosition为A1
uint32_t len = 128;
pipe.InitBuffer(buf, len);
```
