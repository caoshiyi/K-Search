# UnalignRegForLoad & UnalignRegForStore<a name="ZH-CN_TOPIC_0000002554424673"></a>

## 功能说明<a name="section618mcpsimp"></a>

UnalignRegForLoad、UnalignRegForStore用作缓冲区来优化UB和RegTensor之间连续不对齐地址访问的开销。在读不对齐地址前，UnalignRegForLoad、UnalignRegForStore应该通过LoadUnAlignPre API初始化，然后使用LoadUnAlign API。在写不对齐地址时，先使用StoreUnAlign API，再使用StoreUnAlignPost API后处理。

UnalignRegForLoad 、UnalignRegForStore的使用可参考[连续非对齐搬入](连续非对齐搬入.md)、[连续非对齐搬出](连续非对齐搬出.md)。

## 支持的型号<a name="section156721693504"></a>

Ascend 950PR/Ascend 950DT

## 约束说明<a name="section11585101304320"></a>

无

