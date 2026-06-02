# DEVICE\_IMPL\_OP\_OPTILING<a name="ZH-CN_TOPIC_0000002523304010"></a>

## 功能说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section36583473819"></a>

在[Tiling下沉](使能Tiling下沉.md)场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。

## 函数原型<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
namespace optiling {
using SinkTilingFunc = std::function<ge::graphStatus(gert::TilingContext *context)>;

class DeviceOpImplRegisterImpl;
// 开发者仅关注Tiling成员函数
class DeviceOpImplRegister {
public:
  DeviceOpImplRegister(const char *opType);
  ~DeviceOpImplRegister();
  DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept;
  DeviceOpImplRegister(const DeviceOpImplRegister &other);
  DeviceOpImplRegister &operator=(const DeviceOpImplRegister &) = delete;
  DeviceOpImplRegister &operator=(DeviceOpImplRegister &&) = delete;
  DeviceOpImplRegister &Tiling(SinkTilingFunc func);

// ...
};
}  // namespace optiling

#define DEVICE_IMPL_OP_OPTILING(optype)                                                                      \
  static optiling::DeviceOpImplRegister VAR_UNUSED g_deviceOpImplRegister##optype =                                    \
      optiling::DeviceOpImplRegister(#optype)
#endif
```

## 参数说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section75395119104"></a>

**表 1**  DEVICE\_IMPL\_OP\_OPTILING参数说明

<a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="p1693913355118"><a name="p1693913355118"></a><a name="p1693913355118"></a>optype</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001820490272_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="p3511614184115"><a name="p3511614184115"></a><a name="p3511614184115"></a>需要注册Tiling函数的OpType（算子类型）。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  Tiling成员函数参数说明

<a name="table2060634264310"></a>
<table><thead align="left"><tr id="row860654216436"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="p19606154212434"><a name="p19606154212434"></a><a name="p19606154212434"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="p17606174294315"><a name="p17606174294315"></a><a name="p17606174294315"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="p1460614224313"><a name="p1460614224313"></a><a name="p1460614224313"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row160654218436"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="p14606104284320"><a name="p14606104284320"></a><a name="p14606104284320"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p16606124294310"><a name="p16606124294310"></a><a name="p16606124294310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="p25503505020"><a name="p25503505020"></a><a name="p25503505020"></a>需要注册的Tiling函数，该函数接受一个TilingContext作为输入，以ge::graphStatus为返回值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001867289945_zh-cn_topic_0000001389787297_section320753512363"></a>

```
DEVICE_IMPL_OP_OPTILING(TestOptype).Tiling(TestTilingFunc); // 将Tiling函数以及其OpType注册到Tiling下沉
```

