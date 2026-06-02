# GetSortLen<a name="ZH-CN_TOPIC_0000002523344930"></a>

## 功能说明<a name="section618mcpsimp"></a>

根据元素数量，获取Sort数据的大小（单位为字节）。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline uint32_t GetSortLen(const uint32_t elemCount)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table32097383597"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424089_row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002554424089_p1029955044218"><a name="zh-cn_topic_0000002554424089_p1029955044218"></a><a name="zh-cn_topic_0000002554424089_p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002554424089_p1629911506421"><a name="zh-cn_topic_0000002554424089_p1629911506421"></a><a name="zh-cn_topic_0000002554424089_p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424089_row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002554424089_p1329915004219"><a name="zh-cn_topic_0000002554424089_p1329915004219"></a><a name="zh-cn_topic_0000002554424089_p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554424089_p8299155010420"><a name="zh-cn_topic_0000002554424089_p8299155010420"></a><a name="zh-cn_topic_0000002554424089_p8299155010420"></a>操作数的数据类型。</p>
<p id="zh-cn_topic_0000002554424089_p5315184745513"><a name="zh-cn_topic_0000002554424089_p5315184745513"></a><a name="zh-cn_topic_0000002554424089_p5315184745513"></a><span id="zh-cn_topic_0000002554424089_ph2272194216543"><a name="zh-cn_topic_0000002554424089_ph2272194216543"></a><a name="zh-cn_topic_0000002554424089_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p652599101415"><a name="p652599101415"></a><a name="p652599101415"></a>elemCount</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p521618329466"><a name="p521618329466"></a><a name="p521618329466"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>输入元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

sort数据的大小（单位为字节）。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::GetSortLen<half>(128);
```

