# SetGroups<a name="ZH-CN_TOPIC_0000002554343507"></a>

## 功能说明<a name="section42122614271"></a>

设置分组卷积的分组大小。分组大小为1表示普通卷积。**当前Conv3D 高阶API不支持分组卷积**。

## 函数原型<a name="section1855173211271"></a>

```
void SetGroups(int64_t groups)
```

## 参数说明<a name="section19975194016273"></a>

<a name="table18369357386"></a>
<table><thead align="left"><tr id="row6385175173812"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p183869573818"><a name="p183869573818"></a><a name="p183869573818"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p1638675183812"><a name="p1638675183812"></a><a name="p1638675183812"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p038615543817"><a name="p038615543817"></a><a name="p038615543817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row7386145113811"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p638614593813"><a name="p638614593813"></a><a name="p638614593813"></a><span>groups</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p9386165103814"><a name="p9386165103814"></a><a name="p9386165103814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p838611553818"><a name="p838611553818"></a><a name="p838611553818"></a>当前仅支持取值为1，暂不支持分组卷积。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section6577547202715"></a>

无

## 约束说明<a name="section37475082810"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认groups=1，当前仅支持输入groups值配置为1，group\>1的卷积能力暂不支持。

## 调用示例<a name="section1017628102810"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetGroups(groups);
```

