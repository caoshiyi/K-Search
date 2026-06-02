# DECLARE_ERRORNO

**页面ID:** atlasopapi_07_00518  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00518.html

---

错误码及描述注册宏，该宏对外提供如下四个错误码供用户使用：

- SUCCESS：成功。
- FAILED：失败。
- PARAM_INVALID：参数不合法。
- SCOPE_NOT_CHANGED：Scope融合规则未匹配到，忽略当前pass。

声明如下所示：

```
DECLARE_ERRORNO(0, 0, SUCCESS, 0);
DECLARE_ERRORNO(0xFF, 0xFF, FAILED, 0xFFFFFFFF);
DECLARE_ERRORNO_COMMON(PARAM_INVALID, 1);  // 50331649
DECLARE_ERRORNO(SYSID_FWK, 1, SCOPE_NOT_CHANGED, 201);
```

您可以在“${INSTALL_DIR}/compiler/include/register/register_error_codes.h”下查看错误码定义。

其中，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。
