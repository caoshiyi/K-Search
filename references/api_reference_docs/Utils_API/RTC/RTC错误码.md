# RTC错误码

**页面ID:** atlasascendc_api_07_00161  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00161.html

---

**表1 **aclrtc接口返回aclError错误码说明

| 错误码名称 | 错误码值 | 含义 |
| --- | --- | --- |
| ACL_SUCCESS | 0 | 执行成功。 |
| ACL_ERROR_RTC_INVALID_PROG | 176000 | 无效的aclrtcProg (handle)。 |
| ACL_ERROR_RTC_INVALID_INPUT | 176001 | 除prog入参以外的入参错误。 |
| ACL_ERROR_RTC_INVALID_OPTION | 176002 | 编译选项错误。 |
| ACL_ERROR_RTC_COMPILATION | 176003 | 编译报错。 |
| ACL_ERROR_RTC_LINKING | 176004 | 链接报错。 |
| ACL_ERROR_RTC_NO_NAME_EXPR_AFTER_COMPILATION | 176005 | 编译后没有函数名。 |
| ACL_ERROR_RTC_NO_LOWERED_NAMES_BEFORE_COMPILATION | 176006 | 编译后核函数名无法转换成Mangling名称。 |
| ACL_ERROR_RTC_NAME_EXPR_NOT_VALID | 176007 | 传入无效的核函数名。 |
| ACL_ERROR_RTC_NAME_EXPR_NOT_VALID | 276000 | 创建aclrtcProg (handle) 失败。 |
| ACL_ERROR_RTC_OUT_OF_MEMORY | 276001 | 内存不足。 |
| ACL_ERROR_RTC_FAILURE | 576000 | RTC内部错误。 |
