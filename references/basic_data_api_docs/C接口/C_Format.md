# C_Format

**页面ID:** atlasopapi_07_00720  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00720.html

---

```
typedef enum {
  C_FORMAT_NCHW = 0,   // NCHW
  C_FORMAT_NHWC,       // NHWC
  C_FORMAT_ND,         // Nd Tensor
  C_FORMAT_NC1HWC0,    // NC1HWC0
  C_FORMAT_FRACTAL_Z,  // FRACTAL_Z
  C_FORMAT_NC1C0HWPAD = 5,
  C_FORMAT_NHWC1C0,
  C_FORMAT_FSR_NCHW,
  C_FORMAT_FRACTAL_DECONV,
  C_FORMAT_C1HWNC0,
  C_FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
  C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  C_FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
  C_FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
  C_FORMAT_CHWN,
  C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
  C_FORMAT_HWCN,
  C_FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  C_FORMAT_BN_WEIGHT,
  C_FORMAT_FILTER_HWCK,  // filter input tensor format
  C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
  C_FORMAT_HASHTABLE_LOOKUP_KEYS,
  C_FORMAT_HASHTABLE_LOOKUP_VALUE,
  C_FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  C_FORMAT_HASHTABLE_LOOKUP_HITS,
  C_FORMAT_C1HWNCoC0 = 25,
  C_FORMAT_MD,
  C_FORMAT_NDHWC,
  C_FORMAT_FRACTAL_ZZ,
  C_FORMAT_FRACTAL_NZ,
  C_FORMAT_NCDHW = 30,
  C_FORMAT_DHWCN,  // 3D filter input tensor format
  C_FORMAT_NDC1HWC0,
  C_FORMAT_FRACTAL_Z_3D,
  C_FORMAT_CN,
  C_FORMAT_NC = 35,
  C_FORMAT_DHWNC,
  C_FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
  C_FORMAT_FRACTAL_ZN_LSTM,
  C_FORMAT_FRACTAL_Z_G,
  C_FORMAT_RESERVED = 40,
  C_FORMAT_ALL,
  C_FORMAT_NULL,
  C_FORMAT_ND_RNN_BIAS,
  C_FORMAT_FRACTAL_ZN_RNN,
  C_FORMAT_NYUV = 45,
  C_FORMAT_NYUV_A,
  C_FORMAT_NCL,
  C_FORMAT_FRACTAL_Z_WINO,
  C_FORMAT_C1HWC0,
  C_FORMAT_FRACTAL_NZ_C0_16,   //当前版本不支持该类型。
  C_FORMAT_FRACTAL_NZ_C0_32,   //当前版本不支持该类型。
  C_FORMAT_FRACTAL_NZ_C0_2,    //当前版本不支持该类型。
  C_FORMAT_FRACTAL_NZ_C0_4,    //当前版本不支持该类型。
  C_FORMAT_FRACTAL_NZ_C0_8,    //当前版本不支持该类型。
  // Add new formats definition here
  C_FORMAT_END,
  // FORMAT_MAX defines the max value of Format.
  // Any Format should not exceed the value of FORMAT_MAX.
  // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
  //                   If you want to judge the range of Format, you can use FORMAT_END.
  C_FORMAT_MAX = 0xff
} C_Format;
```
