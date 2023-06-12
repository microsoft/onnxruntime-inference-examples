/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter act_bw=8 act_quantizer=tf adjust_nms_features_dims=True algorithms=[] align_matmul_ranks=True arch_checker=False batch=1 bias_bw=32 copyright_file=None custom_io= custom_op_config_paths=None debug=-1 define_symbol=None disable_batchnorm_folding=False disable_node_validation=False dry_run=None dumpIR=False dump_custom_io_config_template= dump_inferred_model=False dump_value_info=False enable_match_gathernd=False exclude_named_tensors=False expand_gru_op_structure=True extract_color_transform=True float_bw=32 force_prune_cast_ops=False handle_gather_negative_indices=True ignore_encodings=False inject_cast_for_gather=True input_dim=None input_dtype=[] input_encoding=[] input_layout=[] input_list=/mnt/c/d/Git/onnxruntime-inference-examples/c_cxx/QNN_EP/mobilenetv2_classification/input.txt input_type=[] keep_disconnected_nodes=False keep_int64_inputs=False keep_quant_nodes=False match_caffe_ssd_to_tf=True no_simplification=False op_package_lib= out_names=['output'] overwrite_model_prefix=False package_name=None param_quantizer=tf perform_axes_to_spatial_first_order=True prepare_inputs_as_params=False preprocess_roi_pool_inputs=True quantization_overrides= squash_box_decoder=True unroll_gru_time_steps=True unroll_lstm_time_steps=True use_convert_quantization_nodes=False use_native_dtype=False use_native_input_files=False use_native_output_files=False use_per_channel_quantization=[False] use_per_row_quantization=False weight_bw=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
extern "C" {
QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for mobilenetv2_12*/
  QnnModel mobilenetv2_12;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("mobilenetv2_12", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(mobilenetv2_12.initialize(backendHandle, interface, contextHandle, "mobilenetv2_12", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  uint32_t dimensions_input[] = {1, 224, 224, 3};
  VALIDATE(mobilenetv2_12.addTensor("input", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "input",
                                          .type= QNN_TENSOR_TYPE_APP_WRITE,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0158752091228962f, .offset= -116}}},
                                          .rank= 4,
                                          .dimensions=dimensions_input,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=nullptr,
                                                         .dataSize=0}}}}}
  ), err);
  uint32_t dimensions__475[] = {3, 3, 3, 32};
  VALIDATE(mobilenetv2_12.addTensor("_475", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_475",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0027297507040203f, .offset= -133}}},
                                          .rank= 4,
                                          .dimensions=dimensions__475,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_475),
                                                         .dataSize=BINLEN(_475)}}}}}
  ), err);
  uint32_t dimensions__476[] = {32};
  VALIDATE(mobilenetv2_12.addTensor("_476", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_476",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004506452f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__476,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_476),
                                                         .dataSize=BINLEN(_476)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_0 */
  uint32_t dimensions_Conv_0_dilation[] = {2};
  uint32_t Conv_0_dilation[] = {1, 1};
  uint32_t dimensions_Conv_0_pad_amount[] = {2, 2};
  uint32_t Conv_0_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_0_stride[] = {2};
  uint32_t Conv_0_stride[] = {2, 2};
  Qnn_Param_t params_Conv_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_0_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_0_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_0_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_0_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_0_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_0_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_0_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_0_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_0_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_0[] = {
    "input",
    "_475",
    "_476"
  };
  uint32_t dimensions__317[] = {1, 112, 112, 32};
  Qnn_Tensor_t outputs_Conv_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_317",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0081563256680965f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__317,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_0", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_0, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_0, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_0, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__478[] = {3, 3, 1, 32};
  VALIDATE(mobilenetv2_12.addTensor("_478", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_478",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.1007483005523682f, .offset= -150}}},
                                          .rank= 4,
                                          .dimensions=dimensions__478,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_478),
                                                         .dataSize=BINLEN(_478)}}}}}
  ), err);
  uint32_t dimensions__479[] = {32};
  VALIDATE(mobilenetv2_12.addTensor("_479", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_479",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000017376086f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__479,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_479),
                                                         .dataSize=BINLEN(_479)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_2 */
  uint32_t dimensions_Conv_2_dilation[] = {2};
  uint32_t Conv_2_dilation[] = {1, 1};
  uint32_t dimensions_Conv_2_pad_amount[] = {2, 2};
  uint32_t Conv_2_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_2_stride[] = {2};
  uint32_t Conv_2_stride[] = {1, 1};
  Qnn_Param_t params_Conv_2[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_2_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_2_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_2_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_2_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_2_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_2_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_2_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_2_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_2_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_2[] = {
    "_317",
    "_478",
    "_479"
  };
  uint32_t dimensions__320[] = {1, 112, 112, 32};
  Qnn_Tensor_t outputs_Conv_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_320",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0235294122248888f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__320,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_2", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_2, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_2, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_2, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__481[] = {1, 1, 32, 16};
  VALIDATE(mobilenetv2_12.addTensor("_481", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_481",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0072936317883432f, .offset= -142}}},
                                          .rank= 4,
                                          .dimensions=dimensions__481,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_481),
                                                         .dataSize=BINLEN(_481)}}}}}
  ), err);
  uint32_t dimensions__482[] = {16};
  VALIDATE(mobilenetv2_12.addTensor("_482", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_482",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000009971171f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__482,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_482),
                                                         .dataSize=BINLEN(_482)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_4 */
  uint32_t dimensions_Conv_4_dilation[] = {2};
  uint32_t Conv_4_dilation[] = {1, 1};
  uint32_t dimensions_Conv_4_pad_amount[] = {2, 2};
  uint32_t Conv_4_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_4_stride[] = {2};
  uint32_t Conv_4_stride[] = {1, 1};
  Qnn_Param_t params_Conv_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_4_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_4_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_4_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_4_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_4_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_4_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_4_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_4_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_4_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_4[] = {
    "_320",
    "_481",
    "_482"
  };
  uint32_t dimensions__480[] = {1, 112, 112, 16};
  Qnn_Tensor_t outputs_Conv_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_480",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0328293889760971f, .offset= -129}}},
            .rank= 4,
            .dimensions=dimensions__480,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_4", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_4, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_4, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_4, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__484[] = {1, 1, 16, 96};
  VALIDATE(mobilenetv2_12.addTensor("_484", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_484",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0042772148735821f, .offset= -114}}},
                                          .rank= 4,
                                          .dimensions=dimensions__484,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_484),
                                                         .dataSize=BINLEN(_484)}}}}}
  ), err);
  uint32_t dimensions__485[] = {96};
  VALIDATE(mobilenetv2_12.addTensor("_485", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_485",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003389383f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__485,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_485),
                                                         .dataSize=BINLEN(_485)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_5 */
  uint32_t dimensions_Conv_5_dilation[] = {2};
  uint32_t Conv_5_dilation[] = {1, 1};
  uint32_t dimensions_Conv_5_pad_amount[] = {2, 2};
  uint32_t Conv_5_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_5_stride[] = {2};
  uint32_t Conv_5_stride[] = {1, 1};
  Qnn_Param_t params_Conv_5[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_5_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_5_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_5_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_5_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_5_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_5_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_5_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_5_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_5_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_5[] = {
    "_480",
    "_484",
    "_485"
  };
  uint32_t dimensions__325[] = {1, 112, 112, 96};
  Qnn_Tensor_t outputs_Conv_5[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_325",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0204057060182095f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__325,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_5", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_5, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_5, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_5, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__487[] = {3, 3, 1, 96};
  VALIDATE(mobilenetv2_12.addTensor("_487", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_487",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0436219684779644f, .offset= -146}}},
                                          .rank= 4,
                                          .dimensions=dimensions__487,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_487),
                                                         .dataSize=BINLEN(_487)}}}}}
  ), err);
  uint32_t dimensions__488[] = {96};
  VALIDATE(mobilenetv2_12.addTensor("_488", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_488",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000007471378f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__488,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_488),
                                                         .dataSize=BINLEN(_488)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_7 */
  uint32_t dimensions_Conv_7_dilation[] = {2};
  uint32_t Conv_7_dilation[] = {1, 1};
  uint32_t dimensions_Conv_7_pad_amount[] = {2, 2};
  uint32_t Conv_7_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_7_stride[] = {2};
  uint32_t Conv_7_stride[] = {2, 2};
  Qnn_Param_t params_Conv_7[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_7_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_7_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_7_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_7_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_7_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_7_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_7_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_7_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_7_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_7[] = {
    "_325",
    "_487",
    "_488"
  };
  uint32_t dimensions__328[] = {1, 56, 56, 96};
  Qnn_Tensor_t outputs_Conv_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_328",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0142498165369034f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__328,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_7", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_7, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_7, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_7, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__490[] = {1, 1, 96, 24};
  VALIDATE(mobilenetv2_12.addTensor("_490", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_490",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0059046219103038f, .offset= -127}}},
                                          .rank= 4,
                                          .dimensions=dimensions__490,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_490),
                                                         .dataSize=BINLEN(_490)}}}}}
  ), err);
  uint32_t dimensions__491[] = {24};
  VALIDATE(mobilenetv2_12.addTensor("_491", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_491",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000008393191f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__491,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_491),
                                                         .dataSize=BINLEN(_491)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_9 */
  uint32_t dimensions_Conv_9_dilation[] = {2};
  uint32_t Conv_9_dilation[] = {1, 1};
  uint32_t dimensions_Conv_9_pad_amount[] = {2, 2};
  uint32_t Conv_9_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_9_stride[] = {2};
  uint32_t Conv_9_stride[] = {1, 1};
  Qnn_Param_t params_Conv_9[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_9_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_9_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_9_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_9_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_9_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_9_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_9_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_9_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_9_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_9[] = {
    "_328",
    "_490",
    "_491"
  };
  uint32_t dimensions__489[] = {1, 56, 56, 24};
  Qnn_Tensor_t outputs_Conv_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_489",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0240260511636734f, .offset= -110}}},
            .rank= 4,
            .dimensions=dimensions__489,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_9", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_9, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_9, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_9, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__493[] = {1, 1, 24, 144};
  VALIDATE(mobilenetv2_12.addTensor("_493", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_493",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0025121737271547f, .offset= -133}}},
                                          .rank= 4,
                                          .dimensions=dimensions__493,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_493),
                                                         .dataSize=BINLEN(_493)}}}}}
  ), err);
  uint32_t dimensions__494[] = {144};
  VALIDATE(mobilenetv2_12.addTensor("_494", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_494",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004713931f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__494,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_494),
                                                         .dataSize=BINLEN(_494)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_10 */
  uint32_t dimensions_Conv_10_dilation[] = {2};
  uint32_t Conv_10_dilation[] = {1, 1};
  uint32_t dimensions_Conv_10_pad_amount[] = {2, 2};
  uint32_t Conv_10_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_10_stride[] = {2};
  uint32_t Conv_10_stride[] = {1, 1};
  Qnn_Param_t params_Conv_10[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_10_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_10_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_10_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_10_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_10_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_10_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_10_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_10_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_10_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_10[] = {
    "_489",
    "_493",
    "_494"
  };
  uint32_t dimensions__333[] = {1, 56, 56, 144};
  Qnn_Tensor_t outputs_Conv_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_333",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0044664251618087f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__333,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_10", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_10, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_10, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_10, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__496[] = {3, 3, 1, 144};
  VALIDATE(mobilenetv2_12.addTensor("_496", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_496",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0376739017665386f, .offset= -128}}},
                                          .rank= 4,
                                          .dimensions=dimensions__496,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_496),
                                                         .dataSize=BINLEN(_496)}}}}}
  ), err);
  uint32_t dimensions__497[] = {144};
  VALIDATE(mobilenetv2_12.addTensor("_497", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_497",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000016774298f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__497,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_497),
                                                         .dataSize=BINLEN(_497)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_12 */
  uint32_t dimensions_Conv_12_dilation[] = {2};
  uint32_t Conv_12_dilation[] = {1, 1};
  uint32_t dimensions_Conv_12_pad_amount[] = {2, 2};
  uint32_t Conv_12_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_12_stride[] = {2};
  uint32_t Conv_12_stride[] = {1, 1};
  Qnn_Param_t params_Conv_12[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_12_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_12_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_12_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_12_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_12_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_12_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_12_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_12_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_12_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_12[] = {
    "_333",
    "_496",
    "_497"
  };
  uint32_t dimensions__336[] = {1, 56, 56, 144};
  Qnn_Tensor_t outputs_Conv_12[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_336",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0111504895612597f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__336,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_12", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_12, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_12, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_12, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__499[] = {1, 1, 144, 24};
  VALIDATE(mobilenetv2_12.addTensor("_499", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_499",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0085288751870394f, .offset= -113}}},
                                          .rank= 4,
                                          .dimensions=dimensions__499,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_499),
                                                         .dataSize=BINLEN(_499)}}}}}
  ), err);
  uint32_t dimensions__500[] = {24};
  VALIDATE(mobilenetv2_12.addTensor("_500", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_500",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000010405421f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__500,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_500),
                                                         .dataSize=BINLEN(_500)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_14 */
  uint32_t dimensions_Conv_14_dilation[] = {2};
  uint32_t Conv_14_dilation[] = {1, 1};
  uint32_t dimensions_Conv_14_pad_amount[] = {2, 2};
  uint32_t Conv_14_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_14_stride[] = {2};
  uint32_t Conv_14_stride[] = {1, 1};
  Qnn_Param_t params_Conv_14[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_14_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_14_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_14_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_14_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_14_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_14_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_14_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_14_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_14_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_14[] = {
    "_336",
    "_499",
    "_500"
  };
  uint32_t dimensions__498[] = {1, 56, 56, 24};
  Qnn_Tensor_t outputs_Conv_14[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_498",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0297339186072350f, .offset= -134}}},
            .rank= 4,
            .dimensions=dimensions__498,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_14", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_14, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_14, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_14, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_15 */
  const char*  inputs_Add_15[] = {
    "_489",
    "_498"
  };
  uint32_t dimensions__339[] = {1, 56, 56, 24};
  Qnn_Tensor_t outputs_Add_15[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_339",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0364860594272614f, .offset= -139}}},
            .rank= 4,
            .dimensions=dimensions__339,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_15", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_15, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_15, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__502[] = {1, 1, 24, 144};
  VALIDATE(mobilenetv2_12.addTensor("_502", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_502",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0024617302697152f, .offset= -127}}},
                                          .rank= 4,
                                          .dimensions=dimensions__502,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_502),
                                                         .dataSize=BINLEN(_502)}}}}}
  ), err);
  uint32_t dimensions__503[] = {144};
  VALIDATE(mobilenetv2_12.addTensor("_503", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_503",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001545701f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__503,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_503),
                                                         .dataSize=BINLEN(_503)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_16 */
  uint32_t dimensions_Conv_16_dilation[] = {2};
  uint32_t Conv_16_dilation[] = {1, 1};
  uint32_t dimensions_Conv_16_pad_amount[] = {2, 2};
  uint32_t Conv_16_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_16_stride[] = {2};
  uint32_t Conv_16_stride[] = {1, 1};
  Qnn_Param_t params_Conv_16[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_16_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_16_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_16_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_16_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_16_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_16_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_16_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_16_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_16_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_16[] = {
    "_339",
    "_502",
    "_503"
  };
  uint32_t dimensions__342[] = {1, 56, 56, 144};
  Qnn_Tensor_t outputs_Conv_16[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_342",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0070290584117174f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__342,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_16", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_16, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_16, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_16, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__505[] = {3, 3, 1, 144};
  VALIDATE(mobilenetv2_12.addTensor("_505", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_505",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0444373302161694f, .offset= -122}}},
                                          .rank= 4,
                                          .dimensions=dimensions__505,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_505),
                                                         .dataSize=BINLEN(_505)}}}}}
  ), err);
  uint32_t dimensions__506[] = {144};
  VALIDATE(mobilenetv2_12.addTensor("_506", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_506",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000007387348f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__506,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_506),
                                                         .dataSize=BINLEN(_506)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_18 */
  uint32_t dimensions_Conv_18_dilation[] = {2};
  uint32_t Conv_18_dilation[] = {1, 1};
  uint32_t dimensions_Conv_18_pad_amount[] = {2, 2};
  uint32_t Conv_18_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_18_stride[] = {2};
  uint32_t Conv_18_stride[] = {2, 2};
  Qnn_Param_t params_Conv_18[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_18_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_18_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_18_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_18_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_18_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_18_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_18_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_18_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_18_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_18[] = {
    "_342",
    "_505",
    "_506"
  };
  uint32_t dimensions__345[] = {1, 28, 28, 144};
  Qnn_Tensor_t outputs_Conv_18[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_345",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0103053636848927f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__345,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_18", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_18, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_18, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_18, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__508[] = {1, 1, 144, 32};
  VALIDATE(mobilenetv2_12.addTensor("_508", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_508",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0061763287521899f, .offset= -114}}},
                                          .rank= 4,
                                          .dimensions=dimensions__508,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_508),
                                                         .dataSize=BINLEN(_508)}}}}}
  ), err);
  uint32_t dimensions__509[] = {32};
  VALIDATE(mobilenetv2_12.addTensor("_509", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_509",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000007857577f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__509,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_509),
                                                         .dataSize=BINLEN(_509)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_20 */
  uint32_t dimensions_Conv_20_dilation[] = {2};
  uint32_t Conv_20_dilation[] = {1, 1};
  uint32_t dimensions_Conv_20_pad_amount[] = {2, 2};
  uint32_t Conv_20_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_20_stride[] = {2};
  uint32_t Conv_20_stride[] = {1, 1};
  Qnn_Param_t params_Conv_20[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_20_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_20_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_20_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_20_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_20_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_20_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_20_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_20_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_20_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_20[] = {
    "_345",
    "_508",
    "_509"
  };
  uint32_t dimensions__507[] = {1, 28, 28, 32};
  Qnn_Tensor_t outputs_Conv_20[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_507",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0188498664647341f, .offset= -141}}},
            .rank= 4,
            .dimensions=dimensions__507,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_20", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_20, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_20, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_20, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__511[] = {1, 1, 32, 192};
  VALIDATE(mobilenetv2_12.addTensor("_511", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_511",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0014762870268896f, .offset= -125}}},
                                          .rank= 4,
                                          .dimensions=dimensions__511,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_511),
                                                         .dataSize=BINLEN(_511)}}}}}
  ), err);
  uint32_t dimensions__512[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_512", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_512",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001885530f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__512,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_512),
                                                         .dataSize=BINLEN(_512)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_21 */
  uint32_t dimensions_Conv_21_dilation[] = {2};
  uint32_t Conv_21_dilation[] = {1, 1};
  uint32_t dimensions_Conv_21_pad_amount[] = {2, 2};
  uint32_t Conv_21_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_21_stride[] = {2};
  uint32_t Conv_21_stride[] = {1, 1};
  Qnn_Param_t params_Conv_21[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_21_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_21_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_21_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_21_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_21_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_21_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_21_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_21_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_21_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_21[] = {
    "_507",
    "_511",
    "_512"
  };
  uint32_t dimensions__350[] = {1, 28, 28, 192};
  Qnn_Tensor_t outputs_Conv_21[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_350",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0032256094273180f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__350,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_21", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_21, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_21, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_21, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__514[] = {3, 3, 1, 192};
  VALIDATE(mobilenetv2_12.addTensor("_514", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_514",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0498814955353737f, .offset= -123}}},
                                          .rank= 4,
                                          .dimensions=dimensions__514,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_514),
                                                         .dataSize=BINLEN(_514)}}}}}
  ), err);
  uint32_t dimensions__515[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_515", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_515",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003962591f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__515,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_515),
                                                         .dataSize=BINLEN(_515)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_23 */
  uint32_t dimensions_Conv_23_dilation[] = {2};
  uint32_t Conv_23_dilation[] = {1, 1};
  uint32_t dimensions_Conv_23_pad_amount[] = {2, 2};
  uint32_t Conv_23_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_23_stride[] = {2};
  uint32_t Conv_23_stride[] = {1, 1};
  Qnn_Param_t params_Conv_23[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_23_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_23_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_23_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_23_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_23_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_23_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_23_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_23_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_23_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_23[] = {
    "_350",
    "_514",
    "_515"
  };
  uint32_t dimensions__353[] = {1, 28, 28, 192};
  Qnn_Tensor_t outputs_Conv_23[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_353",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0053794444538653f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__353,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_23", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_23, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_23, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_23, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__517[] = {1, 1, 192, 32};
  VALIDATE(mobilenetv2_12.addTensor("_517", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_517",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0063708536326885f, .offset= -130}}},
                                          .rank= 4,
                                          .dimensions=dimensions__517,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_517),
                                                         .dataSize=BINLEN(_517)}}}}}
  ), err);
  uint32_t dimensions__518[] = {32};
  VALIDATE(mobilenetv2_12.addTensor("_518", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_518",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003567458f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__518,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_518),
                                                         .dataSize=BINLEN(_518)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_25 */
  uint32_t dimensions_Conv_25_dilation[] = {2};
  uint32_t Conv_25_dilation[] = {1, 1};
  uint32_t dimensions_Conv_25_pad_amount[] = {2, 2};
  uint32_t Conv_25_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_25_stride[] = {2};
  uint32_t Conv_25_stride[] = {1, 1};
  Qnn_Param_t params_Conv_25[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_25_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_25_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_25_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_25_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_25_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_25_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_25_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_25_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_25_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_25[] = {
    "_353",
    "_517",
    "_518"
  };
  uint32_t dimensions__516[] = {1, 28, 28, 32};
  Qnn_Tensor_t outputs_Conv_25[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_516",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0172615032643080f, .offset= -135}}},
            .rank= 4,
            .dimensions=dimensions__516,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_25", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_25, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_25, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_25, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_26 */
  const char*  inputs_Add_26[] = {
    "_507",
    "_516"
  };
  uint32_t dimensions__356[] = {1, 28, 28, 32};
  Qnn_Tensor_t outputs_Add_26[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_356",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0217533968389034f, .offset= -138}}},
            .rank= 4,
            .dimensions=dimensions__356,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_26", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_26, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_26, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__520[] = {1, 1, 32, 192};
  VALIDATE(mobilenetv2_12.addTensor("_520", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_520",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0011028250446543f, .offset= -115}}},
                                          .rank= 4,
                                          .dimensions=dimensions__520,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_520),
                                                         .dataSize=BINLEN(_520)}}}}}
  ), err);
  uint32_t dimensions__521[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_521", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_521",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001571717f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__521,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_521),
                                                         .dataSize=BINLEN(_521)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_27 */
  uint32_t dimensions_Conv_27_dilation[] = {2};
  uint32_t Conv_27_dilation[] = {1, 1};
  uint32_t dimensions_Conv_27_pad_amount[] = {2, 2};
  uint32_t Conv_27_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_27_stride[] = {2};
  uint32_t Conv_27_stride[] = {1, 1};
  Qnn_Param_t params_Conv_27[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_27_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_27_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_27_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_27_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_27_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_27_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_27_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_27_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_27_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_27[] = {
    "_356",
    "_520",
    "_521"
  };
  uint32_t dimensions__359[] = {1, 28, 28, 192};
  Qnn_Tensor_t outputs_Conv_27[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_359",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0044450759887695f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__359,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_27", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_27, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_27, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_27, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__523[] = {3, 3, 1, 192};
  VALIDATE(mobilenetv2_12.addTensor("_523", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_523",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0370818637311459f, .offset= -136}}},
                                          .rank= 4,
                                          .dimensions=dimensions__523,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_523),
                                                         .dataSize=BINLEN(_523)}}}}}
  ), err);
  uint32_t dimensions__524[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_524", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_524",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005607529f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__524,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_524),
                                                         .dataSize=BINLEN(_524)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_29 */
  uint32_t dimensions_Conv_29_dilation[] = {2};
  uint32_t Conv_29_dilation[] = {1, 1};
  uint32_t dimensions_Conv_29_pad_amount[] = {2, 2};
  uint32_t Conv_29_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_29_stride[] = {2};
  uint32_t Conv_29_stride[] = {1, 1};
  Qnn_Param_t params_Conv_29[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_29_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_29_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_29_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_29_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_29_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_29_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_29_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_29_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_29_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_29[] = {
    "_359",
    "_523",
    "_524"
  };
  uint32_t dimensions__362[] = {1, 28, 28, 192};
  Qnn_Tensor_t outputs_Conv_29[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_362",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0056859208270907f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__362,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_29", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_29, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_29, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_29, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__526[] = {1, 1, 192, 32};
  VALIDATE(mobilenetv2_12.addTensor("_526", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_526",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0072229886427522f, .offset= -131}}},
                                          .rank= 4,
                                          .dimensions=dimensions__526,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_526),
                                                         .dataSize=BINLEN(_526)}}}}}
  ), err);
  uint32_t dimensions__527[] = {32};
  VALIDATE(mobilenetv2_12.addTensor("_527", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_527",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003696381f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__527,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_527),
                                                         .dataSize=BINLEN(_527)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_31 */
  uint32_t dimensions_Conv_31_dilation[] = {2};
  uint32_t Conv_31_dilation[] = {1, 1};
  uint32_t dimensions_Conv_31_pad_amount[] = {2, 2};
  uint32_t Conv_31_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_31_stride[] = {2};
  uint32_t Conv_31_stride[] = {1, 1};
  Qnn_Param_t params_Conv_31[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_31_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_31_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_31_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_31_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_31_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_31_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_31_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_31_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_31_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_31[] = {
    "_362",
    "_526",
    "_527"
  };
  uint32_t dimensions__525[] = {1, 28, 28, 32};
  Qnn_Tensor_t outputs_Conv_31[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_525",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0152780450880527f, .offset= -114}}},
            .rank= 4,
            .dimensions=dimensions__525,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_31", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_31, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_31, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_31, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_32 */
  const char*  inputs_Add_32[] = {
    "_356",
    "_525"
  };
  uint32_t dimensions__365[] = {1, 28, 28, 32};
  Qnn_Tensor_t outputs_Add_32[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_365",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0293324608355761f, .offset= -133}}},
            .rank= 4,
            .dimensions=dimensions__365,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_32", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_32, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_32, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__529[] = {1, 1, 32, 192};
  VALIDATE(mobilenetv2_12.addTensor("_529", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_529",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0015565888024867f, .offset= -115}}},
                                          .rank= 4,
                                          .dimensions=dimensions__529,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_529),
                                                         .dataSize=BINLEN(_529)}}}}}
  ), err);
  uint32_t dimensions__530[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_530", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_530",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001450446f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__530,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_530),
                                                         .dataSize=BINLEN(_530)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_33 */
  uint32_t dimensions_Conv_33_dilation[] = {2};
  uint32_t Conv_33_dilation[] = {1, 1};
  uint32_t dimensions_Conv_33_pad_amount[] = {2, 2};
  uint32_t Conv_33_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_33_stride[] = {2};
  uint32_t Conv_33_stride[] = {1, 1};
  Qnn_Param_t params_Conv_33[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_33_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_33_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_33_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_33_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_33_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_33_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_33_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_33_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_33_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_33[] = {
    "_365",
    "_529",
    "_530"
  };
  uint32_t dimensions__368[] = {1, 28, 28, 192};
  Qnn_Tensor_t outputs_Conv_33[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_368",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0056755766272545f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__368,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_33", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_33, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_33, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_33, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__532[] = {3, 3, 1, 192};
  VALIDATE(mobilenetv2_12.addTensor("_532", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_532",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0174710247665644f, .offset= -140}}},
                                          .rank= 4,
                                          .dimensions=dimensions__532,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_532),
                                                         .dataSize=BINLEN(_532)}}}}}
  ), err);
  uint32_t dimensions__533[] = {192};
  VALIDATE(mobilenetv2_12.addTensor("_533", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_533",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000006631224f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__533,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_533),
                                                         .dataSize=BINLEN(_533)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_35 */
  uint32_t dimensions_Conv_35_dilation[] = {2};
  uint32_t Conv_35_dilation[] = {1, 1};
  uint32_t dimensions_Conv_35_pad_amount[] = {2, 2};
  uint32_t Conv_35_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_35_stride[] = {2};
  uint32_t Conv_35_stride[] = {2, 2};
  Qnn_Param_t params_Conv_35[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_35_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_35_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_35_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_35_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_35_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_35_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_35_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_35_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_35_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_35[] = {
    "_368",
    "_532",
    "_533"
  };
  uint32_t dimensions__371[] = {1, 14, 14, 192};
  Qnn_Tensor_t outputs_Conv_35[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_371",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0087239732965827f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__371,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_35", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_35, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_35, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_35, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__535[] = {1, 1, 192, 64};
  VALIDATE(mobilenetv2_12.addTensor("_535", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_535",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0049785454757512f, .offset= -116}}},
                                          .rank= 4,
                                          .dimensions=dimensions__535,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_535),
                                                         .dataSize=BINLEN(_535)}}}}}
  ), err);
  uint32_t dimensions__536[] = {64};
  VALIDATE(mobilenetv2_12.addTensor("_536", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_536",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000008809145f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__536,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_536),
                                                         .dataSize=BINLEN(_536)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_37 */
  uint32_t dimensions_Conv_37_dilation[] = {2};
  uint32_t Conv_37_dilation[] = {1, 1};
  uint32_t dimensions_Conv_37_pad_amount[] = {2, 2};
  uint32_t Conv_37_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_37_stride[] = {2};
  uint32_t Conv_37_stride[] = {1, 1};
  Qnn_Param_t params_Conv_37[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_37_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_37_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_37_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_37_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_37_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_37_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_37_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_37_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_37_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_37[] = {
    "_371",
    "_535",
    "_536"
  };
  uint32_t dimensions__534[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Conv_37[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_534",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0154282813891768f, .offset= -129}}},
            .rank= 4,
            .dimensions=dimensions__534,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_37", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_37, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_37, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_37, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__538[] = {1, 1, 64, 384};
  VALIDATE(mobilenetv2_12.addTensor("_538", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_538",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0010650438489392f, .offset= -122}}},
                                          .rank= 4,
                                          .dimensions=dimensions__538,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_538),
                                                         .dataSize=BINLEN(_538)}}}}}
  ), err);
  uint32_t dimensions__539[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_539", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_539",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000002093869f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__539,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_539),
                                                         .dataSize=BINLEN(_539)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_38 */
  uint32_t dimensions_Conv_38_dilation[] = {2};
  uint32_t Conv_38_dilation[] = {1, 1};
  uint32_t dimensions_Conv_38_pad_amount[] = {2, 2};
  uint32_t Conv_38_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_38_stride[] = {2};
  uint32_t Conv_38_stride[] = {1, 1};
  Qnn_Param_t params_Conv_38[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_38_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_38_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_38_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_38_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_38_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_38_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_38_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_38_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_38_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_38[] = {
    "_534",
    "_538",
    "_539"
  };
  uint32_t dimensions__376[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_38[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_376",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0032589877955616f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__376,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_38", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_38, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_38, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_38, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__541[] = {3, 3, 1, 384};
  VALIDATE(mobilenetv2_12.addTensor("_541", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_541",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0366222187876701f, .offset= -104}}},
                                          .rank= 4,
                                          .dimensions=dimensions__541,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_541),
                                                         .dataSize=BINLEN(_541)}}}}}
  ), err);
  uint32_t dimensions__542[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_542", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_542",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004640918f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__542,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_542),
                                                         .dataSize=BINLEN(_542)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_40 */
  uint32_t dimensions_Conv_40_dilation[] = {2};
  uint32_t Conv_40_dilation[] = {1, 1};
  uint32_t dimensions_Conv_40_pad_amount[] = {2, 2};
  uint32_t Conv_40_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_40_stride[] = {2};
  uint32_t Conv_40_stride[] = {1, 1};
  Qnn_Param_t params_Conv_40[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_40_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_40_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_40_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_40_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_40_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_40_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_40_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_40_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_40_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_40[] = {
    "_376",
    "_541",
    "_542"
  };
  uint32_t dimensions__379[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_40[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_379",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0046272482722998f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__379,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_40", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_40, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_40, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_40, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__544[] = {1, 1, 384, 64};
  VALIDATE(mobilenetv2_12.addTensor("_544", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_544",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0049987728707492f, .offset= -116}}},
                                          .rank= 4,
                                          .dimensions=dimensions__544,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_544),
                                                         .dataSize=BINLEN(_544)}}}}}
  ), err);
  uint32_t dimensions__545[] = {64};
  VALIDATE(mobilenetv2_12.addTensor("_545", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_545",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005330699f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__545,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_545),
                                                         .dataSize=BINLEN(_545)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_42 */
  uint32_t dimensions_Conv_42_dilation[] = {2};
  uint32_t Conv_42_dilation[] = {1, 1};
  uint32_t dimensions_Conv_42_pad_amount[] = {2, 2};
  uint32_t Conv_42_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_42_stride[] = {2};
  uint32_t Conv_42_stride[] = {1, 1};
  Qnn_Param_t params_Conv_42[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_42_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_42_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_42_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_42_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_42_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_42_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_42_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_42_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_42_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_42[] = {
    "_379",
    "_544",
    "_545"
  };
  uint32_t dimensions__543[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Conv_42[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_543",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0122793549671769f, .offset= -137}}},
            .rank= 4,
            .dimensions=dimensions__543,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_42", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_42, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_42, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_42, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_43 */
  const char*  inputs_Add_43[] = {
    "_534",
    "_543"
  };
  uint32_t dimensions__382[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Add_43[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_382",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0173948071897030f, .offset= -131}}},
            .rank= 4,
            .dimensions=dimensions__382,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_43", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_43, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_43, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__547[] = {1, 1, 64, 384};
  VALIDATE(mobilenetv2_12.addTensor("_547", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_547",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0007834002026357f, .offset= -117}}},
                                          .rank= 4,
                                          .dimensions=dimensions__547,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_547),
                                                         .dataSize=BINLEN(_547)}}}}}
  ), err);
  uint32_t dimensions__548[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_548", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_548",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001152372f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__548,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_548),
                                                         .dataSize=BINLEN(_548)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_44 */
  uint32_t dimensions_Conv_44_dilation[] = {2};
  uint32_t Conv_44_dilation[] = {1, 1};
  uint32_t dimensions_Conv_44_pad_amount[] = {2, 2};
  uint32_t Conv_44_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_44_stride[] = {2};
  uint32_t Conv_44_stride[] = {1, 1};
  Qnn_Param_t params_Conv_44[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_44_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_44_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_44_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_44_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_44_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_44_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_44_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_44_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_44_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_44[] = {
    "_382",
    "_547",
    "_548"
  };
  uint32_t dimensions__385[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_44[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_385",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0025319640990347f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__385,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_44", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_44, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_44, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_44, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__550[] = {3, 3, 1, 384};
  VALIDATE(mobilenetv2_12.addTensor("_550", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_550",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0752885937690735f, .offset= -105}}},
                                          .rank= 4,
                                          .dimensions=dimensions__550,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_550),
                                                         .dataSize=BINLEN(_550)}}}}}
  ), err);
  uint32_t dimensions__551[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_551", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_551",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000008426787f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__551,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_551),
                                                         .dataSize=BINLEN(_551)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_46 */
  uint32_t dimensions_Conv_46_dilation[] = {2};
  uint32_t Conv_46_dilation[] = {1, 1};
  uint32_t dimensions_Conv_46_pad_amount[] = {2, 2};
  uint32_t Conv_46_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_46_stride[] = {2};
  uint32_t Conv_46_stride[] = {1, 1};
  Qnn_Param_t params_Conv_46[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_46_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_46_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_46_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_46_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_46_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_46_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_46_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_46_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_46_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_46[] = {
    "_385",
    "_550",
    "_551"
  };
  uint32_t dimensions__388[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_46[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_388",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0046611088328063f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__388,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_46", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_46, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_46, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_46, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__553[] = {1, 1, 384, 64};
  VALIDATE(mobilenetv2_12.addTensor("_553", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_553",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0053957933560014f, .offset= -130}}},
                                          .rank= 4,
                                          .dimensions=dimensions__553,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_553),
                                                         .dataSize=BINLEN(_553)}}}}}
  ), err);
  uint32_t dimensions__554[] = {64};
  VALIDATE(mobilenetv2_12.addTensor("_554", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_554",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004392550f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__554,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_554),
                                                         .dataSize=BINLEN(_554)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_48 */
  uint32_t dimensions_Conv_48_dilation[] = {2};
  uint32_t Conv_48_dilation[] = {1, 1};
  uint32_t dimensions_Conv_48_pad_amount[] = {2, 2};
  uint32_t Conv_48_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_48_stride[] = {2};
  uint32_t Conv_48_stride[] = {1, 1};
  Qnn_Param_t params_Conv_48[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_48_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_48_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_48_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_48_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_48_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_48_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_48_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_48_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_48_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_48[] = {
    "_388",
    "_553",
    "_554"
  };
  uint32_t dimensions__552[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Conv_48[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_552",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0100355809554458f, .offset= -144}}},
            .rank= 4,
            .dimensions=dimensions__552,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_48", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_48, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_48, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_48, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_49 */
  const char*  inputs_Add_49[] = {
    "_382",
    "_552"
  };
  uint32_t dimensions__391[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Add_49[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_391",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0224688332527876f, .offset= -142}}},
            .rank= 4,
            .dimensions=dimensions__391,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_49", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_49, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_49, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__556[] = {1, 1, 64, 384};
  VALIDATE(mobilenetv2_12.addTensor("_556", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_556",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0007902046781965f, .offset= -122}}},
                                          .rank= 4,
                                          .dimensions=dimensions__556,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_556),
                                                         .dataSize=BINLEN(_556)}}}}}
  ), err);
  uint32_t dimensions__557[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_557", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_557",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001769293f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__557,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_557),
                                                         .dataSize=BINLEN(_557)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_50 */
  uint32_t dimensions_Conv_50_dilation[] = {2};
  uint32_t Conv_50_dilation[] = {1, 1};
  uint32_t dimensions_Conv_50_pad_amount[] = {2, 2};
  uint32_t Conv_50_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_50_stride[] = {2};
  uint32_t Conv_50_stride[] = {1, 1};
  Qnn_Param_t params_Conv_50[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_50_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_50_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_50_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_50_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_50_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_50_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_50_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_50_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_50_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_50[] = {
    "_391",
    "_556",
    "_557"
  };
  uint32_t dimensions__394[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_50[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_394",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0028765629976988f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__394,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_50", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_50, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_50, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_50, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__559[] = {3, 3, 1, 384};
  VALIDATE(mobilenetv2_12.addTensor("_559", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_559",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0386356972157955f, .offset= -131}}},
                                          .rank= 4,
                                          .dimensions=dimensions__559,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_559),
                                                         .dataSize=BINLEN(_559)}}}}}
  ), err);
  uint32_t dimensions__560[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_560", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_560",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005135185f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__560,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_560),
                                                         .dataSize=BINLEN(_560)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_52 */
  uint32_t dimensions_Conv_52_dilation[] = {2};
  uint32_t Conv_52_dilation[] = {1, 1};
  uint32_t dimensions_Conv_52_pad_amount[] = {2, 2};
  uint32_t Conv_52_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_52_stride[] = {2};
  uint32_t Conv_52_stride[] = {1, 1};
  Qnn_Param_t params_Conv_52[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_52_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_52_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_52_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_52_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_52_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_52_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_52_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_52_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_52_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_52[] = {
    "_394",
    "_559",
    "_560"
  };
  uint32_t dimensions__397[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_52[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_397",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0065522859804332f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__397,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_52", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_52, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_52, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_52, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__562[] = {1, 1, 384, 64};
  VALIDATE(mobilenetv2_12.addTensor("_562", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_562",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0055364253930748f, .offset= -145}}},
                                          .rank= 4,
                                          .dimensions=dimensions__562,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_562),
                                                         .dataSize=BINLEN(_562)}}}}}
  ), err);
  uint32_t dimensions__563[] = {64};
  VALIDATE(mobilenetv2_12.addTensor("_563", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_563",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003814701f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__563,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_563),
                                                         .dataSize=BINLEN(_563)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_54 */
  uint32_t dimensions_Conv_54_dilation[] = {2};
  uint32_t Conv_54_dilation[] = {1, 1};
  uint32_t dimensions_Conv_54_pad_amount[] = {2, 2};
  uint32_t Conv_54_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_54_stride[] = {2};
  uint32_t Conv_54_stride[] = {1, 1};
  Qnn_Param_t params_Conv_54[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_54_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_54_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_54_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_54_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_54_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_54_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_54_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_54_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_54_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_54[] = {
    "_397",
    "_562",
    "_563"
  };
  uint32_t dimensions__561[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Conv_54[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_561",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0128543870523572f, .offset= -156}}},
            .rank= 4,
            .dimensions=dimensions__561,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_54", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_54, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_54, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_54, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_55 */
  const char*  inputs_Add_55[] = {
    "_391",
    "_561"
  };
  uint32_t dimensions__400[] = {1, 14, 14, 64};
  Qnn_Tensor_t outputs_Add_55[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_400",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0250759981572628f, .offset= -139}}},
            .rank= 4,
            .dimensions=dimensions__400,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_55", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_55, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_55, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__565[] = {1, 1, 64, 384};
  VALIDATE(mobilenetv2_12.addTensor("_565", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_565",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0010571226011962f, .offset= -134}}},
                                          .rank= 4,
                                          .dimensions=dimensions__565,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_565),
                                                         .dataSize=BINLEN(_565)}}}}}
  ), err);
  uint32_t dimensions__566[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_566", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_566",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001433263f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__566,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_566),
                                                         .dataSize=BINLEN(_566)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_56 */
  uint32_t dimensions_Conv_56_dilation[] = {2};
  uint32_t Conv_56_dilation[] = {1, 1};
  uint32_t dimensions_Conv_56_pad_amount[] = {2, 2};
  uint32_t Conv_56_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_56_stride[] = {2};
  uint32_t Conv_56_stride[] = {1, 1};
  Qnn_Param_t params_Conv_56[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_56_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_56_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_56_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_56_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_56_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_56_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_56_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_56_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_56_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_56[] = {
    "_400",
    "_565",
    "_566"
  };
  uint32_t dimensions__403[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_56[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_403",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0033100319560617f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__403,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_56", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_56, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_56, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_56, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__568[] = {3, 3, 1, 384};
  VALIDATE(mobilenetv2_12.addTensor("_568", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_568",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0458321794867516f, .offset= -142}}},
                                          .rank= 4,
                                          .dimensions=dimensions__568,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_568),
                                                         .dataSize=BINLEN(_568)}}}}}
  ), err);
  uint32_t dimensions__569[] = {384};
  VALIDATE(mobilenetv2_12.addTensor("_569", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_569",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000006780503f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__569,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_569),
                                                         .dataSize=BINLEN(_569)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_58 */
  uint32_t dimensions_Conv_58_dilation[] = {2};
  uint32_t Conv_58_dilation[] = {1, 1};
  uint32_t dimensions_Conv_58_pad_amount[] = {2, 2};
  uint32_t Conv_58_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_58_stride[] = {2};
  uint32_t Conv_58_stride[] = {1, 1};
  Qnn_Param_t params_Conv_58[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_58_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_58_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_58_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_58_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_58_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_58_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_58_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_58_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_58_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_58[] = {
    "_403",
    "_568",
    "_569"
  };
  uint32_t dimensions__406[] = {1, 14, 14, 384};
  Qnn_Tensor_t outputs_Conv_58[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_406",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0073174741119146f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__406,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_58", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_58, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_58, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_58, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__571[] = {1, 1, 384, 96};
  VALIDATE(mobilenetv2_12.addTensor("_571", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_571",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0042201448231936f, .offset= -108}}},
                                          .rank= 4,
                                          .dimensions=dimensions__571,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_571),
                                                         .dataSize=BINLEN(_571)}}}}}
  ), err);
  uint32_t dimensions__572[] = {96};
  VALIDATE(mobilenetv2_12.addTensor("_572", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_572",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003655247f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__572,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_572),
                                                         .dataSize=BINLEN(_572)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_60 */
  uint32_t dimensions_Conv_60_dilation[] = {2};
  uint32_t Conv_60_dilation[] = {1, 1};
  uint32_t dimensions_Conv_60_pad_amount[] = {2, 2};
  uint32_t Conv_60_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_60_stride[] = {2};
  uint32_t Conv_60_stride[] = {1, 1};
  Qnn_Param_t params_Conv_60[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_60_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_60_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_60_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_60_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_60_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_60_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_60_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_60_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_60_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_60[] = {
    "_406",
    "_571",
    "_572"
  };
  uint32_t dimensions__570[] = {1, 14, 14, 96};
  Qnn_Tensor_t outputs_Conv_60[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_570",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0140790343284607f, .offset= -129}}},
            .rank= 4,
            .dimensions=dimensions__570,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_60", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_60, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_60, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_60, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__574[] = {1, 1, 96, 576};
  VALIDATE(mobilenetv2_12.addTensor("_574", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_574",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0012536892900243f, .offset= -133}}},
                                          .rank= 4,
                                          .dimensions=dimensions__574,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_574),
                                                         .dataSize=BINLEN(_574)}}}}}
  ), err);
  uint32_t dimensions__575[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_575", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_575",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003164063f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__575,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_575),
                                                         .dataSize=BINLEN(_575)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_61 */
  uint32_t dimensions_Conv_61_dilation[] = {2};
  uint32_t Conv_61_dilation[] = {1, 1};
  uint32_t dimensions_Conv_61_pad_amount[] = {2, 2};
  uint32_t Conv_61_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_61_stride[] = {2};
  uint32_t Conv_61_stride[] = {1, 1};
  Qnn_Param_t params_Conv_61[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_61_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_61_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_61_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_61_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_61_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_61_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_61_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_61_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_61_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_61[] = {
    "_570",
    "_574",
    "_575"
  };
  uint32_t dimensions__411[] = {1, 14, 14, 576};
  Qnn_Tensor_t outputs_Conv_61[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_411",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0039677019231021f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__411,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_61", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_61, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_61, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_61, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__577[] = {3, 3, 1, 576};
  VALIDATE(mobilenetv2_12.addTensor("_577", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_577",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0700598731637001f, .offset= -141}}},
                                          .rank= 4,
                                          .dimensions=dimensions__577,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_577),
                                                         .dataSize=BINLEN(_577)}}}}}
  ), err);
  uint32_t dimensions__578[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_578", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_578",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000006187264f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__578,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_578),
                                                         .dataSize=BINLEN(_578)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_63 */
  uint32_t dimensions_Conv_63_dilation[] = {2};
  uint32_t Conv_63_dilation[] = {1, 1};
  uint32_t dimensions_Conv_63_pad_amount[] = {2, 2};
  uint32_t Conv_63_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_63_stride[] = {2};
  uint32_t Conv_63_stride[] = {1, 1};
  Qnn_Param_t params_Conv_63[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_63_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_63_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_63_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_63_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_63_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_63_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_63_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_63_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_63_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_63[] = {
    "_411",
    "_577",
    "_578"
  };
  uint32_t dimensions__414[] = {1, 14, 14, 576};
  Qnn_Tensor_t outputs_Conv_63[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_414",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0068314508534968f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__414,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_63", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_63, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_63, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_63, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__580[] = {1, 1, 576, 96};
  VALIDATE(mobilenetv2_12.addTensor("_580", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_580",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0036609319504350f, .offset= -137}}},
                                          .rank= 4,
                                          .dimensions=dimensions__580,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_580),
                                                         .dataSize=BINLEN(_580)}}}}}
  ), err);
  uint32_t dimensions__581[] = {96};
  VALIDATE(mobilenetv2_12.addTensor("_581", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_581",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003626696f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__581,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_581),
                                                         .dataSize=BINLEN(_581)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_65 */
  uint32_t dimensions_Conv_65_dilation[] = {2};
  uint32_t Conv_65_dilation[] = {1, 1};
  uint32_t dimensions_Conv_65_pad_amount[] = {2, 2};
  uint32_t Conv_65_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_65_stride[] = {2};
  uint32_t Conv_65_stride[] = {1, 1};
  Qnn_Param_t params_Conv_65[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_65_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_65_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_65_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_65_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_65_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_65_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_65_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_65_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_65_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_65[] = {
    "_414",
    "_580",
    "_581"
  };
  uint32_t dimensions__579[] = {1, 14, 14, 96};
  Qnn_Tensor_t outputs_Conv_65[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_579",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0122327338904142f, .offset= -131}}},
            .rank= 4,
            .dimensions=dimensions__579,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_65", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_65, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_65, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_65, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_66 */
  const char*  inputs_Add_66[] = {
    "_570",
    "_579"
  };
  uint32_t dimensions__417[] = {1, 14, 14, 96};
  Qnn_Tensor_t outputs_Add_66[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_417",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0183310210704803f, .offset= -133}}},
            .rank= 4,
            .dimensions=dimensions__417,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_66", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_66, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_66, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__583[] = {1, 1, 96, 576};
  VALIDATE(mobilenetv2_12.addTensor("_583", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_583",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0014394279569387f, .offset= -104}}},
                                          .rank= 4,
                                          .dimensions=dimensions__583,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_583),
                                                         .dataSize=BINLEN(_583)}}}}}
  ), err);
  uint32_t dimensions__584[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_584", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_584",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001857420f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__584,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_584),
                                                         .dataSize=BINLEN(_584)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_67 */
  uint32_t dimensions_Conv_67_dilation[] = {2};
  uint32_t Conv_67_dilation[] = {1, 1};
  uint32_t dimensions_Conv_67_pad_amount[] = {2, 2};
  uint32_t Conv_67_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_67_stride[] = {2};
  uint32_t Conv_67_stride[] = {1, 1};
  Qnn_Param_t params_Conv_67[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_67_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_67_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_67_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_67_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_67_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_67_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_67_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_67_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_67_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_67[] = {
    "_417",
    "_583",
    "_584"
  };
  uint32_t dimensions__420[] = {1, 14, 14, 576};
  Qnn_Tensor_t outputs_Conv_67[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_420",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0040480368770659f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__420,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_67", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_67, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_67, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_67, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__586[] = {3, 3, 1, 576};
  VALIDATE(mobilenetv2_12.addTensor("_586", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_586",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0559509582817554f, .offset= -153}}},
                                          .rank= 4,
                                          .dimensions=dimensions__586,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_586),
                                                         .dataSize=BINLEN(_586)}}}}}
  ), err);
  uint32_t dimensions__587[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_587", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_587",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004323155f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__587,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_587),
                                                         .dataSize=BINLEN(_587)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_69 */
  uint32_t dimensions_Conv_69_dilation[] = {2};
  uint32_t Conv_69_dilation[] = {1, 1};
  uint32_t dimensions_Conv_69_pad_amount[] = {2, 2};
  uint32_t Conv_69_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_69_stride[] = {2};
  uint32_t Conv_69_stride[] = {1, 1};
  Qnn_Param_t params_Conv_69[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_69_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_69_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_69_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_69_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_69_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_69_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_69_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_69_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_69_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_69[] = {
    "_420",
    "_586",
    "_587"
  };
  uint32_t dimensions__423[] = {1, 14, 14, 576};
  Qnn_Tensor_t outputs_Conv_69[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_423",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0087096439674497f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__423,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_69", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_69, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_69, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_69, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__589[] = {1, 1, 576, 96};
  VALIDATE(mobilenetv2_12.addTensor("_589", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_589",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0091442335397005f, .offset= -137}}},
                                          .rank= 4,
                                          .dimensions=dimensions__589,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_589),
                                                         .dataSize=BINLEN(_589)}}}}}
  ), err);
  uint32_t dimensions__590[] = {96};
  VALIDATE(mobilenetv2_12.addTensor("_590", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_590",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005086014f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__590,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_590),
                                                         .dataSize=BINLEN(_590)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_71 */
  uint32_t dimensions_Conv_71_dilation[] = {2};
  uint32_t Conv_71_dilation[] = {1, 1};
  uint32_t dimensions_Conv_71_pad_amount[] = {2, 2};
  uint32_t Conv_71_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_71_stride[] = {2};
  uint32_t Conv_71_stride[] = {1, 1};
  Qnn_Param_t params_Conv_71[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_71_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_71_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_71_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_71_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_71_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_71_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_71_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_71_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_71_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_71[] = {
    "_423",
    "_589",
    "_590"
  };
  uint32_t dimensions__588[] = {1, 14, 14, 96};
  Qnn_Tensor_t outputs_Conv_71[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_588",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0157923698425293f, .offset= -126}}},
            .rank= 4,
            .dimensions=dimensions__588,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_71", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_71, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_71, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_71, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_72 */
  const char*  inputs_Add_72[] = {
    "_417",
    "_588"
  };
  uint32_t dimensions__426[] = {1, 14, 14, 96};
  Qnn_Tensor_t outputs_Add_72[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_426",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0233675707131624f, .offset= -125}}},
            .rank= 4,
            .dimensions=dimensions__426,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_72", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_72, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_72, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__592[] = {1, 1, 96, 576};
  VALIDATE(mobilenetv2_12.addTensor("_592", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_592",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0013479562476277f, .offset= -121}}},
                                          .rank= 4,
                                          .dimensions=dimensions__592,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_592),
                                                         .dataSize=BINLEN(_592)}}}}}
  ), err);
  uint32_t dimensions__593[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_593", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_593",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001697144f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__593,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_593),
                                                         .dataSize=BINLEN(_593)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_73 */
  uint32_t dimensions_Conv_73_dilation[] = {2};
  uint32_t Conv_73_dilation[] = {1, 1};
  uint32_t dimensions_Conv_73_pad_amount[] = {2, 2};
  uint32_t Conv_73_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_73_stride[] = {2};
  uint32_t Conv_73_stride[] = {1, 1};
  Qnn_Param_t params_Conv_73[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_73_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_73_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_73_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_73_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_73_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_73_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_73_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_73_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_73_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_73[] = {
    "_426",
    "_592",
    "_593"
  };
  uint32_t dimensions__429[] = {1, 14, 14, 576};
  Qnn_Tensor_t outputs_Conv_73[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_429",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0041894298046827f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__429,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_73", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_73, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_73, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_73, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__595[] = {3, 3, 1, 576};
  VALIDATE(mobilenetv2_12.addTensor("_595", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_595",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0163178630173206f, .offset= -113}}},
                                          .rank= 4,
                                          .dimensions=dimensions__595,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_595),
                                                         .dataSize=BINLEN(_595)}}}}}
  ), err);
  uint32_t dimensions__596[] = {576};
  VALIDATE(mobilenetv2_12.addTensor("_596", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_596",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003343594f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__596,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_596),
                                                         .dataSize=BINLEN(_596)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_75 */
  uint32_t dimensions_Conv_75_dilation[] = {2};
  uint32_t Conv_75_dilation[] = {1, 1};
  uint32_t dimensions_Conv_75_pad_amount[] = {2, 2};
  uint32_t Conv_75_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_75_stride[] = {2};
  uint32_t Conv_75_stride[] = {2, 2};
  Qnn_Param_t params_Conv_75[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_75_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_75_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_75_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_75_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_75_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_75_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_75_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_75_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_75_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_75[] = {
    "_429",
    "_595",
    "_596"
  };
  uint32_t dimensions__432[] = {1, 7, 7, 576};
  Qnn_Tensor_t outputs_Conv_75[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_432",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0078383786603808f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__432,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_75", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_75, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_75, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_75, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__598[] = {1, 1, 576, 160};
  VALIDATE(mobilenetv2_12.addTensor("_598", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_598",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0024518675636500f, .offset= -125}}},
                                          .rank= 4,
                                          .dimensions=dimensions__598,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_598),
                                                         .dataSize=BINLEN(_598)}}}}}
  ), err);
  uint32_t dimensions__599[] = {160};
  VALIDATE(mobilenetv2_12.addTensor("_599", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_599",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005458168f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__599,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_599),
                                                         .dataSize=BINLEN(_599)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_77 */
  uint32_t dimensions_Conv_77_dilation[] = {2};
  uint32_t Conv_77_dilation[] = {1, 1};
  uint32_t dimensions_Conv_77_pad_amount[] = {2, 2};
  uint32_t Conv_77_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_77_stride[] = {2};
  uint32_t Conv_77_stride[] = {1, 1};
  Qnn_Param_t params_Conv_77[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_77_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_77_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_77_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_77_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_77_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_77_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_77_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_77_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_77_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_77[] = {
    "_432",
    "_598",
    "_599"
  };
  uint32_t dimensions__597[] = {1, 7, 7, 160};
  Qnn_Tensor_t outputs_Conv_77[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_597",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0086918193846941f, .offset= -131}}},
            .rank= 4,
            .dimensions=dimensions__597,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_77", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_77, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_77, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_77, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__601[] = {1, 1, 160, 960};
  VALIDATE(mobilenetv2_12.addTensor("_601", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_601",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0023674434050918f, .offset= -156}}},
                                          .rank= 4,
                                          .dimensions=dimensions__601,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_601),
                                                         .dataSize=BINLEN(_601)}}}}}
  ), err);
  uint32_t dimensions__602[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_602", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_602",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001610434f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__602,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_602),
                                                         .dataSize=BINLEN(_602)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_78 */
  uint32_t dimensions_Conv_78_dilation[] = {2};
  uint32_t Conv_78_dilation[] = {1, 1};
  uint32_t dimensions_Conv_78_pad_amount[] = {2, 2};
  uint32_t Conv_78_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_78_stride[] = {2};
  uint32_t Conv_78_stride[] = {1, 1};
  Qnn_Param_t params_Conv_78[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_78_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_78_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_78_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_78_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_78_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_78_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_78_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_78_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_78_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_78[] = {
    "_597",
    "_601",
    "_602"
  };
  uint32_t dimensions__437[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_78[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_437",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0034719898831099f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__437,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_78", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_78, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_78, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_78, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__604[] = {3, 3, 1, 960};
  VALIDATE(mobilenetv2_12.addTensor("_604", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_604",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0735570043325424f, .offset= -136}}},
                                          .rank= 4,
                                          .dimensions=dimensions__604,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_604),
                                                         .dataSize=BINLEN(_604)}}}}}
  ), err);
  uint32_t dimensions__605[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_605", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_605",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000003398322f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__605,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_605),
                                                         .dataSize=BINLEN(_605)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_80 */
  uint32_t dimensions_Conv_80_dilation[] = {2};
  uint32_t Conv_80_dilation[] = {1, 1};
  uint32_t dimensions_Conv_80_pad_amount[] = {2, 2};
  uint32_t Conv_80_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_80_stride[] = {2};
  uint32_t Conv_80_stride[] = {1, 1};
  Qnn_Param_t params_Conv_80[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_80_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_80_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_80_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_80_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_80_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_80_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_80_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_80_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_80_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_80[] = {
    "_437",
    "_604",
    "_605"
  };
  uint32_t dimensions__440[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_80[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_440",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0076447455212474f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__440,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_80", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_80, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_80, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_80, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__607[] = {1, 1, 960, 160};
  VALIDATE(mobilenetv2_12.addTensor("_607", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_607",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0024453334044665f, .offset= -134}}},
                                          .rank= 4,
                                          .dimensions=dimensions__607,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_607),
                                                         .dataSize=BINLEN(_607)}}}}}
  ), err);
  uint32_t dimensions__608[] = {160};
  VALIDATE(mobilenetv2_12.addTensor("_608", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_608",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000002434383f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__608,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_608),
                                                         .dataSize=BINLEN(_608)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_82 */
  uint32_t dimensions_Conv_82_dilation[] = {2};
  uint32_t Conv_82_dilation[] = {1, 1};
  uint32_t dimensions_Conv_82_pad_amount[] = {2, 2};
  uint32_t Conv_82_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_82_stride[] = {2};
  uint32_t Conv_82_stride[] = {1, 1};
  Qnn_Param_t params_Conv_82[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_82_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_82_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_82_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_82_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_82_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_82_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_82_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_82_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_82_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_82[] = {
    "_440",
    "_607",
    "_608"
  };
  uint32_t dimensions__606[] = {1, 7, 7, 160};
  Qnn_Tensor_t outputs_Conv_82[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_606",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0102385636419058f, .offset= -119}}},
            .rank= 4,
            .dimensions=dimensions__606,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_82", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_82, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_82, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_82, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_83 */
  const char*  inputs_Add_83[] = {
    "_597",
    "_606"
  };
  uint32_t dimensions__443[] = {1, 7, 7, 160};
  Qnn_Tensor_t outputs_Add_83[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_443",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0098382337018847f, .offset= -121}}},
            .rank= 4,
            .dimensions=dimensions__443,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_83", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_83, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_83, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__610[] = {1, 1, 160, 960};
  VALIDATE(mobilenetv2_12.addTensor("_610", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_610",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0015982929617167f, .offset= -121}}},
                                          .rank= 4,
                                          .dimensions=dimensions__610,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_610),
                                                         .dataSize=BINLEN(_610)}}}}}
  ), err);
  uint32_t dimensions__611[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_611", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_611",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001169134f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__611,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_611),
                                                         .dataSize=BINLEN(_611)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_84 */
  uint32_t dimensions_Conv_84_dilation[] = {2};
  uint32_t Conv_84_dilation[] = {1, 1};
  uint32_t dimensions_Conv_84_pad_amount[] = {2, 2};
  uint32_t Conv_84_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_84_stride[] = {2};
  uint32_t Conv_84_stride[] = {1, 1};
  Qnn_Param_t params_Conv_84[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_84_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_84_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_84_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_84_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_84_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_84_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_84_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_84_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_84_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_84[] = {
    "_443",
    "_610",
    "_611"
  };
  uint32_t dimensions__446[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_84[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_446",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0035441098734736f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__446,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_84", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_84, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_84, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_84, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__613[] = {3, 3, 1, 960};
  VALIDATE(mobilenetv2_12.addTensor("_613", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_613",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0573238059878349f, .offset= -127}}},
                                          .rank= 4,
                                          .dimensions=dimensions__613,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_613),
                                                         .dataSize=BINLEN(_613)}}}}}
  ), err);
  uint32_t dimensions__614[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_614", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_614",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004500377f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__614,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_614),
                                                         .dataSize=BINLEN(_614)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_86 */
  uint32_t dimensions_Conv_86_dilation[] = {2};
  uint32_t Conv_86_dilation[] = {1, 1};
  uint32_t dimensions_Conv_86_pad_amount[] = {2, 2};
  uint32_t Conv_86_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_86_stride[] = {2};
  uint32_t Conv_86_stride[] = {1, 1};
  Qnn_Param_t params_Conv_86[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_86_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_86_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_86_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_86_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_86_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_86_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_86_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_86_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_86_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_86[] = {
    "_446",
    "_613",
    "_614"
  };
  uint32_t dimensions__449[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_86[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_449",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0125177316367626f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__449,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_86", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_86, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_86, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_86, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__616[] = {1, 1, 960, 160};
  VALIDATE(mobilenetv2_12.addTensor("_616", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_616",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0045087877660990f, .offset= -135}}},
                                          .rank= 4,
                                          .dimensions=dimensions__616,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_616),
                                                         .dataSize=BINLEN(_616)}}}}}
  ), err);
  uint32_t dimensions__617[] = {160};
  VALIDATE(mobilenetv2_12.addTensor("_617", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_617",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000004267541f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__617,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_617),
                                                         .dataSize=BINLEN(_617)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_88 */
  uint32_t dimensions_Conv_88_dilation[] = {2};
  uint32_t Conv_88_dilation[] = {1, 1};
  uint32_t dimensions_Conv_88_pad_amount[] = {2, 2};
  uint32_t Conv_88_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_88_stride[] = {2};
  uint32_t Conv_88_stride[] = {1, 1};
  Qnn_Param_t params_Conv_88[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_88_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_88_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_88_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_88_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_88_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_88_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_88_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_88_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_88_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_88[] = {
    "_449",
    "_616",
    "_617"
  };
  uint32_t dimensions__615[] = {1, 7, 7, 160};
  Qnn_Tensor_t outputs_Conv_88[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_615",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0120670022442937f, .offset= -128}}},
            .rank= 4,
            .dimensions=dimensions__615,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_88", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_88, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_88, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_88, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR Add_89 */
  const char*  inputs_Add_89[] = {
    "_443",
    "_615"
  };
  uint32_t dimensions__452[] = {1, 7, 7, 160};
  Qnn_Tensor_t outputs_Add_89[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_452",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0167211629450321f, .offset= -138}}},
            .rank= 4,
            .dimensions=dimensions__452,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Add_89", // Node Name
                                  "qti.aisw", // Package Name
                                  "ElementWiseAdd", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Add_89, // Input Tensor Names
                                  2, // Num Input Tensor Names
                                  outputs_Add_89, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__619[] = {1, 1, 160, 960};
  VALIDATE(mobilenetv2_12.addTensor("_619", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_619",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0010245944140479f, .offset= -126}}},
                                          .rank= 4,
                                          .dimensions=dimensions__619,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_619),
                                                         .dataSize=BINLEN(_619)}}}}}
  ), err);
  uint32_t dimensions__620[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_620", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_620",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001278459f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__620,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_620),
                                                         .dataSize=BINLEN(_620)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_90 */
  uint32_t dimensions_Conv_90_dilation[] = {2};
  uint32_t Conv_90_dilation[] = {1, 1};
  uint32_t dimensions_Conv_90_pad_amount[] = {2, 2};
  uint32_t Conv_90_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_90_stride[] = {2};
  uint32_t Conv_90_stride[] = {1, 1};
  Qnn_Param_t params_Conv_90[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_90_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_90_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_90_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_90_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_90_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_90_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_90_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_90_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_90_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_90[] = {
    "_452",
    "_619",
    "_620"
  };
  uint32_t dimensions__455[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_90[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_455",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0029152960050851f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__455,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_90", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_90, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_90, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_90, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__622[] = {3, 3, 1, 960};
  VALIDATE(mobilenetv2_12.addTensor("_622", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_622",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0585044100880623f, .offset= -102}}},
                                          .rank= 4,
                                          .dimensions=dimensions__622,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_622),
                                                         .dataSize=BINLEN(_622)}}}}}
  ), err);
  uint32_t dimensions__623[] = {960};
  VALIDATE(mobilenetv2_12.addTensor("_623", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_623",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000002422212f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__623,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_623),
                                                         .dataSize=BINLEN(_623)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_92 */
  uint32_t dimensions_Conv_92_dilation[] = {2};
  uint32_t Conv_92_dilation[] = {1, 1};
  uint32_t dimensions_Conv_92_pad_amount[] = {2, 2};
  uint32_t Conv_92_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_Conv_92_stride[] = {2};
  uint32_t Conv_92_stride[] = {1, 1};
  Qnn_Param_t params_Conv_92[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_92_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_92_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_92_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_92_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_92_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_92_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_92_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_92_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_92_stride,
                           .dataSize=8}}}}}}}
  };
  const char*  inputs_Conv_92[] = {
    "_455",
    "_622",
    "_623"
  };
  uint32_t dimensions__458[] = {1, 7, 7, 960};
  Qnn_Tensor_t outputs_Conv_92[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_458",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0034981076605618f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__458,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_92", // Node Name
                                  "qti.aisw", // Package Name
                                  "DepthWiseConv2d", // Qnn Node Type
                                  params_Conv_92, // Node Params
                                  3, // Num Node Params
                                  inputs_Conv_92, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_92, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__625[] = {1, 1, 960, 320};
  VALIDATE(mobilenetv2_12.addTensor("_625", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_625",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0048637287691236f, .offset= -133}}},
                                          .rank= 4,
                                          .dimensions=dimensions__625,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_625),
                                                         .dataSize=BINLEN(_625)}}}}}
  ), err);
  uint32_t dimensions__626[] = {320};
  VALIDATE(mobilenetv2_12.addTensor("_626", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_626",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000005163378f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__626,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_626),
                                                         .dataSize=BINLEN(_626)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_94 */
  uint32_t dimensions_Conv_94_dilation[] = {2};
  uint32_t Conv_94_dilation[] = {1, 1};
  uint32_t dimensions_Conv_94_pad_amount[] = {2, 2};
  uint32_t Conv_94_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_94_stride[] = {2};
  uint32_t Conv_94_stride[] = {1, 1};
  Qnn_Param_t params_Conv_94[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_94_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_94_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_94_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_94_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_94_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_94_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_94_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_94_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_94_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_94[] = {
    "_458",
    "_625",
    "_626"
  };
  uint32_t dimensions__624[] = {1, 7, 7, 320};
  Qnn_Tensor_t outputs_Conv_94[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_624",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0087862741202116f, .offset= -138}}},
            .rank= 4,
            .dimensions=dimensions__624,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_94", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_94, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_94, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_94, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions__628[] = {1, 1, 320, 1280};
  VALIDATE(mobilenetv2_12.addTensor("_628", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_628",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0135058313608170f, .offset= -130}}},
                                          .rank= 4,
                                          .dimensions=dimensions__628,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_628),
                                                         .dataSize=BINLEN(_628)}}}}}
  ), err);
  uint32_t dimensions__629[] = {1280};
  VALIDATE(mobilenetv2_12.addTensor("_629", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "_629",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000001350600f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions__629,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(_629),
                                                         .dataSize=BINLEN(_629)}}}}}
  ), err);

  /* ADDING NODE FOR Conv_95 */
  uint32_t dimensions_Conv_95_dilation[] = {2};
  uint32_t Conv_95_dilation[] = {1, 1};
  uint32_t dimensions_Conv_95_pad_amount[] = {2, 2};
  uint32_t Conv_95_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_Conv_95_stride[] = {2};
  uint32_t Conv_95_stride[] = {1, 1};
  Qnn_Param_t params_Conv_95[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_95_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_95_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_95_dilation,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_95_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_Conv_95_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_95_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "Conv_95_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_Conv_95_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)Conv_95_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}}
  };
  const char*  inputs_Conv_95[] = {
    "_624",
    "_628",
    "_629"
  };
  uint32_t dimensions__463[] = {1, 7, 7, 1280};
  Qnn_Tensor_t outputs_Conv_95[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_463",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0235294122248888f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__463,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Conv_95", // Node Name
                                  "qti.aisw", // Package Name
                                  "Conv2d", // Qnn Node Type
                                  params_Conv_95, // Node Params
                                  4, // Num Node Params
                                  inputs_Conv_95, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Conv_95, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR GlobalAveragePool_97 */
  uint32_t dimensions_GlobalAveragePool_97_filter_size[] = {2};
  uint32_t GlobalAveragePool_97_filter_size[] = {7, 7};
  uint32_t dimensions_GlobalAveragePool_97_pad_amount[] = {2, 2};
  uint32_t GlobalAveragePool_97_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_GlobalAveragePool_97_stride[] = {2};
  uint32_t GlobalAveragePool_97_stride[] = {7, 7};
  Qnn_Param_t params_GlobalAveragePool_97[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "GlobalAveragePool_97_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_GlobalAveragePool_97_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)GlobalAveragePool_97_filter_size,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "GlobalAveragePool_97_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_GlobalAveragePool_97_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)GlobalAveragePool_97_pad_amount,
                           .dataSize=16}}}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "GlobalAveragePool_97_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_GlobalAveragePool_97_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)GlobalAveragePool_97_stride,
                           .dataSize=8}}}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="count_pad_for_edges",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_GlobalAveragePool_97[] = {
    "_463"
  };
  uint32_t dimensions__464[] = {1, 1, 1, 1280};
  Qnn_Tensor_t outputs_GlobalAveragePool_97[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_464",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0103086791932583f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__464,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "GlobalAveragePool_97", // Node Name
                                  "qti.aisw", // Package Name
                                  "PoolAvg2d", // Qnn Node Type
                                  params_GlobalAveragePool_97, // Node Params
                                  4, // Num Node Params
                                  inputs_GlobalAveragePool_97, // Input Tensor Names
                                  1, // Num Input Tensor Names
                                  outputs_GlobalAveragePool_97, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  /* ADDING NODE FOR _464_ncs */
  uint32_t dimensions___464_ncs_perm[] = {4};
  uint32_t __464_ncs_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params__464_ncs[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "__464_ncs_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions___464_ncs_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)__464_ncs_perm,
                           .dataSize=16}}}}}}}
  };
  const char*  inputs__464_ncs[] = {
    "_464"
  };
  uint32_t dimensions__464_ncs[] = {1, 1280, 1, 1};
  Qnn_Tensor_t outputs__464_ncs[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "_464_ncs",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0103086791932583f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions__464_ncs,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "_464_ncs", // Node Name
                                  "qti.aisw", // Package Name
                                  "Transpose", // Qnn Node Type
                                  params__464_ncs, // Node Params
                                  1, // Num Node Params
                                  inputs__464_ncs, // Input Tensor Names
                                  1, // Num Input Tensor Names
                                  outputs__464_ncs, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);

  uint32_t dimensions_classifier_1_weight[] = {1000, 1280};
  VALIDATE(mobilenetv2_12.addTensor("classifier_1_weight", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "classifier_1_weight",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_UFIXED_POINT_8,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0022603925317526f, .offset= -109}}},
                                          .rank= 2,
                                          .dimensions=dimensions_classifier_1_weight,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(classifier_1_weight),
                                                         .dataSize=BINLEN(classifier_1_weight)}}}}}
  ), err);
  uint32_t dimensions_classifier_1_bias[] = {1000};
  VALIDATE(mobilenetv2_12.addTensor("classifier_1_bias", // Node Name
                                    (Qnn_Tensor_t) {
                                        .version= QNN_TENSOR_VERSION_1,
                                        {.v1= {
                                          .id=0,
                                          .name= "classifier_1_bias",
                                          .type= QNN_TENSOR_TYPE_STATIC,
                                          .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                          .dataType= QNN_DATATYPE_SFIXED_POINT_32,
                                          .quantizeParams= { QNN_DEFINITION_DEFINED,
                                                             QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                                             {.scaleOffsetEncoding= {.scale= 0.0000000000722255f, .offset= 0}}},
                                          .rank= 1,
                                          .dimensions=dimensions_classifier_1_bias,
                                          .memType= QNN_TENSORMEMTYPE_RAW,
                                          {.clientBuf= { .data=BINVARSTART(classifier_1_bias),
                                                         .dataSize=BINLEN(classifier_1_bias)}}}}}
  ), err);

  /* ADDING NODE FOR Gemm_104 */
  const char*  inputs_Gemm_104[] = {
    "_464_ncs",
    "classifier_1_weight",
    "classifier_1_bias"
  };
  uint32_t dimensions_output[] = {1, 1000};
  Qnn_Tensor_t outputs_Gemm_104[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_1,
          {.v1= {
            .id=0,
            .name= "output",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType= QNN_DATATYPE_UFIXED_POINT_8,
            .quantizeParams= { QNN_DEFINITION_DEFINED,
                               QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                               {.scaleOffsetEncoding= {.scale= 0.0806941762566566f, .offset= -86}}},
            .rank= 2,
            .dimensions=dimensions_output,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}}}}}
  };
  VALIDATE(mobilenetv2_12.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                                  "Gemm_104", // Node Name
                                  "qti.aisw", // Package Name
                                  "FullyConnected", // Qnn Node Type
                                  nullptr, // Node Params
                                  0, // Num Node Params
                                  inputs_Gemm_104, // Input Tensor Names
                                  3, // Num Input Tensor Names
                                  outputs_Gemm_104, // Output Tensors 
                                  1// Num Output Tensors 
  ), err);


  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&mobilenetv2_12};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}