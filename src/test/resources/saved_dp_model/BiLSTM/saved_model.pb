??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
?
/bi_lstm_model/FirstBlockLSTMModule/w_first_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*@
shared_name1/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm
?
Cbi_lstm_model/FirstBlockLSTMModule/w_first_lstm/Read/ReadVariableOpReadVariableOp/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm* 
_output_shapes
:
??*
dtype0
?
1bi_lstm_model/FirstBlockLSTMModule/wig_first_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*B
shared_name31bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm
?
Ebi_lstm_model/FirstBlockLSTMModule/wig_first_lstm/Read/ReadVariableOpReadVariableOp1bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm*
_output_shapes
:~*
dtype0
?
1bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*B
shared_name31bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm
?
Ebi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm/Read/ReadVariableOpReadVariableOp1bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm*
_output_shapes
:~*
dtype0
?
1bi_lstm_model/FirstBlockLSTMModule/wog_first_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*B
shared_name31bi_lstm_model/FirstBlockLSTMModule/wog_first_lstm
?
Ebi_lstm_model/FirstBlockLSTMModule/wog_first_lstm/Read/ReadVariableOpReadVariableOp1bi_lstm_model/FirstBlockLSTMModule/wog_first_lstm*
_output_shapes
:~*
dtype0
?
'bi_lstm_model/NextBlockLSTM/w_next_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*8
shared_name)'bi_lstm_model/NextBlockLSTM/w_next_lstm
?
;bi_lstm_model/NextBlockLSTM/w_next_lstm/Read/ReadVariableOpReadVariableOp'bi_lstm_model/NextBlockLSTM/w_next_lstm* 
_output_shapes
:
??*
dtype0
?
)bi_lstm_model/NextBlockLSTM/wig_next_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*:
shared_name+)bi_lstm_model/NextBlockLSTM/wig_next_lstm
?
=bi_lstm_model/NextBlockLSTM/wig_next_lstm/Read/ReadVariableOpReadVariableOp)bi_lstm_model/NextBlockLSTM/wig_next_lstm*
_output_shapes
:~*
dtype0
?
)bi_lstm_model/NextBlockLSTM/wfg_next_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*:
shared_name+)bi_lstm_model/NextBlockLSTM/wfg_next_lstm
?
=bi_lstm_model/NextBlockLSTM/wfg_next_lstm/Read/ReadVariableOpReadVariableOp)bi_lstm_model/NextBlockLSTM/wfg_next_lstm*
_output_shapes
:~*
dtype0
?
)bi_lstm_model/NextBlockLSTM/wog_next_lstmVarHandleOp*
_output_shapes
: *
dtype0*
shape:~*:
shared_name+)bi_lstm_model/NextBlockLSTM/wog_next_lstm
?
=bi_lstm_model/NextBlockLSTM/wog_next_lstm/Read/ReadVariableOpReadVariableOp)bi_lstm_model/NextBlockLSTM/wog_next_lstm*
_output_shapes
:~*
dtype0
?
ConstConst*
_output_shapes

:~*
dtype0*?
value?B?~"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
?
Const_1Const*
_output_shapes

:~*
dtype0*?
value?B?~"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
?
Const_2Const*
_output_shapes	
:?*
dtype0*?
value?B??"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
?
Const_3Const*
_output_shapes

:~*
dtype0*?
value?B?~"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
?
Const_4Const*
_output_shapes

:~*
dtype0*?
value?B?~"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
?
Const_5Const*
_output_shapes	
:?*
dtype0*?
value?B??"?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

NoOpNoOp
?
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	blockLstm
nextBlockLstm
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?
weight_matrix
	weight_input_gate

weight_forget_gate
weight_output_gate
	variables
regularization_losses
trainable_variables
	keras_api
?
weight_matrix
weight_input_gate
weight_forget_gate
weight_output_gate
	variables
regularization_losses
trainable_variables
	keras_api
8
0
	1

2
3
4
5
6
7
 
8
0
	1

2
3
4
5
6
7
?
layer_regularization_losses
	variables
metrics

layers
regularization_losses
trainable_variables
non_trainable_variables
layer_metrics
 
wu
VARIABLE_VALUE/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm2blockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE1bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm6blockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE1bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm7blockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE1bi_lstm_model/FirstBlockLSTMModule/wog_first_lstm7blockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUE

0
	1

2
3
 

0
	1

2
3
?
layer_regularization_losses
	variables
metrics

layers
regularization_losses
trainable_variables
 non_trainable_variables
!layer_metrics
sq
VARIABLE_VALUE'bi_lstm_model/NextBlockLSTM/w_next_lstm6nextBlockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE)bi_lstm_model/NextBlockLSTM/wig_next_lstm:nextBlockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE)bi_lstm_model/NextBlockLSTM/wfg_next_lstm;nextBlockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE)bi_lstm_model/NextBlockLSTM/wog_next_lstm;nextBlockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
2
3
?
"layer_regularization_losses
	variables
#metrics

$layers
regularization_losses
trainable_variables
%non_trainable_variables
&layer_metrics
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????d*
dtype0* 
shape:?????????d
?
serving_default_input_2Placeholder*+
_output_shapes
:?????????d*
dtype0* 
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2ConstConst_1/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wog_first_lstmConst_2Const_3Const_4'bi_lstm_model/NextBlockLSTM/w_next_lstm)bi_lstm_model/NextBlockLSTM/wig_next_lstm)bi_lstm_model/NextBlockLSTM/wfg_next_lstm)bi_lstm_model/NextBlockLSTM/wog_next_lstmConst_5*
Tin
2*8
Tout0
.2,*
_collective_manager_ids
 *?
_output_shapes?
?:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_122101994
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameCbi_lstm_model/FirstBlockLSTMModule/w_first_lstm/Read/ReadVariableOpEbi_lstm_model/FirstBlockLSTMModule/wig_first_lstm/Read/ReadVariableOpEbi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm/Read/ReadVariableOpEbi_lstm_model/FirstBlockLSTMModule/wog_first_lstm/Read/ReadVariableOp;bi_lstm_model/NextBlockLSTM/w_next_lstm/Read/ReadVariableOp=bi_lstm_model/NextBlockLSTM/wig_next_lstm/Read/ReadVariableOp=bi_lstm_model/NextBlockLSTM/wfg_next_lstm/Read/ReadVariableOp=bi_lstm_model/NextBlockLSTM/wog_next_lstm/Read/ReadVariableOpConst_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_save_122102593
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm1bi_lstm_model/FirstBlockLSTMModule/wog_first_lstm'bi_lstm_model/NextBlockLSTM/w_next_lstm)bi_lstm_model/NextBlockLSTM/wig_next_lstm)bi_lstm_model/NextBlockLSTM/wfg_next_lstm)bi_lstm_model/NextBlockLSTM/wog_next_lstm*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference__traced_restore_122102627??
??

?
$__inference__wrapped_model_122100896
input_1
input_28
4bi_lstm_model_firstblocklstmmodule_blocklstm_cs_prev7
3bi_lstm_model_firstblocklstmmodule_blocklstm_h_prevH
Dbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_resourceJ
Fbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_1_resourceJ
Fbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_2_resourceJ
Fbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_3_resource2
.bi_lstm_model_firstblocklstmmodule_blocklstm_b1
-bi_lstm_model_nextblocklstm_blocklstm_cs_prev0
,bi_lstm_model_nextblocklstm_blocklstm_h_prevA
=bi_lstm_model_nextblocklstm_blocklstm_readvariableop_resourceC
?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_1_resourceC
?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_2_resourceC
?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_3_resource+
'bi_lstm_model_nextblocklstm_blocklstm_b
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43??
0bi_lstm_model/FirstBlockLSTMModule/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   22
0bi_lstm_model/FirstBlockLSTMModule/Reshape/shape?
*bi_lstm_model/FirstBlockLSTMModule/ReshapeReshapeinput_19bi_lstm_model/FirstBlockLSTMModule/Reshape/shape:output:0*
T0*"
_output_shapes
:d2,
*bi_lstm_model/FirstBlockLSTMModule/Reshape?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_1/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_1Reshapeinput_2;bi_lstm_model/FirstBlockLSTMModule/Reshape_1/shape:output:0*
T0*"
_output_shapes
:d2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_1?
8bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2:
8bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/seq_len_max?
;bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOpReadVariableOpDbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_1ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_1?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_2ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_2?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_3ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_3?
,bi_lstm_model/FirstBlockLSTMModule/BlockLSTM	BlockLSTMAbi_lstm_model/FirstBlockLSTMModule/BlockLSTM/seq_len_max:output:03bi_lstm_model/FirstBlockLSTMModule/Reshape:output:04bi_lstm_model_firstblocklstmmodule_blocklstm_cs_prev3bi_lstm_model_firstblocklstmmodule_blocklstm_h_prevCbi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp:value:0Ebi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_1:value:0Ebi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_2:value:0Ebi_lstm_model/FirstBlockLSTMModule/BlockLSTM/ReadVariableOp_3:value:0.bi_lstm_model_firstblocklstmmodule_blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2.
,bi_lstm_model/FirstBlockLSTMModule/BlockLSTM?
:bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/seq_len_max?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOpReadVariableOpDbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02?
=bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp?
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_1ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02A
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_1?
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_2ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02A
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_2?
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_3ReadVariableOpFbi_lstm_model_firstblocklstmmodule_blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02A
?bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_3?
.bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1	BlockLSTMCbi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/seq_len_max:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_1:output:04bi_lstm_model_firstblocklstmmodule_blocklstm_cs_prev3bi_lstm_model_firstblocklstmmodule_blocklstm_h_prevEbi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp:value:0Gbi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_1:value:0Gbi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_2:value:0Gbi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1/ReadVariableOp_3:value:0.bi_lstm_model_firstblocklstmmodule_blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~20
.bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_2/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_2Reshape0bi_lstm_model/FirstBlockLSTMModule/BlockLSTM:h:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_2?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_3/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_3Reshape2bi_lstm_model/FirstBlockLSTMModule/BlockLSTM_1:h:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_3?
6bi_lstm_model/FirstBlockLSTMModule/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_1?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_2?
0bi_lstm_model/FirstBlockLSTMModule/strided_sliceStridedSlice5bi_lstm_model/FirstBlockLSTMModule/Reshape_2:output:0?bi_lstm_model/FirstBlockLSTMModule/strided_slice/stack:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_1:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask22
0bi_lstm_model/FirstBlockLSTMModule/strided_slice?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_2?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_1StridedSlice5bi_lstm_model/FirstBlockLSTMModule/Reshape_3:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_1?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_2?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_2StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_2?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stackConst*
_output_shapes
: *
dtype0*
value	B :2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_2?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_4?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_3StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_3:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_3/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_3?
.bi_lstm_model/FirstBlockLSTMModule/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.bi_lstm_model/FirstBlockLSTMModule/concat/axis?
)bi_lstm_model/FirstBlockLSTMModule/concatConcatV2;bi_lstm_model/FirstBlockLSTMModule/strided_slice_2:output:0;bi_lstm_model/FirstBlockLSTMModule/strided_slice_3:output:07bi_lstm_model/FirstBlockLSTMModule/concat/axis:output:0*
N*
T0*
_output_shapes	
:?2+
)bi_lstm_model/FirstBlockLSTMModule/concat?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_4/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_4Reshape2bi_lstm_model/FirstBlockLSTMModule/concat:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_4/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_4?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_2?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_4StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_4?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stackConst*
_output_shapes
: *
dtype0*
value	B :2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_2?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_4?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_5StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_3:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_5/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_5?
0bi_lstm_model/FirstBlockLSTMModule/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_1/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_1ConcatV2;bi_lstm_model/FirstBlockLSTMModule/strided_slice_4:output:0;bi_lstm_model/FirstBlockLSTMModule/strided_slice_5:output:09bi_lstm_model/FirstBlockLSTMModule/concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_1?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_5/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_5Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_1:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_5/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_5?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_2?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_6StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_6?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stackConst*
_output_shapes
: *
dtype0*
value	B :2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_2?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_4?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_7StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_3:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_7/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_7?
0bi_lstm_model/FirstBlockLSTMModule/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_2/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_2ConcatV2;bi_lstm_model/FirstBlockLSTMModule/strided_slice_6:output:0;bi_lstm_model/FirstBlockLSTMModule/strided_slice_7:output:09bi_lstm_model/FirstBlockLSTMModule/concat_2/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_2?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_6/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_6Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_2:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_6/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_6?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_2?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_8StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Abi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_8?
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stackConst*
_output_shapes
: *
dtype0*
value	B :2:
8bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_2?
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3/values_0?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3PackLbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3?
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2<
:bi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_4?
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_9StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_1:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_3:output:0Cbi_lstm_model/FirstBlockLSTMModule/strided_slice_9/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask24
2bi_lstm_model/FirstBlockLSTMModule/strided_slice_9?
0bi_lstm_model/FirstBlockLSTMModule/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_3/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_3ConcatV2;bi_lstm_model/FirstBlockLSTMModule/strided_slice_8:output:0;bi_lstm_model/FirstBlockLSTMModule/strided_slice_9:output:09bi_lstm_model/FirstBlockLSTMModule/concat_3/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_3?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_7/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_7Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_3:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_7/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_7?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_10StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_10?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_11StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_11/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_11?
0bi_lstm_model/FirstBlockLSTMModule/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_4/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_4ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_10:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_11:output:09bi_lstm_model/FirstBlockLSTMModule/concat_4/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_4?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_8/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_8Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_4:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_8/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_8?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_12StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_12?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_13StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_13/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_13?
0bi_lstm_model/FirstBlockLSTMModule/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_5/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_5ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_12:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_13:output:09bi_lstm_model/FirstBlockLSTMModule/concat_5/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_5?
2bi_lstm_model/FirstBlockLSTMModule/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   24
2bi_lstm_model/FirstBlockLSTMModule/Reshape_9/shape?
,bi_lstm_model/FirstBlockLSTMModule/Reshape_9Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_5:output:0;bi_lstm_model/FirstBlockLSTMModule/Reshape_9/shape:output:0*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/Reshape_9?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_14StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_14?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_15StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_15/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_15?
0bi_lstm_model/FirstBlockLSTMModule/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_6/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_6ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_14:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_15:output:09bi_lstm_model/FirstBlockLSTMModule/concat_6/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_6?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_10/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_10Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_6:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_10/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_10?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_16StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_16?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_17StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_17/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_17?
0bi_lstm_model/FirstBlockLSTMModule/concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_7/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_7ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_16:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_17:output:09bi_lstm_model/FirstBlockLSTMModule/concat_7/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_7?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_11/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_11Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_7:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_11/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_11?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_18StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_18?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_19StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_19/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_19?
0bi_lstm_model/FirstBlockLSTMModule/concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_8/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_8ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_18:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_19:output:09bi_lstm_model/FirstBlockLSTMModule/concat_8/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_8?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_12/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_12Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_8:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_12/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_12?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:	2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_20StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_20?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_21StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_21/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_21?
0bi_lstm_model/FirstBlockLSTMModule/concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0bi_lstm_model/FirstBlockLSTMModule/concat_9/axis?
+bi_lstm_model/FirstBlockLSTMModule/concat_9ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_20:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_21:output:09bi_lstm_model/FirstBlockLSTMModule/concat_9/axis:output:0*
N*
T0*
_output_shapes	
:?2-
+bi_lstm_model/FirstBlockLSTMModule/concat_9?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_13/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_13Reshape4bi_lstm_model/FirstBlockLSTMModule/concat_9:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_13/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_13?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:
2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_22StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_22?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_23StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_23/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_23?
1bi_lstm_model/FirstBlockLSTMModule/concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_10/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_10ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_22:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_23:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_10/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_10?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_14/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_14Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_10:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_14/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_14?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_24StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_24?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stackConst*
_output_shapes
: *
dtype0*
value	B :
2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_25StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_25/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_25?
1bi_lstm_model/FirstBlockLSTMModule/concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_11/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_11ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_24:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_25:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_11/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_11?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_15/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_15Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_11:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_15/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_15?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_26StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_26?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stackConst*
_output_shapes
: *
dtype0*
value	B :	2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_2Const*
_output_shapes
: *
dtype0*
value	B :
2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_27StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_27/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_27?
1bi_lstm_model/FirstBlockLSTMModule/concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_12/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_12ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_26:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_27:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_12/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_12?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_16/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_16Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_12:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_16/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_16?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_28StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_28?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_2Const*
_output_shapes
: *
dtype0*
value	B :	2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_29StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_29/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_29?
1bi_lstm_model/FirstBlockLSTMModule/concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_13/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_13ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_28:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_29:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_13/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_13?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_17/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_17Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_13:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_17/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_17?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_30StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_30?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_31StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_31/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_31?
1bi_lstm_model/FirstBlockLSTMModule/concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_14/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_14ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_30:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_31:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_14/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_14?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_18/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_18Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_14:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_18/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_18?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_32StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_32?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_33StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_33/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_33?
1bi_lstm_model/FirstBlockLSTMModule/concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_15/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_15ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_32:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_33:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_15/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_15?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_19/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_19Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_15:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_19/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_19?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_34StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_34?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_35StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_35/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_35?
1bi_lstm_model/FirstBlockLSTMModule/concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_16/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_16ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_34:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_35:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_16/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_16?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_20/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_20Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_16:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_20/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_20?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_36StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_36?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_37StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_37/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_37?
1bi_lstm_model/FirstBlockLSTMModule/concat_17/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_17/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_17ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_36:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_37:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_17/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_17?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_21/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_21Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_17:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_21/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_21?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_38StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_38?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_39StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_39/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_39?
1bi_lstm_model/FirstBlockLSTMModule/concat_18/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_18/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_18ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_38:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_39:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_18/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_18?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_22/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_22Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_18:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_22/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_22?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_40StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_40?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_41StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_41/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_41?
1bi_lstm_model/FirstBlockLSTMModule/concat_19/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_19/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_19ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_40:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_41:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_19/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_19?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_23/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_23Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_19:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_23/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_23?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_42StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_42?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stackConst*
_output_shapes
: *
dtype0*
value	B :2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_43StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_43/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_43?
1bi_lstm_model/FirstBlockLSTMModule/concat_20/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_20/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_20ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_42:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_43:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_20/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_20?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_24/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_24Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_20:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_24/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_24?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_2?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_44StridedSlice9bi_lstm_model/FirstBlockLSTMModule/strided_slice:output:0Bbi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_44/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_44?
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stackConst*
_output_shapes
: *
dtype0*
value	B : 2;
9bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B : 2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_2?
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3/values_0?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3PackMbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3?
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2=
;bi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_4?
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_45StridedSlice;bi_lstm_model/FirstBlockLSTMModule/strided_slice_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_1:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_3:output:0Dbi_lstm_model/FirstBlockLSTMModule/strided_slice_45/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask25
3bi_lstm_model/FirstBlockLSTMModule/strided_slice_45?
1bi_lstm_model/FirstBlockLSTMModule/concat_21/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_21/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_21ConcatV2<bi_lstm_model/FirstBlockLSTMModule/strided_slice_44:output:0<bi_lstm_model/FirstBlockLSTMModule/strided_slice_45:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_21/axis:output:0*
N*
T0*
_output_shapes	
:?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_21?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_25/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_25Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_21:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_25/shape:output:0*
T0*
_output_shapes
:	?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_25?
1bi_lstm_model/FirstBlockLSTMModule/concat_22/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_22/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_22ConcatV25bi_lstm_model/FirstBlockLSTMModule/Reshape_4:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_5:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_6:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_7:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_8:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_9:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_10:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_11:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_12:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_13:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_14:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_15:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_16:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_17:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_18:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_19:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_20:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_21:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_22:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_23:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_24:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_25:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_22/axis:output:0*
N*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_22?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_26/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_26/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_26Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_22:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_26/shape:output:0*
T0*#
_output_shapes
:?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_26?
1bi_lstm_model/FirstBlockLSTMModule/concat_23/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1bi_lstm_model/FirstBlockLSTMModule/concat_23/axis?
,bi_lstm_model/FirstBlockLSTMModule/concat_23ConcatV26bi_lstm_model/FirstBlockLSTMModule/Reshape_25:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_24:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_23:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_22:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_21:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_20:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_19:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_18:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_17:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_16:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_15:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_14:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_13:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_12:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_11:output:06bi_lstm_model/FirstBlockLSTMModule/Reshape_10:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_9:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_8:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_7:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_6:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_5:output:05bi_lstm_model/FirstBlockLSTMModule/Reshape_4:output:0:bi_lstm_model/FirstBlockLSTMModule/concat_23/axis:output:0*
N*
T0*
_output_shapes
:	?2.
,bi_lstm_model/FirstBlockLSTMModule/concat_23?
3bi_lstm_model/FirstBlockLSTMModule/Reshape_27/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????25
3bi_lstm_model/FirstBlockLSTMModule/Reshape_27/shape?
-bi_lstm_model/FirstBlockLSTMModule/Reshape_27Reshape5bi_lstm_model/FirstBlockLSTMModule/concat_23:output:0<bi_lstm_model/FirstBlockLSTMModule/Reshape_27/shape:output:0*
T0*#
_output_shapes
:?2/
-bi_lstm_model/FirstBlockLSTMModule/Reshape_27?
)bi_lstm_model/NextBlockLSTM/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2+
)bi_lstm_model/NextBlockLSTM/Reshape/shape?
#bi_lstm_model/NextBlockLSTM/ReshapeReshape6bi_lstm_model/FirstBlockLSTMModule/Reshape_26:output:02bi_lstm_model/NextBlockLSTM/Reshape/shape:output:0*
T0*#
_output_shapes
:?2%
#bi_lstm_model/NextBlockLSTM/Reshape?
+bi_lstm_model/NextBlockLSTM/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2-
+bi_lstm_model/NextBlockLSTM/Reshape_1/shape?
%bi_lstm_model/NextBlockLSTM/Reshape_1Reshape6bi_lstm_model/FirstBlockLSTMModule/Reshape_27:output:04bi_lstm_model/NextBlockLSTM/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2'
%bi_lstm_model/NextBlockLSTM/Reshape_1?
1bi_lstm_model/NextBlockLSTM/BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R23
1bi_lstm_model/NextBlockLSTM/BlockLSTM/seq_len_max?
4bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOpReadVariableOp=bi_lstm_model_nextblocklstm_blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp?
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_1ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype028
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_1?
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_2ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype028
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_2?
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_3ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype028
6bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_3?
%bi_lstm_model/NextBlockLSTM/BlockLSTM	BlockLSTM:bi_lstm_model/NextBlockLSTM/BlockLSTM/seq_len_max:output:0,bi_lstm_model/NextBlockLSTM/Reshape:output:0-bi_lstm_model_nextblocklstm_blocklstm_cs_prev,bi_lstm_model_nextblocklstm_blocklstm_h_prev<bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp:value:0>bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_1:value:0>bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_2:value:0>bi_lstm_model/NextBlockLSTM/BlockLSTM/ReadVariableOp_3:value:0'bi_lstm_model_nextblocklstm_blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2'
%bi_lstm_model/NextBlockLSTM/BlockLSTM?
3bi_lstm_model/NextBlockLSTM/BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3bi_lstm_model/NextBlockLSTM/BlockLSTM_1/seq_len_max?
6bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOpReadVariableOp=bi_lstm_model_nextblocklstm_blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp?
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_1ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02:
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_1?
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_2ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02:
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_2?
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_3ReadVariableOp?bi_lstm_model_nextblocklstm_blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02:
8bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_3?
'bi_lstm_model/NextBlockLSTM/BlockLSTM_1	BlockLSTM<bi_lstm_model/NextBlockLSTM/BlockLSTM_1/seq_len_max:output:0.bi_lstm_model/NextBlockLSTM/Reshape_1:output:0-bi_lstm_model_nextblocklstm_blocklstm_cs_prev,bi_lstm_model_nextblocklstm_blocklstm_h_prev>bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp:value:0@bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_1:value:0@bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_2:value:0@bi_lstm_model/NextBlockLSTM/BlockLSTM_1/ReadVariableOp_3:value:0'bi_lstm_model_nextblocklstm_blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2)
'bi_lstm_model/NextBlockLSTM/BlockLSTM_1?
+bi_lstm_model/NextBlockLSTM/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2-
+bi_lstm_model/NextBlockLSTM/Reshape_2/shape?
%bi_lstm_model/NextBlockLSTM/Reshape_2Reshape)bi_lstm_model/NextBlockLSTM/BlockLSTM:h:04bi_lstm_model/NextBlockLSTM/Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2'
%bi_lstm_model/NextBlockLSTM/Reshape_2?
+bi_lstm_model/NextBlockLSTM/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2-
+bi_lstm_model/NextBlockLSTM/Reshape_3/shape?
%bi_lstm_model/NextBlockLSTM/Reshape_3Reshape+bi_lstm_model/NextBlockLSTM/BlockLSTM_1:h:04bi_lstm_model/NextBlockLSTM/Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2'
%bi_lstm_model/NextBlockLSTM/Reshape_3?
bi_lstm_model/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ~   2
bi_lstm_model/Reshape/shape?
bi_lstm_model/ReshapeReshape.bi_lstm_model/NextBlockLSTM/Reshape_2:output:0$bi_lstm_model/Reshape/shape:output:0*
T0*
_output_shapes

:~2
bi_lstm_model/Reshape?
bi_lstm_model/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ~   2
bi_lstm_model/Reshape_1/shape?
bi_lstm_model/Reshape_1Reshape.bi_lstm_model/NextBlockLSTM/Reshape_3:output:0&bi_lstm_model/Reshape_1/shape:output:0*
T0*
_output_shapes

:~2
bi_lstm_model/Reshape_1?
!bi_lstm_model/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!bi_lstm_model/strided_slice/stack?
#bi_lstm_model/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice/stack_1?
#bi_lstm_model/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice/stack_2?
bi_lstm_model/strided_sliceStridedSlicebi_lstm_model/Reshape:output:0*bi_lstm_model/strided_slice/stack:output:0,bi_lstm_model/strided_slice/stack_1:output:0,bi_lstm_model/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice?
#bi_lstm_model/strided_slice_1/stackConst*
_output_shapes
: *
dtype0*
value	B :2%
#bi_lstm_model/strided_slice_1/stack?
.bi_lstm_model/strided_slice_1/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_1/stack_1/values_0?
%bi_lstm_model/strided_slice_1/stack_1Pack7bi_lstm_model/strided_slice_1/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_1/stack_1?
%bi_lstm_model/strided_slice_1/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2'
%bi_lstm_model/strided_slice_1/stack_2?
.bi_lstm_model/strided_slice_1/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_1/stack_3/values_0?
%bi_lstm_model/strided_slice_1/stack_3Pack7bi_lstm_model/strided_slice_1/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_1/stack_3?
%bi_lstm_model/strided_slice_1/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_1/stack_4?
bi_lstm_model/strided_slice_1StridedSlice bi_lstm_model/Reshape_1:output:0.bi_lstm_model/strided_slice_1/stack_1:output:0.bi_lstm_model/strided_slice_1/stack_3:output:0.bi_lstm_model/strided_slice_1/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_1?
#bi_lstm_model/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice_2/stack?
%bi_lstm_model/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_2/stack_1?
%bi_lstm_model/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_2/stack_2?
bi_lstm_model/strided_slice_2StridedSlicebi_lstm_model/Reshape:output:0,bi_lstm_model/strided_slice_2/stack:output:0.bi_lstm_model/strided_slice_2/stack_1:output:0.bi_lstm_model/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_2?
#bi_lstm_model/strided_slice_3/stackConst*
_output_shapes
: *
dtype0*
value	B :2%
#bi_lstm_model/strided_slice_3/stack?
.bi_lstm_model/strided_slice_3/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_3/stack_1/values_0?
%bi_lstm_model/strided_slice_3/stack_1Pack7bi_lstm_model/strided_slice_3/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_3/stack_1?
%bi_lstm_model/strided_slice_3/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2'
%bi_lstm_model/strided_slice_3/stack_2?
.bi_lstm_model/strided_slice_3/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_3/stack_3/values_0?
%bi_lstm_model/strided_slice_3/stack_3Pack7bi_lstm_model/strided_slice_3/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_3/stack_3?
%bi_lstm_model/strided_slice_3/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_3/stack_4?
bi_lstm_model/strided_slice_3StridedSlice bi_lstm_model/Reshape_1:output:0.bi_lstm_model/strided_slice_3/stack_1:output:0.bi_lstm_model/strided_slice_3/stack_3:output:0.bi_lstm_model/strided_slice_3/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_3?
#bi_lstm_model/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice_4/stack?
%bi_lstm_model/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_4/stack_1?
%bi_lstm_model/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_4/stack_2?
bi_lstm_model/strided_slice_4StridedSlicebi_lstm_model/Reshape:output:0,bi_lstm_model/strided_slice_4/stack:output:0.bi_lstm_model/strided_slice_4/stack_1:output:0.bi_lstm_model/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_4?
#bi_lstm_model/strided_slice_5/stackConst*
_output_shapes
: *
dtype0*
value	B :2%
#bi_lstm_model/strided_slice_5/stack?
.bi_lstm_model/strided_slice_5/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_5/stack_1/values_0?
%bi_lstm_model/strided_slice_5/stack_1Pack7bi_lstm_model/strided_slice_5/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_5/stack_1?
%bi_lstm_model/strided_slice_5/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2'
%bi_lstm_model/strided_slice_5/stack_2?
.bi_lstm_model/strided_slice_5/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_5/stack_3/values_0?
%bi_lstm_model/strided_slice_5/stack_3Pack7bi_lstm_model/strided_slice_5/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_5/stack_3?
%bi_lstm_model/strided_slice_5/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_5/stack_4?
bi_lstm_model/strided_slice_5StridedSlice bi_lstm_model/Reshape_1:output:0.bi_lstm_model/strided_slice_5/stack_1:output:0.bi_lstm_model/strided_slice_5/stack_3:output:0.bi_lstm_model/strided_slice_5/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_5?
#bi_lstm_model/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice_6/stack?
%bi_lstm_model/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_6/stack_1?
%bi_lstm_model/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_6/stack_2?
bi_lstm_model/strided_slice_6StridedSlicebi_lstm_model/Reshape:output:0,bi_lstm_model/strided_slice_6/stack:output:0.bi_lstm_model/strided_slice_6/stack_1:output:0.bi_lstm_model/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_6?
#bi_lstm_model/strided_slice_7/stackConst*
_output_shapes
: *
dtype0*
value	B :2%
#bi_lstm_model/strided_slice_7/stack?
.bi_lstm_model/strided_slice_7/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_7/stack_1/values_0?
%bi_lstm_model/strided_slice_7/stack_1Pack7bi_lstm_model/strided_slice_7/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_7/stack_1?
%bi_lstm_model/strided_slice_7/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2'
%bi_lstm_model/strided_slice_7/stack_2?
.bi_lstm_model/strided_slice_7/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_7/stack_3/values_0?
%bi_lstm_model/strided_slice_7/stack_3Pack7bi_lstm_model/strided_slice_7/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_7/stack_3?
%bi_lstm_model/strided_slice_7/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_7/stack_4?
bi_lstm_model/strided_slice_7StridedSlice bi_lstm_model/Reshape_1:output:0.bi_lstm_model/strided_slice_7/stack_1:output:0.bi_lstm_model/strided_slice_7/stack_3:output:0.bi_lstm_model/strided_slice_7/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_7?
#bi_lstm_model/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#bi_lstm_model/strided_slice_8/stack?
%bi_lstm_model/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_8/stack_1?
%bi_lstm_model/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_8/stack_2?
bi_lstm_model/strided_slice_8StridedSlicebi_lstm_model/Reshape:output:0,bi_lstm_model/strided_slice_8/stack:output:0.bi_lstm_model/strided_slice_8/stack_1:output:0.bi_lstm_model/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_8?
#bi_lstm_model/strided_slice_9/stackConst*
_output_shapes
: *
dtype0*
value	B :2%
#bi_lstm_model/strided_slice_9/stack?
.bi_lstm_model/strided_slice_9/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_9/stack_1/values_0?
%bi_lstm_model/strided_slice_9/stack_1Pack7bi_lstm_model/strided_slice_9/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_9/stack_1?
%bi_lstm_model/strided_slice_9/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2'
%bi_lstm_model/strided_slice_9/stack_2?
.bi_lstm_model/strided_slice_9/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :20
.bi_lstm_model/strided_slice_9/stack_3/values_0?
%bi_lstm_model/strided_slice_9/stack_3Pack7bi_lstm_model/strided_slice_9/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2'
%bi_lstm_model/strided_slice_9/stack_3?
%bi_lstm_model/strided_slice_9/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2'
%bi_lstm_model/strided_slice_9/stack_4?
bi_lstm_model/strided_slice_9StridedSlice bi_lstm_model/Reshape_1:output:0.bi_lstm_model/strided_slice_9/stack_1:output:0.bi_lstm_model/strided_slice_9/stack_3:output:0.bi_lstm_model/strided_slice_9/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
bi_lstm_model/strided_slice_9?
$bi_lstm_model/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_10/stack?
&bi_lstm_model/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_10/stack_1?
&bi_lstm_model/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_10/stack_2?
bi_lstm_model/strided_slice_10StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_10/stack:output:0/bi_lstm_model/strided_slice_10/stack_1:output:0/bi_lstm_model/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_10?
$bi_lstm_model/strided_slice_11/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_11/stack?
/bi_lstm_model/strided_slice_11/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_11/stack_1/values_0?
&bi_lstm_model/strided_slice_11/stack_1Pack8bi_lstm_model/strided_slice_11/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_11/stack_1?
&bi_lstm_model/strided_slice_11/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_11/stack_2?
/bi_lstm_model/strided_slice_11/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_11/stack_3/values_0?
&bi_lstm_model/strided_slice_11/stack_3Pack8bi_lstm_model/strided_slice_11/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_11/stack_3?
&bi_lstm_model/strided_slice_11/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_11/stack_4?
bi_lstm_model/strided_slice_11StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_11/stack_1:output:0/bi_lstm_model/strided_slice_11/stack_3:output:0/bi_lstm_model/strided_slice_11/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_11?
$bi_lstm_model/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_12/stack?
&bi_lstm_model/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_12/stack_1?
&bi_lstm_model/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_12/stack_2?
bi_lstm_model/strided_slice_12StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_12/stack:output:0/bi_lstm_model/strided_slice_12/stack_1:output:0/bi_lstm_model/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_12?
$bi_lstm_model/strided_slice_13/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_13/stack?
/bi_lstm_model/strided_slice_13/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_13/stack_1/values_0?
&bi_lstm_model/strided_slice_13/stack_1Pack8bi_lstm_model/strided_slice_13/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_13/stack_1?
&bi_lstm_model/strided_slice_13/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_13/stack_2?
/bi_lstm_model/strided_slice_13/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_13/stack_3/values_0?
&bi_lstm_model/strided_slice_13/stack_3Pack8bi_lstm_model/strided_slice_13/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_13/stack_3?
&bi_lstm_model/strided_slice_13/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_13/stack_4?
bi_lstm_model/strided_slice_13StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_13/stack_1:output:0/bi_lstm_model/strided_slice_13/stack_3:output:0/bi_lstm_model/strided_slice_13/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_13?
$bi_lstm_model/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_14/stack?
&bi_lstm_model/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_14/stack_1?
&bi_lstm_model/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_14/stack_2?
bi_lstm_model/strided_slice_14StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_14/stack:output:0/bi_lstm_model/strided_slice_14/stack_1:output:0/bi_lstm_model/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_14?
$bi_lstm_model/strided_slice_15/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_15/stack?
/bi_lstm_model/strided_slice_15/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_15/stack_1/values_0?
&bi_lstm_model/strided_slice_15/stack_1Pack8bi_lstm_model/strided_slice_15/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_15/stack_1?
&bi_lstm_model/strided_slice_15/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_15/stack_2?
/bi_lstm_model/strided_slice_15/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_15/stack_3/values_0?
&bi_lstm_model/strided_slice_15/stack_3Pack8bi_lstm_model/strided_slice_15/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_15/stack_3?
&bi_lstm_model/strided_slice_15/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_15/stack_4?
bi_lstm_model/strided_slice_15StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_15/stack_1:output:0/bi_lstm_model/strided_slice_15/stack_3:output:0/bi_lstm_model/strided_slice_15/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_15?
$bi_lstm_model/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_16/stack?
&bi_lstm_model/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2(
&bi_lstm_model/strided_slice_16/stack_1?
&bi_lstm_model/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_16/stack_2?
bi_lstm_model/strided_slice_16StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_16/stack:output:0/bi_lstm_model/strided_slice_16/stack_1:output:0/bi_lstm_model/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_16?
$bi_lstm_model/strided_slice_17/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_17/stack?
/bi_lstm_model/strided_slice_17/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_17/stack_1/values_0?
&bi_lstm_model/strided_slice_17/stack_1Pack8bi_lstm_model/strided_slice_17/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_17/stack_1?
&bi_lstm_model/strided_slice_17/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_17/stack_2?
/bi_lstm_model/strided_slice_17/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_17/stack_3/values_0?
&bi_lstm_model/strided_slice_17/stack_3Pack8bi_lstm_model/strided_slice_17/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_17/stack_3?
&bi_lstm_model/strided_slice_17/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_17/stack_4?
bi_lstm_model/strided_slice_17StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_17/stack_1:output:0/bi_lstm_model/strided_slice_17/stack_3:output:0/bi_lstm_model/strided_slice_17/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_17?
$bi_lstm_model/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:	2&
$bi_lstm_model/strided_slice_18/stack?
&bi_lstm_model/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2(
&bi_lstm_model/strided_slice_18/stack_1?
&bi_lstm_model/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_18/stack_2?
bi_lstm_model/strided_slice_18StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_18/stack:output:0/bi_lstm_model/strided_slice_18/stack_1:output:0/bi_lstm_model/strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_18?
$bi_lstm_model/strided_slice_19/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_19/stack?
/bi_lstm_model/strided_slice_19/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_19/stack_1/values_0?
&bi_lstm_model/strided_slice_19/stack_1Pack8bi_lstm_model/strided_slice_19/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_19/stack_1?
&bi_lstm_model/strided_slice_19/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_19/stack_2?
/bi_lstm_model/strided_slice_19/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_19/stack_3/values_0?
&bi_lstm_model/strided_slice_19/stack_3Pack8bi_lstm_model/strided_slice_19/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_19/stack_3?
&bi_lstm_model/strided_slice_19/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_19/stack_4?
bi_lstm_model/strided_slice_19StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_19/stack_1:output:0/bi_lstm_model/strided_slice_19/stack_3:output:0/bi_lstm_model/strided_slice_19/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_19?
$bi_lstm_model/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:
2&
$bi_lstm_model/strided_slice_20/stack?
&bi_lstm_model/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_20/stack_1?
&bi_lstm_model/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_20/stack_2?
bi_lstm_model/strided_slice_20StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_20/stack:output:0/bi_lstm_model/strided_slice_20/stack_1:output:0/bi_lstm_model/strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_20?
$bi_lstm_model/strided_slice_21/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_21/stack?
/bi_lstm_model/strided_slice_21/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_21/stack_1/values_0?
&bi_lstm_model/strided_slice_21/stack_1Pack8bi_lstm_model/strided_slice_21/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_21/stack_1?
&bi_lstm_model/strided_slice_21/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_21/stack_2?
/bi_lstm_model/strided_slice_21/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_21/stack_3/values_0?
&bi_lstm_model/strided_slice_21/stack_3Pack8bi_lstm_model/strided_slice_21/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_21/stack_3?
&bi_lstm_model/strided_slice_21/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_21/stack_4?
bi_lstm_model/strided_slice_21StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_21/stack_1:output:0/bi_lstm_model/strided_slice_21/stack_3:output:0/bi_lstm_model/strided_slice_21/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_21?
$bi_lstm_model/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_22/stack?
&bi_lstm_model/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_22/stack_1?
&bi_lstm_model/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_22/stack_2?
bi_lstm_model/strided_slice_22StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_22/stack:output:0/bi_lstm_model/strided_slice_22/stack_1:output:0/bi_lstm_model/strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_22?
$bi_lstm_model/strided_slice_23/stackConst*
_output_shapes
: *
dtype0*
value	B :
2&
$bi_lstm_model/strided_slice_23/stack?
/bi_lstm_model/strided_slice_23/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :
21
/bi_lstm_model/strided_slice_23/stack_1/values_0?
&bi_lstm_model/strided_slice_23/stack_1Pack8bi_lstm_model/strided_slice_23/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_23/stack_1?
&bi_lstm_model/strided_slice_23/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_23/stack_2?
/bi_lstm_model/strided_slice_23/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_23/stack_3/values_0?
&bi_lstm_model/strided_slice_23/stack_3Pack8bi_lstm_model/strided_slice_23/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_23/stack_3?
&bi_lstm_model/strided_slice_23/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_23/stack_4?
bi_lstm_model/strided_slice_23StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_23/stack_1:output:0/bi_lstm_model/strided_slice_23/stack_3:output:0/bi_lstm_model/strided_slice_23/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_23?
$bi_lstm_model/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_24/stack?
&bi_lstm_model/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_24/stack_1?
&bi_lstm_model/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_24/stack_2?
bi_lstm_model/strided_slice_24StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_24/stack:output:0/bi_lstm_model/strided_slice_24/stack_1:output:0/bi_lstm_model/strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_24?
$bi_lstm_model/strided_slice_25/stackConst*
_output_shapes
: *
dtype0*
value	B :	2&
$bi_lstm_model/strided_slice_25/stack?
/bi_lstm_model/strided_slice_25/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :	21
/bi_lstm_model/strided_slice_25/stack_1/values_0?
&bi_lstm_model/strided_slice_25/stack_1Pack8bi_lstm_model/strided_slice_25/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_25/stack_1?
&bi_lstm_model/strided_slice_25/stack_2Const*
_output_shapes
: *
dtype0*
value	B :
2(
&bi_lstm_model/strided_slice_25/stack_2?
/bi_lstm_model/strided_slice_25/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :
21
/bi_lstm_model/strided_slice_25/stack_3/values_0?
&bi_lstm_model/strided_slice_25/stack_3Pack8bi_lstm_model/strided_slice_25/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_25/stack_3?
&bi_lstm_model/strided_slice_25/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_25/stack_4?
bi_lstm_model/strided_slice_25StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_25/stack_1:output:0/bi_lstm_model/strided_slice_25/stack_3:output:0/bi_lstm_model/strided_slice_25/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_25?
$bi_lstm_model/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_26/stack?
&bi_lstm_model/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_26/stack_1?
&bi_lstm_model/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_26/stack_2?
bi_lstm_model/strided_slice_26StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_26/stack:output:0/bi_lstm_model/strided_slice_26/stack_1:output:0/bi_lstm_model/strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_26?
$bi_lstm_model/strided_slice_27/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_27/stack?
/bi_lstm_model/strided_slice_27/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_27/stack_1/values_0?
&bi_lstm_model/strided_slice_27/stack_1Pack8bi_lstm_model/strided_slice_27/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_27/stack_1?
&bi_lstm_model/strided_slice_27/stack_2Const*
_output_shapes
: *
dtype0*
value	B :	2(
&bi_lstm_model/strided_slice_27/stack_2?
/bi_lstm_model/strided_slice_27/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :	21
/bi_lstm_model/strided_slice_27/stack_3/values_0?
&bi_lstm_model/strided_slice_27/stack_3Pack8bi_lstm_model/strided_slice_27/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_27/stack_3?
&bi_lstm_model/strided_slice_27/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_27/stack_4?
bi_lstm_model/strided_slice_27StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_27/stack_1:output:0/bi_lstm_model/strided_slice_27/stack_3:output:0/bi_lstm_model/strided_slice_27/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_27?
$bi_lstm_model/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_28/stack?
&bi_lstm_model/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_28/stack_1?
&bi_lstm_model/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_28/stack_2?
bi_lstm_model/strided_slice_28StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_28/stack:output:0/bi_lstm_model/strided_slice_28/stack_1:output:0/bi_lstm_model/strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_28?
$bi_lstm_model/strided_slice_29/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_29/stack?
/bi_lstm_model/strided_slice_29/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_29/stack_1/values_0?
&bi_lstm_model/strided_slice_29/stack_1Pack8bi_lstm_model/strided_slice_29/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_29/stack_1?
&bi_lstm_model/strided_slice_29/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_29/stack_2?
/bi_lstm_model/strided_slice_29/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_29/stack_3/values_0?
&bi_lstm_model/strided_slice_29/stack_3Pack8bi_lstm_model/strided_slice_29/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_29/stack_3?
&bi_lstm_model/strided_slice_29/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_29/stack_4?
bi_lstm_model/strided_slice_29StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_29/stack_1:output:0/bi_lstm_model/strided_slice_29/stack_3:output:0/bi_lstm_model/strided_slice_29/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_29?
$bi_lstm_model/strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_30/stack?
&bi_lstm_model/strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_30/stack_1?
&bi_lstm_model/strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_30/stack_2?
bi_lstm_model/strided_slice_30StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_30/stack:output:0/bi_lstm_model/strided_slice_30/stack_1:output:0/bi_lstm_model/strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_30?
$bi_lstm_model/strided_slice_31/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_31/stack?
/bi_lstm_model/strided_slice_31/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_31/stack_1/values_0?
&bi_lstm_model/strided_slice_31/stack_1Pack8bi_lstm_model/strided_slice_31/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_31/stack_1?
&bi_lstm_model/strided_slice_31/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_31/stack_2?
/bi_lstm_model/strided_slice_31/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_31/stack_3/values_0?
&bi_lstm_model/strided_slice_31/stack_3Pack8bi_lstm_model/strided_slice_31/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_31/stack_3?
&bi_lstm_model/strided_slice_31/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_31/stack_4?
bi_lstm_model/strided_slice_31StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_31/stack_1:output:0/bi_lstm_model/strided_slice_31/stack_3:output:0/bi_lstm_model/strided_slice_31/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_31?
$bi_lstm_model/strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_32/stack?
&bi_lstm_model/strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_32/stack_1?
&bi_lstm_model/strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_32/stack_2?
bi_lstm_model/strided_slice_32StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_32/stack:output:0/bi_lstm_model/strided_slice_32/stack_1:output:0/bi_lstm_model/strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_32?
$bi_lstm_model/strided_slice_33/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_33/stack?
/bi_lstm_model/strided_slice_33/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_33/stack_1/values_0?
&bi_lstm_model/strided_slice_33/stack_1Pack8bi_lstm_model/strided_slice_33/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_33/stack_1?
&bi_lstm_model/strided_slice_33/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_33/stack_2?
/bi_lstm_model/strided_slice_33/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_33/stack_3/values_0?
&bi_lstm_model/strided_slice_33/stack_3Pack8bi_lstm_model/strided_slice_33/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_33/stack_3?
&bi_lstm_model/strided_slice_33/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_33/stack_4?
bi_lstm_model/strided_slice_33StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_33/stack_1:output:0/bi_lstm_model/strided_slice_33/stack_3:output:0/bi_lstm_model/strided_slice_33/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_33?
$bi_lstm_model/strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_34/stack?
&bi_lstm_model/strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_34/stack_1?
&bi_lstm_model/strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_34/stack_2?
bi_lstm_model/strided_slice_34StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_34/stack:output:0/bi_lstm_model/strided_slice_34/stack_1:output:0/bi_lstm_model/strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_34?
$bi_lstm_model/strided_slice_35/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_35/stack?
/bi_lstm_model/strided_slice_35/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_35/stack_1/values_0?
&bi_lstm_model/strided_slice_35/stack_1Pack8bi_lstm_model/strided_slice_35/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_35/stack_1?
&bi_lstm_model/strided_slice_35/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_35/stack_2?
/bi_lstm_model/strided_slice_35/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_35/stack_3/values_0?
&bi_lstm_model/strided_slice_35/stack_3Pack8bi_lstm_model/strided_slice_35/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_35/stack_3?
&bi_lstm_model/strided_slice_35/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_35/stack_4?
bi_lstm_model/strided_slice_35StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_35/stack_1:output:0/bi_lstm_model/strided_slice_35/stack_3:output:0/bi_lstm_model/strided_slice_35/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_35?
$bi_lstm_model/strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_36/stack?
&bi_lstm_model/strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_36/stack_1?
&bi_lstm_model/strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_36/stack_2?
bi_lstm_model/strided_slice_36StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_36/stack:output:0/bi_lstm_model/strided_slice_36/stack_1:output:0/bi_lstm_model/strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_36?
$bi_lstm_model/strided_slice_37/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_37/stack?
/bi_lstm_model/strided_slice_37/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_37/stack_1/values_0?
&bi_lstm_model/strided_slice_37/stack_1Pack8bi_lstm_model/strided_slice_37/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_37/stack_1?
&bi_lstm_model/strided_slice_37/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_37/stack_2?
/bi_lstm_model/strided_slice_37/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_37/stack_3/values_0?
&bi_lstm_model/strided_slice_37/stack_3Pack8bi_lstm_model/strided_slice_37/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_37/stack_3?
&bi_lstm_model/strided_slice_37/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_37/stack_4?
bi_lstm_model/strided_slice_37StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_37/stack_1:output:0/bi_lstm_model/strided_slice_37/stack_3:output:0/bi_lstm_model/strided_slice_37/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_37?
$bi_lstm_model/strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_38/stack?
&bi_lstm_model/strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_38/stack_1?
&bi_lstm_model/strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_38/stack_2?
bi_lstm_model/strided_slice_38StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_38/stack:output:0/bi_lstm_model/strided_slice_38/stack_1:output:0/bi_lstm_model/strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_38?
$bi_lstm_model/strided_slice_39/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_39/stack?
/bi_lstm_model/strided_slice_39/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_39/stack_1/values_0?
&bi_lstm_model/strided_slice_39/stack_1Pack8bi_lstm_model/strided_slice_39/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_39/stack_1?
&bi_lstm_model/strided_slice_39/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_39/stack_2?
/bi_lstm_model/strided_slice_39/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_39/stack_3/values_0?
&bi_lstm_model/strided_slice_39/stack_3Pack8bi_lstm_model/strided_slice_39/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_39/stack_3?
&bi_lstm_model/strided_slice_39/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_39/stack_4?
bi_lstm_model/strided_slice_39StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_39/stack_1:output:0/bi_lstm_model/strided_slice_39/stack_3:output:0/bi_lstm_model/strided_slice_39/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_39?
$bi_lstm_model/strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_40/stack?
&bi_lstm_model/strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_40/stack_1?
&bi_lstm_model/strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_40/stack_2?
bi_lstm_model/strided_slice_40StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_40/stack:output:0/bi_lstm_model/strided_slice_40/stack_1:output:0/bi_lstm_model/strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_40?
$bi_lstm_model/strided_slice_41/stackConst*
_output_shapes
: *
dtype0*
value	B :2&
$bi_lstm_model/strided_slice_41/stack?
/bi_lstm_model/strided_slice_41/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_41/stack_1/values_0?
&bi_lstm_model/strided_slice_41/stack_1Pack8bi_lstm_model/strided_slice_41/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_41/stack_1?
&bi_lstm_model/strided_slice_41/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_41/stack_2?
/bi_lstm_model/strided_slice_41/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_41/stack_3/values_0?
&bi_lstm_model/strided_slice_41/stack_3Pack8bi_lstm_model/strided_slice_41/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_41/stack_3?
&bi_lstm_model/strided_slice_41/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_41/stack_4?
bi_lstm_model/strided_slice_41StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_41/stack_1:output:0/bi_lstm_model/strided_slice_41/stack_3:output:0/bi_lstm_model/strided_slice_41/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_41?
$bi_lstm_model/strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$bi_lstm_model/strided_slice_42/stack?
&bi_lstm_model/strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_42/stack_1?
&bi_lstm_model/strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_42/stack_2?
bi_lstm_model/strided_slice_42StridedSlicebi_lstm_model/Reshape:output:0-bi_lstm_model/strided_slice_42/stack:output:0/bi_lstm_model/strided_slice_42/stack_1:output:0/bi_lstm_model/strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_42?
$bi_lstm_model/strided_slice_43/stackConst*
_output_shapes
: *
dtype0*
value	B : 2&
$bi_lstm_model/strided_slice_43/stack?
/bi_lstm_model/strided_slice_43/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B : 21
/bi_lstm_model/strided_slice_43/stack_1/values_0?
&bi_lstm_model/strided_slice_43/stack_1Pack8bi_lstm_model/strided_slice_43/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_43/stack_1?
&bi_lstm_model/strided_slice_43/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2(
&bi_lstm_model/strided_slice_43/stack_2?
/bi_lstm_model/strided_slice_43/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :21
/bi_lstm_model/strided_slice_43/stack_3/values_0?
&bi_lstm_model/strided_slice_43/stack_3Pack8bi_lstm_model/strided_slice_43/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2(
&bi_lstm_model/strided_slice_43/stack_3?
&bi_lstm_model/strided_slice_43/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2(
&bi_lstm_model/strided_slice_43/stack_4?
bi_lstm_model/strided_slice_43StridedSlice bi_lstm_model/Reshape_1:output:0/bi_lstm_model/strided_slice_43/stack_1:output:0/bi_lstm_model/strided_slice_43/stack_3:output:0/bi_lstm_model/strided_slice_43/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2 
bi_lstm_model/strided_slice_43n
IdentityIdentity'bi_lstm_model/strided_slice_18:output:0*
T0*
_output_shapes
:~2

Identityr

Identity_1Identity'bi_lstm_model/strided_slice_19:output:0*
T0*
_output_shapes
:~2

Identity_1r

Identity_2Identity'bi_lstm_model/strided_slice_20:output:0*
T0*
_output_shapes
:~2

Identity_2r

Identity_3Identity'bi_lstm_model/strided_slice_21:output:0*
T0*
_output_shapes
:~2

Identity_3r

Identity_4Identity'bi_lstm_model/strided_slice_22:output:0*
T0*
_output_shapes
:~2

Identity_4r

Identity_5Identity'bi_lstm_model/strided_slice_23:output:0*
T0*
_output_shapes
:~2

Identity_5r

Identity_6Identity'bi_lstm_model/strided_slice_24:output:0*
T0*
_output_shapes
:~2

Identity_6r

Identity_7Identity'bi_lstm_model/strided_slice_25:output:0*
T0*
_output_shapes
:~2

Identity_7r

Identity_8Identity'bi_lstm_model/strided_slice_26:output:0*
T0*
_output_shapes
:~2

Identity_8r

Identity_9Identity'bi_lstm_model/strided_slice_27:output:0*
T0*
_output_shapes
:~2

Identity_9t
Identity_10Identity'bi_lstm_model/strided_slice_28:output:0*
T0*
_output_shapes
:~2
Identity_10t
Identity_11Identity'bi_lstm_model/strided_slice_29:output:0*
T0*
_output_shapes
:~2
Identity_11t
Identity_12Identity'bi_lstm_model/strided_slice_30:output:0*
T0*
_output_shapes
:~2
Identity_12t
Identity_13Identity'bi_lstm_model/strided_slice_31:output:0*
T0*
_output_shapes
:~2
Identity_13t
Identity_14Identity'bi_lstm_model/strided_slice_32:output:0*
T0*
_output_shapes
:~2
Identity_14t
Identity_15Identity'bi_lstm_model/strided_slice_33:output:0*
T0*
_output_shapes
:~2
Identity_15t
Identity_16Identity'bi_lstm_model/strided_slice_34:output:0*
T0*
_output_shapes
:~2
Identity_16t
Identity_17Identity'bi_lstm_model/strided_slice_35:output:0*
T0*
_output_shapes
:~2
Identity_17t
Identity_18Identity'bi_lstm_model/strided_slice_36:output:0*
T0*
_output_shapes
:~2
Identity_18t
Identity_19Identity'bi_lstm_model/strided_slice_37:output:0*
T0*
_output_shapes
:~2
Identity_19q
Identity_20Identity$bi_lstm_model/strided_slice:output:0*
T0*
_output_shapes
:~2
Identity_20s
Identity_21Identity&bi_lstm_model/strided_slice_1:output:0*
T0*
_output_shapes
:~2
Identity_21t
Identity_22Identity'bi_lstm_model/strided_slice_38:output:0*
T0*
_output_shapes
:~2
Identity_22t
Identity_23Identity'bi_lstm_model/strided_slice_39:output:0*
T0*
_output_shapes
:~2
Identity_23t
Identity_24Identity'bi_lstm_model/strided_slice_40:output:0*
T0*
_output_shapes
:~2
Identity_24t
Identity_25Identity'bi_lstm_model/strided_slice_41:output:0*
T0*
_output_shapes
:~2
Identity_25t
Identity_26Identity'bi_lstm_model/strided_slice_42:output:0*
T0*
_output_shapes
:~2
Identity_26t
Identity_27Identity'bi_lstm_model/strided_slice_43:output:0*
T0*
_output_shapes
:~2
Identity_27s
Identity_28Identity&bi_lstm_model/strided_slice_2:output:0*
T0*
_output_shapes
:~2
Identity_28s
Identity_29Identity&bi_lstm_model/strided_slice_3:output:0*
T0*
_output_shapes
:~2
Identity_29s
Identity_30Identity&bi_lstm_model/strided_slice_4:output:0*
T0*
_output_shapes
:~2
Identity_30s
Identity_31Identity&bi_lstm_model/strided_slice_5:output:0*
T0*
_output_shapes
:~2
Identity_31s
Identity_32Identity&bi_lstm_model/strided_slice_6:output:0*
T0*
_output_shapes
:~2
Identity_32s
Identity_33Identity&bi_lstm_model/strided_slice_7:output:0*
T0*
_output_shapes
:~2
Identity_33s
Identity_34Identity&bi_lstm_model/strided_slice_8:output:0*
T0*
_output_shapes
:~2
Identity_34s
Identity_35Identity&bi_lstm_model/strided_slice_9:output:0*
T0*
_output_shapes
:~2
Identity_35t
Identity_36Identity'bi_lstm_model/strided_slice_10:output:0*
T0*
_output_shapes
:~2
Identity_36t
Identity_37Identity'bi_lstm_model/strided_slice_11:output:0*
T0*
_output_shapes
:~2
Identity_37t
Identity_38Identity'bi_lstm_model/strided_slice_12:output:0*
T0*
_output_shapes
:~2
Identity_38t
Identity_39Identity'bi_lstm_model/strided_slice_13:output:0*
T0*
_output_shapes
:~2
Identity_39t
Identity_40Identity'bi_lstm_model/strided_slice_14:output:0*
T0*
_output_shapes
:~2
Identity_40t
Identity_41Identity'bi_lstm_model/strided_slice_15:output:0*
T0*
_output_shapes
:~2
Identity_41t
Identity_42Identity'bi_lstm_model/strided_slice_16:output:0*
T0*
_output_shapes
:~2
Identity_42t
Identity_43Identity'bi_lstm_model/strided_slice_17:output:0*
T0*
_output_shapes
:~2
Identity_43"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_4Identity_4:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????d:?????????d:~:~:::::?:~:~:::::?:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????d
!
_user_specified_name	input_2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
?
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_122101399

inputs
inputs_1
blocklstm_cs_prev
blocklstm_h_prev%
!blocklstm_readvariableop_resource'
#blocklstm_readvariableop_1_resource'
#blocklstm_readvariableop_2_resource'
#blocklstm_readvariableop_3_resource
blocklstm_b
identity

identity_1?s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2
Reshape/shapek
ReshapeReshapeinputsReshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2
Reshape_1/shapes
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1p
BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM/seq_len_max?
BlockLSTM/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM/ReadVariableOp?
BlockLSTM/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_1?
BlockLSTM/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_2?
BlockLSTM/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_3?
	BlockLSTM	BlockLSTMBlockLSTM/seq_len_max:output:0Reshape:output:0blocklstm_cs_prevblocklstm_h_prev BlockLSTM/ReadVariableOp:value:0"BlockLSTM/ReadVariableOp_1:value:0"BlockLSTM/ReadVariableOp_2:value:0"BlockLSTM/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
	BlockLSTMt
BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM_1/seq_len_max?
BlockLSTM_1/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM_1/ReadVariableOp?
BlockLSTM_1/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_1?
BlockLSTM_1/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_2?
BlockLSTM_1/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_3?
BlockLSTM_1	BlockLSTM BlockLSTM_1/seq_len_max:output:0Reshape_1:output:0blocklstm_cs_prevblocklstm_h_prev"BlockLSTM_1/ReadVariableOp:value:0$BlockLSTM_1/ReadVariableOp_1:value:0$BlockLSTM_1/ReadVariableOp_2:value:0$BlockLSTM_1/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
BlockLSTM_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_2/shapew
	Reshape_2ReshapeBlockLSTM:h:0Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_2w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_3/shapey
	Reshape_3ReshapeBlockLSTM_1:h:0Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_3a
IdentityIdentityReshape_2:output:0*
T0*"
_output_shapes
:~2

Identitye

Identity_1IdentityReshape_3:output:0*
T0*"
_output_shapes
:~2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*\
_input_shapesK
I:?:?:~:~:::::?:K G
#
_output_shapes
:?
 
_user_specified_nameinputs:KG
#
_output_shapes
:?
 
_user_specified_nameinputs:!

_output_shapes	
:?
?
?
8__inference_FirstBlockLSTMModule_layer_call_fn_122102429
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_1221013142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:?2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*#
_output_shapes
:?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*l
_input_shapes[
Y:?????????d:?????????d:~:~:::::?22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????d
"
_user_specified_name
inputs/1:!

_output_shapes	
:?
?)
?
%__inference__traced_restore_122102627
file_prefixD
@assignvariableop_bi_lstm_model_firstblocklstmmodule_w_first_lstmH
Dassignvariableop_1_bi_lstm_model_firstblocklstmmodule_wig_first_lstmH
Dassignvariableop_2_bi_lstm_model_firstblocklstmmodule_wfg_first_lstmH
Dassignvariableop_3_bi_lstm_model_firstblocklstmmodule_wog_first_lstm>
:assignvariableop_4_bi_lstm_model_nextblocklstm_w_next_lstm@
<assignvariableop_5_bi_lstm_model_nextblocklstm_wig_next_lstm@
<assignvariableop_6_bi_lstm_model_nextblocklstm_wfg_next_lstm@
<assignvariableop_7_bi_lstm_model_nextblocklstm_wog_next_lstm

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B2blockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUEB6blockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUEB7blockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUEB7blockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUEB6nextBlockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUEB:nextBlockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUEB;nextBlockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUEB;nextBlockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp@assignvariableop_bi_lstm_model_firstblocklstmmodule_w_first_lstmIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpDassignvariableop_1_bi_lstm_model_firstblocklstmmodule_wig_first_lstmIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpDassignvariableop_2_bi_lstm_model_firstblocklstmmodule_wfg_first_lstmIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpDassignvariableop_3_bi_lstm_model_firstblocklstmmodule_wog_first_lstmIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp:assignvariableop_4_bi_lstm_model_nextblocklstm_w_next_lstmIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp<assignvariableop_5_bi_lstm_model_nextblocklstm_wig_next_lstmIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp<assignvariableop_6_bi_lstm_model_nextblocklstm_wfg_next_lstmIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp<assignvariableop_7_bi_lstm_model_nextblocklstm_wog_next_lstmIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?M
?
'__inference_signature_wrapper_122101994
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*8
Tout0
.2,*
_collective_manager_ids
 *?
_output_shapes?
?:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_1221008962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_11?
Identity_12Identity!StatefulPartitionedCall:output:12^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_12?
Identity_13Identity!StatefulPartitionedCall:output:13^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_13?
Identity_14Identity!StatefulPartitionedCall:output:14^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_14?
Identity_15Identity!StatefulPartitionedCall:output:15^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_15?
Identity_16Identity!StatefulPartitionedCall:output:16^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_16?
Identity_17Identity!StatefulPartitionedCall:output:17^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_17?
Identity_18Identity!StatefulPartitionedCall:output:18^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_18?
Identity_19Identity!StatefulPartitionedCall:output:19^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_19?
Identity_20Identity!StatefulPartitionedCall:output:20^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_20?
Identity_21Identity!StatefulPartitionedCall:output:21^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_21?
Identity_22Identity!StatefulPartitionedCall:output:22^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_22?
Identity_23Identity!StatefulPartitionedCall:output:23^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_23?
Identity_24Identity!StatefulPartitionedCall:output:24^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_24?
Identity_25Identity!StatefulPartitionedCall:output:25^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_25?
Identity_26Identity!StatefulPartitionedCall:output:26^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_26?
Identity_27Identity!StatefulPartitionedCall:output:27^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_27?
Identity_28Identity!StatefulPartitionedCall:output:28^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_28?
Identity_29Identity!StatefulPartitionedCall:output:29^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_29?
Identity_30Identity!StatefulPartitionedCall:output:30^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_30?
Identity_31Identity!StatefulPartitionedCall:output:31^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_31?
Identity_32Identity!StatefulPartitionedCall:output:32^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_32?
Identity_33Identity!StatefulPartitionedCall:output:33^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_33?
Identity_34Identity!StatefulPartitionedCall:output:34^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_34?
Identity_35Identity!StatefulPartitionedCall:output:35^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_35?
Identity_36Identity!StatefulPartitionedCall:output:36^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_36?
Identity_37Identity!StatefulPartitionedCall:output:37^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_37?
Identity_38Identity!StatefulPartitionedCall:output:38^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_38?
Identity_39Identity!StatefulPartitionedCall:output:39^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_39?
Identity_40Identity!StatefulPartitionedCall:output:40^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_40?
Identity_41Identity!StatefulPartitionedCall:output:41^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_41?
Identity_42Identity!StatefulPartitionedCall:output:42^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_42?
Identity_43Identity!StatefulPartitionedCall:output:43^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_43"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_4Identity_4:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????d:?????????d:~:~:::::?:~:~:::::?22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????d
!
_user_specified_name	input_2:!

_output_shapes	
:?:!

_output_shapes	
:?
?"
?
"__inference__traced_save_122102593
file_prefixN
Jsavev2_bi_lstm_model_firstblocklstmmodule_w_first_lstm_read_readvariableopP
Lsavev2_bi_lstm_model_firstblocklstmmodule_wig_first_lstm_read_readvariableopP
Lsavev2_bi_lstm_model_firstblocklstmmodule_wfg_first_lstm_read_readvariableopP
Lsavev2_bi_lstm_model_firstblocklstmmodule_wog_first_lstm_read_readvariableopF
Bsavev2_bi_lstm_model_nextblocklstm_w_next_lstm_read_readvariableopH
Dsavev2_bi_lstm_model_nextblocklstm_wig_next_lstm_read_readvariableopH
Dsavev2_bi_lstm_model_nextblocklstm_wfg_next_lstm_read_readvariableopH
Dsavev2_bi_lstm_model_nextblocklstm_wog_next_lstm_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_22d28b3ec1fe4c7fbc13a4a87ac089a0/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B2blockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUEB6blockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUEB7blockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUEB7blockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUEB6nextBlockLstm/weight_matrix/.ATTRIBUTES/VARIABLE_VALUEB:nextBlockLstm/weight_input_gate/.ATTRIBUTES/VARIABLE_VALUEB;nextBlockLstm/weight_forget_gate/.ATTRIBUTES/VARIABLE_VALUEB;nextBlockLstm/weight_output_gate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Jsavev2_bi_lstm_model_firstblocklstmmodule_w_first_lstm_read_readvariableopLsavev2_bi_lstm_model_firstblocklstmmodule_wig_first_lstm_read_readvariableopLsavev2_bi_lstm_model_firstblocklstmmodule_wfg_first_lstm_read_readvariableopLsavev2_bi_lstm_model_firstblocklstmmodule_wog_first_lstm_read_readvariableopBsavev2_bi_lstm_model_nextblocklstm_w_next_lstm_read_readvariableopDsavev2_bi_lstm_model_nextblocklstm_wig_next_lstm_read_readvariableopDsavev2_bi_lstm_model_nextblocklstm_wfg_next_lstm_read_readvariableopDsavev2_bi_lstm_model_nextblocklstm_wog_next_lstm_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*S
_input_shapesB
@: :
??:~:~:~:
??:~:~:~: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??: 

_output_shapes
:~: 

_output_shapes
:~: 

_output_shapes
:~:&"
 
_output_shapes
:
??: 

_output_shapes
:~: 

_output_shapes
:~: 

_output_shapes
:~:	

_output_shapes
: 
إ
?

L__inference_bi_lstm_model_layer_call_and_return_conditional_losses_122101751
input_1
input_2"
firstblocklstmmodule_122101338"
firstblocklstmmodule_122101340"
firstblocklstmmodule_122101342"
firstblocklstmmodule_122101344"
firstblocklstmmodule_122101346"
firstblocklstmmodule_122101348"
firstblocklstmmodule_122101350
nextblocklstm_122101423
nextblocklstm_122101425
nextblocklstm_122101427
nextblocklstm_122101429
nextblocklstm_122101431
nextblocklstm_122101433
nextblocklstm_122101435
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43??,FirstBlockLSTMModule/StatefulPartitionedCall?%NextBlockLSTM/StatefulPartitionedCall?
,FirstBlockLSTMModule/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2firstblocklstmmodule_122101338firstblocklstmmodule_122101340firstblocklstmmodule_122101342firstblocklstmmodule_122101344firstblocklstmmodule_122101346firstblocklstmmodule_122101348firstblocklstmmodule_122101350*
Tin
2	*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:?:?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_1221013142.
,FirstBlockLSTMModule/StatefulPartitionedCall?
%NextBlockLSTM/StatefulPartitionedCallStatefulPartitionedCall5FirstBlockLSTMModule/StatefulPartitionedCall:output:05FirstBlockLSTMModule/StatefulPartitionedCall:output:1nextblocklstm_122101423nextblocklstm_122101425nextblocklstm_122101427nextblocklstm_122101429nextblocklstm_122101431nextblocklstm_122101433nextblocklstm_122101435*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:~:~*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_1221013992'
%NextBlockLSTM/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ~   2
Reshape/shape?
ReshapeReshape.NextBlockLSTM/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:~2	
Reshapes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ~   2
Reshape_1/shape?
	Reshape_1Reshape.NextBlockLSTM/StatefulPartitionedCall:output:1Reshape_1/shape:output:0*
T0*
_output_shapes

:~2
	Reshape_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceReshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slicep
strided_slice_1/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_1/stack?
 strided_slice_1/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_1/stack_1/values_0?
strided_slice_1/stack_1Pack)strided_slice_1/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_1/stack_1t
strided_slice_1/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_1/stack_2?
 strided_slice_1/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_1/stack_3/values_0?
strided_slice_1/stack_3Pack)strided_slice_1/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_1/stack_3|
strided_slice_1/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_4?
strided_slice_1StridedSliceReshape_1:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_3:output:0 strided_slice_1/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReshape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_2p
strided_slice_3/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack?
 strided_slice_3/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_1/values_0?
strided_slice_3/stack_1Pack)strided_slice_3/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_1t
strided_slice_3/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack_2?
 strided_slice_3/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_3/values_0?
strided_slice_3/stack_3Pack)strided_slice_3/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_3|
strided_slice_3/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_4?
strided_slice_3StridedSliceReshape_1:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_3:output:0 strided_slice_3/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReshape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_4p
strided_slice_5/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack?
 strided_slice_5/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_1/values_0?
strided_slice_5/stack_1Pack)strided_slice_5/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_1t
strided_slice_5/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack_2?
 strided_slice_5/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_3/values_0?
strided_slice_5/stack_3Pack)strided_slice_5/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_3|
strided_slice_5/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_4?
strided_slice_5StridedSliceReshape_1:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_3:output:0 strided_slice_5/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReshape:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_6p
strided_slice_7/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack?
 strided_slice_7/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_1/values_0?
strided_slice_7/stack_1Pack)strided_slice_7/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_1t
strided_slice_7/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack_2?
 strided_slice_7/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_3/values_0?
strided_slice_7/stack_3Pack)strided_slice_7/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_3|
strided_slice_7/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_4?
strided_slice_7StridedSliceReshape_1:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_3:output:0 strided_slice_7/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReshape:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_8p
strided_slice_9/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack?
 strided_slice_9/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_1/values_0?
strided_slice_9/stack_1Pack)strided_slice_9/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_1t
strided_slice_9/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack_2?
 strided_slice_9/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_3/values_0?
strided_slice_9/stack_3Pack)strided_slice_9/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_3|
strided_slice_9/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_4?
strided_slice_9StridedSliceReshape_1:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_3:output:0 strided_slice_9/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_9z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReshape:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_10r
strided_slice_11/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack?
!strided_slice_11/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_1/values_0?
strided_slice_11/stack_1Pack*strided_slice_11/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_1v
strided_slice_11/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack_2?
!strided_slice_11/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_3/values_0?
strided_slice_11/stack_3Pack*strided_slice_11/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_3~
strided_slice_11/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_4?
strided_slice_11StridedSliceReshape_1:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_3:output:0!strided_slice_11/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_11z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSliceReshape:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_12r
strided_slice_13/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack?
!strided_slice_13/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_1/values_0?
strided_slice_13/stack_1Pack*strided_slice_13/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_1v
strided_slice_13/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack_2?
!strided_slice_13/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_3/values_0?
strided_slice_13/stack_3Pack*strided_slice_13/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_3~
strided_slice_13/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_4?
strided_slice_13StridedSliceReshape_1:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_3:output:0!strided_slice_13/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_13z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSliceReshape:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_14r
strided_slice_15/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack?
!strided_slice_15/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_1/values_0?
strided_slice_15/stack_1Pack*strided_slice_15/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_1v
strided_slice_15/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack_2?
!strided_slice_15/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_3/values_0?
strided_slice_15/stack_3Pack*strided_slice_15/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_3~
strided_slice_15/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_4?
strided_slice_15StridedSliceReshape_1:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_3:output:0!strided_slice_15/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_15z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSliceReshape:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_16r
strided_slice_17/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack?
!strided_slice_17/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_1/values_0?
strided_slice_17/stack_1Pack*strided_slice_17/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_1v
strided_slice_17/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack_2?
!strided_slice_17/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_3/values_0?
strided_slice_17/stack_3Pack*strided_slice_17/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_3~
strided_slice_17/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_4?
strided_slice_17StridedSliceReshape_1:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_3:output:0!strided_slice_17/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_17z
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_18/stack~
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_18/stack_1~
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_2?
strided_slice_18StridedSliceReshape:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_18r
strided_slice_19/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack?
!strided_slice_19/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_1/values_0?
strided_slice_19/stack_1Pack*strided_slice_19/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_1v
strided_slice_19/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack_2?
!strided_slice_19/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_3/values_0?
strided_slice_19/stack_3Pack*strided_slice_19/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_3~
strided_slice_19/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack_4?
strided_slice_19StridedSliceReshape_1:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_3:output:0!strided_slice_19/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_19z
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_20/stack~
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_1~
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_2?
strided_slice_20StridedSliceReshape:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_20r
strided_slice_21/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack?
!strided_slice_21/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_1/values_0?
strided_slice_21/stack_1Pack*strided_slice_21/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_1v
strided_slice_21/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack_2?
!strided_slice_21/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_3/values_0?
strided_slice_21/stack_3Pack*strided_slice_21/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_3~
strided_slice_21/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack_4?
strided_slice_21StridedSliceReshape_1:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_3:output:0!strided_slice_21/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_21z
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack~
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_1~
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_2?
strided_slice_22StridedSliceReshape:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_22r
strided_slice_23/stackConst*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_23/stack?
!strided_slice_23/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_23/stack_1/values_0?
strided_slice_23/stack_1Pack*strided_slice_23/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_1v
strided_slice_23/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_23/stack_2?
!strided_slice_23/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_23/stack_3/values_0?
strided_slice_23/stack_3Pack*strided_slice_23/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_3~
strided_slice_23/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack_4?
strided_slice_23StridedSliceReshape_1:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_3:output:0!strided_slice_23/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_23z
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack~
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_1~
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_2?
strided_slice_24StridedSliceReshape:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_24r
strided_slice_25/stackConst*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_25/stack?
!strided_slice_25/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_25/stack_1/values_0?
strided_slice_25/stack_1Pack*strided_slice_25/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_1v
strided_slice_25/stack_2Const*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_25/stack_2?
!strided_slice_25/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_25/stack_3/values_0?
strided_slice_25/stack_3Pack*strided_slice_25/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_3~
strided_slice_25/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack_4?
strided_slice_25StridedSliceReshape_1:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_3:output:0!strided_slice_25/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_25z
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack~
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_1~
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_2?
strided_slice_26StridedSliceReshape:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_26r
strided_slice_27/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_27/stack?
!strided_slice_27/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_27/stack_1/values_0?
strided_slice_27/stack_1Pack*strided_slice_27/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_1v
strided_slice_27/stack_2Const*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_27/stack_2?
!strided_slice_27/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_27/stack_3/values_0?
strided_slice_27/stack_3Pack*strided_slice_27/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_3~
strided_slice_27/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack_4?
strided_slice_27StridedSliceReshape_1:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_3:output:0!strided_slice_27/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_27z
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack~
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_1~
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_2?
strided_slice_28StridedSliceReshape:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_28r
strided_slice_29/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_29/stack?
!strided_slice_29/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_29/stack_1/values_0?
strided_slice_29/stack_1Pack*strided_slice_29/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_1v
strided_slice_29/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_29/stack_2?
!strided_slice_29/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_29/stack_3/values_0?
strided_slice_29/stack_3Pack*strided_slice_29/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_3~
strided_slice_29/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack_4?
strided_slice_29StridedSliceReshape_1:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_3:output:0!strided_slice_29/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_29z
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack~
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_1~
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_2?
strided_slice_30StridedSliceReshape:output:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_30r
strided_slice_31/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack?
!strided_slice_31/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_1/values_0?
strided_slice_31/stack_1Pack*strided_slice_31/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_1v
strided_slice_31/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack_2?
!strided_slice_31/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_3/values_0?
strided_slice_31/stack_3Pack*strided_slice_31/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_3~
strided_slice_31/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_31/stack_4?
strided_slice_31StridedSliceReshape_1:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_3:output:0!strided_slice_31/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_31z
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack~
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_1~
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_2?
strided_slice_32StridedSliceReshape:output:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_32r
strided_slice_33/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack?
!strided_slice_33/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_1/values_0?
strided_slice_33/stack_1Pack*strided_slice_33/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_1v
strided_slice_33/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack_2?
!strided_slice_33/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_3/values_0?
strided_slice_33/stack_3Pack*strided_slice_33/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_3~
strided_slice_33/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_33/stack_4?
strided_slice_33StridedSliceReshape_1:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_3:output:0!strided_slice_33/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_33z
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack~
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_1~
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_2?
strided_slice_34StridedSliceReshape:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_34r
strided_slice_35/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack?
!strided_slice_35/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_1/values_0?
strided_slice_35/stack_1Pack*strided_slice_35/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_1v
strided_slice_35/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack_2?
!strided_slice_35/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_3/values_0?
strided_slice_35/stack_3Pack*strided_slice_35/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_3~
strided_slice_35/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_35/stack_4?
strided_slice_35StridedSliceReshape_1:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_3:output:0!strided_slice_35/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_35z
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack~
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_1~
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_2?
strided_slice_36StridedSliceReshape:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_36r
strided_slice_37/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack?
!strided_slice_37/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_1/values_0?
strided_slice_37/stack_1Pack*strided_slice_37/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_1v
strided_slice_37/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack_2?
!strided_slice_37/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_3/values_0?
strided_slice_37/stack_3Pack*strided_slice_37/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_3~
strided_slice_37/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_37/stack_4?
strided_slice_37StridedSliceReshape_1:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_3:output:0!strided_slice_37/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_37z
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack~
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_1~
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_2?
strided_slice_38StridedSliceReshape:output:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_38r
strided_slice_39/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack?
!strided_slice_39/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_1/values_0?
strided_slice_39/stack_1Pack*strided_slice_39/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_1v
strided_slice_39/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack_2?
!strided_slice_39/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_3/values_0?
strided_slice_39/stack_3Pack*strided_slice_39/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_3~
strided_slice_39/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_39/stack_4?
strided_slice_39StridedSliceReshape_1:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_3:output:0!strided_slice_39/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_39z
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack~
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_1~
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_2?
strided_slice_40StridedSliceReshape:output:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_40r
strided_slice_41/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack?
!strided_slice_41/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_1/values_0?
strided_slice_41/stack_1Pack*strided_slice_41/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_1v
strided_slice_41/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack_2?
!strided_slice_41/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_3/values_0?
strided_slice_41/stack_3Pack*strided_slice_41/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_3~
strided_slice_41/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_41/stack_4?
strided_slice_41StridedSliceReshape_1:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_3:output:0!strided_slice_41/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_41z
strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack~
strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_1~
strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_2?
strided_slice_42StridedSliceReshape:output:0strided_slice_42/stack:output:0!strided_slice_42/stack_1:output:0!strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_42r
strided_slice_43/stackConst*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_43/stack?
!strided_slice_43/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B : 2#
!strided_slice_43/stack_1/values_0?
strided_slice_43/stack_1Pack*strided_slice_43/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_1v
strided_slice_43/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_43/stack_2?
!strided_slice_43/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_43/stack_3/values_0?
strided_slice_43/stack_3Pack*strided_slice_43/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_3~
strided_slice_43/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_43/stack_4?
strided_slice_43StridedSliceReshape_1:output:0!strided_slice_43/stack_1:output:0!strided_slice_43/stack_3:output:0!strided_slice_43/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_43?
IdentityIdentitystrided_slice:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity?

Identity_1Identitystrided_slice_1:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_1?

Identity_2Identitystrided_slice_2:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_2?

Identity_3Identitystrided_slice_3:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_3?

Identity_4Identitystrided_slice_4:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_4?

Identity_5Identitystrided_slice_5:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_5?

Identity_6Identitystrided_slice_6:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_6?

Identity_7Identitystrided_slice_7:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_7?

Identity_8Identitystrided_slice_8:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_8?

Identity_9Identitystrided_slice_9:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_9?
Identity_10Identitystrided_slice_10:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_10?
Identity_11Identitystrided_slice_11:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_11?
Identity_12Identitystrided_slice_12:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_12?
Identity_13Identitystrided_slice_13:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_13?
Identity_14Identitystrided_slice_14:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_14?
Identity_15Identitystrided_slice_15:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_15?
Identity_16Identitystrided_slice_16:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_16?
Identity_17Identitystrided_slice_17:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_17?
Identity_18Identitystrided_slice_18:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_18?
Identity_19Identitystrided_slice_19:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_19?
Identity_20Identitystrided_slice_20:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_20?
Identity_21Identitystrided_slice_21:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_21?
Identity_22Identitystrided_slice_22:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_22?
Identity_23Identitystrided_slice_23:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_23?
Identity_24Identitystrided_slice_24:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_24?
Identity_25Identitystrided_slice_25:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_25?
Identity_26Identitystrided_slice_26:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_26?
Identity_27Identitystrided_slice_27:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_27?
Identity_28Identitystrided_slice_28:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_28?
Identity_29Identitystrided_slice_29:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_29?
Identity_30Identitystrided_slice_30:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_30?
Identity_31Identitystrided_slice_31:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_31?
Identity_32Identitystrided_slice_32:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_32?
Identity_33Identitystrided_slice_33:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_33?
Identity_34Identitystrided_slice_34:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_34?
Identity_35Identitystrided_slice_35:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_35?
Identity_36Identitystrided_slice_36:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_36?
Identity_37Identitystrided_slice_37:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_37?
Identity_38Identitystrided_slice_38:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_38?
Identity_39Identitystrided_slice_39:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_39?
Identity_40Identitystrided_slice_40:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_40?
Identity_41Identitystrided_slice_41:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_41?
Identity_42Identitystrided_slice_42:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_42?
Identity_43Identitystrided_slice_43:output:0-^FirstBlockLSTMModule/StatefulPartitionedCall&^NextBlockLSTM/StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_43"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_4Identity_4:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????d:?????????d:~:~:::::?:~:~:::::?2\
,FirstBlockLSTMModule/StatefulPartitionedCall,FirstBlockLSTMModule/StatefulPartitionedCall2N
%NextBlockLSTM/StatefulPartitionedCall%NextBlockLSTM/StatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????d
!
_user_specified_name	input_2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
?
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_122102474
inputs_0
inputs_1
blocklstm_cs_prev
blocklstm_h_prev%
!blocklstm_readvariableop_resource'
#blocklstm_readvariableop_1_resource'
#blocklstm_readvariableop_2_resource'
#blocklstm_readvariableop_3_resource
blocklstm_b
identity

identity_1?s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2
Reshape/shapem
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*#
_output_shapes
:?2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?   2
Reshape_1/shapes
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*#
_output_shapes
:?2
	Reshape_1p
BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM/seq_len_max?
BlockLSTM/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM/ReadVariableOp?
BlockLSTM/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_1?
BlockLSTM/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_2?
BlockLSTM/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_3?
	BlockLSTM	BlockLSTMBlockLSTM/seq_len_max:output:0Reshape:output:0blocklstm_cs_prevblocklstm_h_prev BlockLSTM/ReadVariableOp:value:0"BlockLSTM/ReadVariableOp_1:value:0"BlockLSTM/ReadVariableOp_2:value:0"BlockLSTM/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
	BlockLSTMt
BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM_1/seq_len_max?
BlockLSTM_1/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM_1/ReadVariableOp?
BlockLSTM_1/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_1?
BlockLSTM_1/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_2?
BlockLSTM_1/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_3?
BlockLSTM_1	BlockLSTM BlockLSTM_1/seq_len_max:output:0Reshape_1:output:0blocklstm_cs_prevblocklstm_h_prev"BlockLSTM_1/ReadVariableOp:value:0$BlockLSTM_1/ReadVariableOp_1:value:0$BlockLSTM_1/ReadVariableOp_2:value:0$BlockLSTM_1/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
BlockLSTM_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_2/shapew
	Reshape_2ReshapeBlockLSTM:h:0Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_2w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_3/shapey
	Reshape_3ReshapeBlockLSTM_1:h:0Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_3a
IdentityIdentityReshape_2:output:0*
T0*"
_output_shapes
:~2

Identitye

Identity_1IdentityReshape_3:output:0*
T0*"
_output_shapes
:~2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*\
_input_shapesK
I:?:?:~:~:::::?:M I
#
_output_shapes
:?
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?
"
_user_specified_name
inputs/1:!

_output_shapes	
:?
??
?
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_122101314

inputs
inputs_1
blocklstm_cs_prev
blocklstm_h_prev%
!blocklstm_readvariableop_resource'
#blocklstm_readvariableop_1_resource'
#blocklstm_readvariableop_2_resource'
#blocklstm_readvariableop_3_resource
blocklstm_b
identity

identity_1?s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   2
Reshape/shapej
ReshapeReshapeinputsReshape/shape:output:0*
T0*"
_output_shapes
:d2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   2
Reshape_1/shaper
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*"
_output_shapes
:d2
	Reshape_1p
BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM/seq_len_max?
BlockLSTM/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM/ReadVariableOp?
BlockLSTM/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_1?
BlockLSTM/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_2?
BlockLSTM/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_3?
	BlockLSTM	BlockLSTMBlockLSTM/seq_len_max:output:0Reshape:output:0blocklstm_cs_prevblocklstm_h_prev BlockLSTM/ReadVariableOp:value:0"BlockLSTM/ReadVariableOp_1:value:0"BlockLSTM/ReadVariableOp_2:value:0"BlockLSTM/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
	BlockLSTMt
BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM_1/seq_len_max?
BlockLSTM_1/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM_1/ReadVariableOp?
BlockLSTM_1/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_1?
BlockLSTM_1/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_2?
BlockLSTM_1/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_3?
BlockLSTM_1	BlockLSTM BlockLSTM_1/seq_len_max:output:0Reshape_1:output:0blocklstm_cs_prevblocklstm_h_prev"BlockLSTM_1/ReadVariableOp:value:0$BlockLSTM_1/ReadVariableOp_1:value:0$BlockLSTM_1/ReadVariableOp_2:value:0$BlockLSTM_1/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
BlockLSTM_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_2/shapew
	Reshape_2ReshapeBlockLSTM:h:0Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_2w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_3/shapey
	Reshape_3ReshapeBlockLSTM_1:h:0Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_3t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceReshape_2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReshape_3:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_2p
strided_slice_3/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack?
 strided_slice_3/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_1/values_0?
strided_slice_3/stack_1Pack)strided_slice_3/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_1t
strided_slice_3/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack_2?
 strided_slice_3/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_3/values_0?
strided_slice_3/stack_3Pack)strided_slice_3/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_3|
strided_slice_3/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_4?
strided_slice_3StridedSlicestrided_slice_1:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_3:output:0 strided_slice_3/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concats
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_4/shapev
	Reshape_4Reshapeconcat:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_4p
strided_slice_5/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack?
 strided_slice_5/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_1/values_0?
strided_slice_5/stack_1Pack)strided_slice_5/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_1t
strided_slice_5/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack_2?
 strided_slice_5/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_3/values_0?
strided_slice_5/stack_3Pack)strided_slice_5/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_3|
strided_slice_5/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_4?
strided_slice_5StridedSlicestrided_slice_1:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_3:output:0 strided_slice_5/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_5`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2strided_slice_4:output:0strided_slice_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_5/shapex
	Reshape_5Reshapeconcat_1:output:0Reshape_5/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_6p
strided_slice_7/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack?
 strided_slice_7/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_1/values_0?
strided_slice_7/stack_1Pack)strided_slice_7/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_1t
strided_slice_7/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack_2?
 strided_slice_7/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_3/values_0?
strided_slice_7/stack_3Pack)strided_slice_7/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_3|
strided_slice_7/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_4?
strided_slice_7StridedSlicestrided_slice_1:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_3:output:0 strided_slice_7/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_7`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis?
concat_2ConcatV2strided_slice_6:output:0strided_slice_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_2s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_6/shapex
	Reshape_6Reshapeconcat_2:output:0Reshape_6/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_6x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_8p
strided_slice_9/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack?
 strided_slice_9/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_1/values_0?
strided_slice_9/stack_1Pack)strided_slice_9/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_1t
strided_slice_9/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack_2?
 strided_slice_9/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_3/values_0?
strided_slice_9/stack_3Pack)strided_slice_9/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_3|
strided_slice_9/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_4?
strided_slice_9StridedSlicestrided_slice_1:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_3:output:0 strided_slice_9/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_9`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis?
concat_3ConcatV2strided_slice_8:output:0strided_slice_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_3s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_7/shapex
	Reshape_7Reshapeconcat_3:output:0Reshape_7/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_7z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_10r
strided_slice_11/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack?
!strided_slice_11/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_1/values_0?
strided_slice_11/stack_1Pack*strided_slice_11/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_1v
strided_slice_11/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack_2?
!strided_slice_11/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_3/values_0?
strided_slice_11/stack_3Pack*strided_slice_11/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_3~
strided_slice_11/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_4?
strided_slice_11StridedSlicestrided_slice_1:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_3:output:0!strided_slice_11/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_11`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis?
concat_4ConcatV2strided_slice_10:output:0strided_slice_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_4s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_8/shapex
	Reshape_8Reshapeconcat_4:output:0Reshape_8/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_8z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_12r
strided_slice_13/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack?
!strided_slice_13/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_1/values_0?
strided_slice_13/stack_1Pack*strided_slice_13/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_1v
strided_slice_13/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack_2?
!strided_slice_13/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_3/values_0?
strided_slice_13/stack_3Pack*strided_slice_13/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_3~
strided_slice_13/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_4?
strided_slice_13StridedSlicestrided_slice_1:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_3:output:0!strided_slice_13/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_13`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis?
concat_5ConcatV2strided_slice_12:output:0strided_slice_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_5s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_9/shapex
	Reshape_9Reshapeconcat_5:output:0Reshape_9/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_9z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_14r
strided_slice_15/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack?
!strided_slice_15/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_1/values_0?
strided_slice_15/stack_1Pack*strided_slice_15/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_1v
strided_slice_15/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack_2?
!strided_slice_15/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_3/values_0?
strided_slice_15/stack_3Pack*strided_slice_15/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_3~
strided_slice_15/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_4?
strided_slice_15StridedSlicestrided_slice_1:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_3:output:0!strided_slice_15/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_15`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis?
concat_6ConcatV2strided_slice_14:output:0strided_slice_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_6u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_10/shape{

Reshape_10Reshapeconcat_6:output:0Reshape_10/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_10z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_16r
strided_slice_17/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack?
!strided_slice_17/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_1/values_0?
strided_slice_17/stack_1Pack*strided_slice_17/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_1v
strided_slice_17/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack_2?
!strided_slice_17/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_3/values_0?
strided_slice_17/stack_3Pack*strided_slice_17/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_3~
strided_slice_17/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_4?
strided_slice_17StridedSlicestrided_slice_1:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_3:output:0!strided_slice_17/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_17`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis?
concat_7ConcatV2strided_slice_16:output:0strided_slice_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_7u
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_11/shape{

Reshape_11Reshapeconcat_7:output:0Reshape_11/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_11z
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack~
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_18/stack_1~
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_18r
strided_slice_19/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack?
!strided_slice_19/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_1/values_0?
strided_slice_19/stack_1Pack*strided_slice_19/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_1v
strided_slice_19/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack_2?
!strided_slice_19/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_3/values_0?
strided_slice_19/stack_3Pack*strided_slice_19/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_3~
strided_slice_19/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack_4?
strided_slice_19StridedSlicestrided_slice_1:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_3:output:0!strided_slice_19/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_19`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis?
concat_8ConcatV2strided_slice_18:output:0strided_slice_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_8u
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_12/shape{

Reshape_12Reshapeconcat_8:output:0Reshape_12/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_12z
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_20/stack~
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_20/stack_1~
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_20r
strided_slice_21/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack?
!strided_slice_21/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_1/values_0?
strided_slice_21/stack_1Pack*strided_slice_21/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_1v
strided_slice_21/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack_2?
!strided_slice_21/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_3/values_0?
strided_slice_21/stack_3Pack*strided_slice_21/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_3~
strided_slice_21/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack_4?
strided_slice_21StridedSlicestrided_slice_1:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_3:output:0!strided_slice_21/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_21`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis?
concat_9ConcatV2strided_slice_20:output:0strided_slice_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_9u
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_13/shape{

Reshape_13Reshapeconcat_9:output:0Reshape_13/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_13z
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_22/stack~
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_1~
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_22r
strided_slice_23/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_23/stack?
!strided_slice_23/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_23/stack_1/values_0?
strided_slice_23/stack_1Pack*strided_slice_23/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_1v
strided_slice_23/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_23/stack_2?
!strided_slice_23/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_23/stack_3/values_0?
strided_slice_23/stack_3Pack*strided_slice_23/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_3~
strided_slice_23/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack_4?
strided_slice_23StridedSlicestrided_slice_1:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_3:output:0!strided_slice_23/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_23b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_10/axis?
	concat_10ConcatV2strided_slice_22:output:0strided_slice_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_10u
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_14/shape|

Reshape_14Reshapeconcat_10:output:0Reshape_14/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_14z
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack~
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_1~
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_24r
strided_slice_25/stackConst*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_25/stack?
!strided_slice_25/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_25/stack_1/values_0?
strided_slice_25/stack_1Pack*strided_slice_25/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_1v
strided_slice_25/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_25/stack_2?
!strided_slice_25/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_25/stack_3/values_0?
strided_slice_25/stack_3Pack*strided_slice_25/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_3~
strided_slice_25/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack_4?
strided_slice_25StridedSlicestrided_slice_1:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_3:output:0!strided_slice_25/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_25b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_11/axis?
	concat_11ConcatV2strided_slice_24:output:0strided_slice_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_11u
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_15/shape|

Reshape_15Reshapeconcat_11:output:0Reshape_15/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_15z
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack~
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_1~
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_26r
strided_slice_27/stackConst*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_27/stack?
!strided_slice_27/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_27/stack_1/values_0?
strided_slice_27/stack_1Pack*strided_slice_27/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_1v
strided_slice_27/stack_2Const*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_27/stack_2?
!strided_slice_27/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_27/stack_3/values_0?
strided_slice_27/stack_3Pack*strided_slice_27/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_3~
strided_slice_27/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack_4?
strided_slice_27StridedSlicestrided_slice_1:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_3:output:0!strided_slice_27/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_27b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_12/axis?
	concat_12ConcatV2strided_slice_26:output:0strided_slice_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_12u
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_16/shape|

Reshape_16Reshapeconcat_12:output:0Reshape_16/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_16z
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack~
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_1~
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_2?
strided_slice_28StridedSlicestrided_slice:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_28r
strided_slice_29/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_29/stack?
!strided_slice_29/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_29/stack_1/values_0?
strided_slice_29/stack_1Pack*strided_slice_29/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_1v
strided_slice_29/stack_2Const*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_29/stack_2?
!strided_slice_29/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_29/stack_3/values_0?
strided_slice_29/stack_3Pack*strided_slice_29/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_3~
strided_slice_29/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack_4?
strided_slice_29StridedSlicestrided_slice_1:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_3:output:0!strided_slice_29/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_29b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_13/axis?
	concat_13ConcatV2strided_slice_28:output:0strided_slice_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_13u
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_17/shape|

Reshape_17Reshapeconcat_13:output:0Reshape_17/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_17z
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack~
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_1~
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_2?
strided_slice_30StridedSlicestrided_slice:output:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_30r
strided_slice_31/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack?
!strided_slice_31/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_1/values_0?
strided_slice_31/stack_1Pack*strided_slice_31/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_1v
strided_slice_31/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack_2?
!strided_slice_31/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_3/values_0?
strided_slice_31/stack_3Pack*strided_slice_31/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_3~
strided_slice_31/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_31/stack_4?
strided_slice_31StridedSlicestrided_slice_1:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_3:output:0!strided_slice_31/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_31b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_14/axis?
	concat_14ConcatV2strided_slice_30:output:0strided_slice_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_14u
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_18/shape|

Reshape_18Reshapeconcat_14:output:0Reshape_18/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_18z
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack~
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_1~
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_2?
strided_slice_32StridedSlicestrided_slice:output:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_32r
strided_slice_33/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack?
!strided_slice_33/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_1/values_0?
strided_slice_33/stack_1Pack*strided_slice_33/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_1v
strided_slice_33/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack_2?
!strided_slice_33/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_3/values_0?
strided_slice_33/stack_3Pack*strided_slice_33/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_3~
strided_slice_33/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_33/stack_4?
strided_slice_33StridedSlicestrided_slice_1:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_3:output:0!strided_slice_33/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_33b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_15/axis?
	concat_15ConcatV2strided_slice_32:output:0strided_slice_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_15u
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_19/shape|

Reshape_19Reshapeconcat_15:output:0Reshape_19/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_19z
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack~
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_1~
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_2?
strided_slice_34StridedSlicestrided_slice:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_34r
strided_slice_35/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack?
!strided_slice_35/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_1/values_0?
strided_slice_35/stack_1Pack*strided_slice_35/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_1v
strided_slice_35/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack_2?
!strided_slice_35/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_3/values_0?
strided_slice_35/stack_3Pack*strided_slice_35/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_3~
strided_slice_35/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_35/stack_4?
strided_slice_35StridedSlicestrided_slice_1:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_3:output:0!strided_slice_35/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_35b
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_16/axis?
	concat_16ConcatV2strided_slice_34:output:0strided_slice_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_16u
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_20/shape|

Reshape_20Reshapeconcat_16:output:0Reshape_20/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_20z
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack~
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_1~
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_2?
strided_slice_36StridedSlicestrided_slice:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_36r
strided_slice_37/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack?
!strided_slice_37/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_1/values_0?
strided_slice_37/stack_1Pack*strided_slice_37/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_1v
strided_slice_37/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack_2?
!strided_slice_37/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_3/values_0?
strided_slice_37/stack_3Pack*strided_slice_37/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_3~
strided_slice_37/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_37/stack_4?
strided_slice_37StridedSlicestrided_slice_1:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_3:output:0!strided_slice_37/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_37b
concat_17/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_17/axis?
	concat_17ConcatV2strided_slice_36:output:0strided_slice_37:output:0concat_17/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_17u
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_21/shape|

Reshape_21Reshapeconcat_17:output:0Reshape_21/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_21z
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack~
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_1~
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_2?
strided_slice_38StridedSlicestrided_slice:output:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_38r
strided_slice_39/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack?
!strided_slice_39/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_1/values_0?
strided_slice_39/stack_1Pack*strided_slice_39/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_1v
strided_slice_39/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack_2?
!strided_slice_39/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_3/values_0?
strided_slice_39/stack_3Pack*strided_slice_39/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_3~
strided_slice_39/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_39/stack_4?
strided_slice_39StridedSlicestrided_slice_1:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_3:output:0!strided_slice_39/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_39b
concat_18/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_18/axis?
	concat_18ConcatV2strided_slice_38:output:0strided_slice_39:output:0concat_18/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_18u
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_22/shape|

Reshape_22Reshapeconcat_18:output:0Reshape_22/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_22z
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack~
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_1~
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_2?
strided_slice_40StridedSlicestrided_slice:output:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_40r
strided_slice_41/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack?
!strided_slice_41/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_1/values_0?
strided_slice_41/stack_1Pack*strided_slice_41/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_1v
strided_slice_41/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack_2?
!strided_slice_41/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_3/values_0?
strided_slice_41/stack_3Pack*strided_slice_41/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_3~
strided_slice_41/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_41/stack_4?
strided_slice_41StridedSlicestrided_slice_1:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_3:output:0!strided_slice_41/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_41b
concat_19/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_19/axis?
	concat_19ConcatV2strided_slice_40:output:0strided_slice_41:output:0concat_19/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_19u
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_23/shape|

Reshape_23Reshapeconcat_19:output:0Reshape_23/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_23z
strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack~
strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_1~
strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_2?
strided_slice_42StridedSlicestrided_slice:output:0strided_slice_42/stack:output:0!strided_slice_42/stack_1:output:0!strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_42r
strided_slice_43/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_43/stack?
!strided_slice_43/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_43/stack_1/values_0?
strided_slice_43/stack_1Pack*strided_slice_43/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_1v
strided_slice_43/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_43/stack_2?
!strided_slice_43/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_43/stack_3/values_0?
strided_slice_43/stack_3Pack*strided_slice_43/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_3~
strided_slice_43/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_43/stack_4?
strided_slice_43StridedSlicestrided_slice_1:output:0!strided_slice_43/stack_1:output:0!strided_slice_43/stack_3:output:0!strided_slice_43/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_43b
concat_20/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_20/axis?
	concat_20ConcatV2strided_slice_42:output:0strided_slice_43:output:0concat_20/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_20u
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_24/shape|

Reshape_24Reshapeconcat_20:output:0Reshape_24/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_24z
strided_slice_44/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack~
strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack_1~
strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack_2?
strided_slice_44StridedSlicestrided_slice:output:0strided_slice_44/stack:output:0!strided_slice_44/stack_1:output:0!strided_slice_44/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_44r
strided_slice_45/stackConst*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_45/stack?
!strided_slice_45/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B : 2#
!strided_slice_45/stack_1/values_0?
strided_slice_45/stack_1Pack*strided_slice_45/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_45/stack_1v
strided_slice_45/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_45/stack_2?
!strided_slice_45/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_45/stack_3/values_0?
strided_slice_45/stack_3Pack*strided_slice_45/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_45/stack_3~
strided_slice_45/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_45/stack_4?
strided_slice_45StridedSlicestrided_slice_1:output:0!strided_slice_45/stack_1:output:0!strided_slice_45/stack_3:output:0!strided_slice_45/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_45b
concat_21/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_21/axis?
	concat_21ConcatV2strided_slice_44:output:0strided_slice_45:output:0concat_21/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_21u
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_25/shape|

Reshape_25Reshapeconcat_21:output:0Reshape_25/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_25b
concat_22/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_22/axis?
	concat_22ConcatV2Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0Reshape_16:output:0Reshape_17:output:0Reshape_18:output:0Reshape_19:output:0Reshape_20:output:0Reshape_21:output:0Reshape_22:output:0Reshape_23:output:0Reshape_24:output:0Reshape_25:output:0concat_22/axis:output:0*
N*
T0*
_output_shapes
:	?2
	concat_22y
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_26/shape?

Reshape_26Reshapeconcat_22:output:0Reshape_26/shape:output:0*
T0*#
_output_shapes
:?2

Reshape_26b
concat_23/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_23/axis?
	concat_23ConcatV2Reshape_25:output:0Reshape_24:output:0Reshape_23:output:0Reshape_22:output:0Reshape_21:output:0Reshape_20:output:0Reshape_19:output:0Reshape_18:output:0Reshape_17:output:0Reshape_16:output:0Reshape_15:output:0Reshape_14:output:0Reshape_13:output:0Reshape_12:output:0Reshape_11:output:0Reshape_10:output:0Reshape_9:output:0Reshape_8:output:0Reshape_7:output:0Reshape_6:output:0Reshape_5:output:0Reshape_4:output:0concat_23/axis:output:0*
N*
T0*
_output_shapes
:	?2
	concat_23y
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_27/shape?

Reshape_27Reshapeconcat_23:output:0Reshape_27/shape:output:0*
T0*#
_output_shapes
:?2

Reshape_27c
IdentityIdentityReshape_26:output:0*
T0*#
_output_shapes
:?2

Identityg

Identity_1IdentityReshape_27:output:0*
T0*#
_output_shapes
:?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*l
_input_shapes[
Y:?????????d:?????????d:~:~:::::?:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????d
 
_user_specified_nameinputs:!

_output_shapes	
:?
?N
?
1__inference_bi_lstm_model_layer_call_fn_122101872
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*8
Tout0
.2,*
_collective_manager_ids
 *?
_output_shapes?
?:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~:~**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_bi_lstm_model_layer_call_and_return_conditional_losses_1221017512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*
_output_shapes
:~2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_11?
Identity_12Identity!StatefulPartitionedCall:output:12^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_12?
Identity_13Identity!StatefulPartitionedCall:output:13^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_13?
Identity_14Identity!StatefulPartitionedCall:output:14^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_14?
Identity_15Identity!StatefulPartitionedCall:output:15^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_15?
Identity_16Identity!StatefulPartitionedCall:output:16^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_16?
Identity_17Identity!StatefulPartitionedCall:output:17^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_17?
Identity_18Identity!StatefulPartitionedCall:output:18^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_18?
Identity_19Identity!StatefulPartitionedCall:output:19^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_19?
Identity_20Identity!StatefulPartitionedCall:output:20^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_20?
Identity_21Identity!StatefulPartitionedCall:output:21^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_21?
Identity_22Identity!StatefulPartitionedCall:output:22^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_22?
Identity_23Identity!StatefulPartitionedCall:output:23^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_23?
Identity_24Identity!StatefulPartitionedCall:output:24^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_24?
Identity_25Identity!StatefulPartitionedCall:output:25^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_25?
Identity_26Identity!StatefulPartitionedCall:output:26^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_26?
Identity_27Identity!StatefulPartitionedCall:output:27^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_27?
Identity_28Identity!StatefulPartitionedCall:output:28^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_28?
Identity_29Identity!StatefulPartitionedCall:output:29^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_29?
Identity_30Identity!StatefulPartitionedCall:output:30^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_30?
Identity_31Identity!StatefulPartitionedCall:output:31^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_31?
Identity_32Identity!StatefulPartitionedCall:output:32^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_32?
Identity_33Identity!StatefulPartitionedCall:output:33^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_33?
Identity_34Identity!StatefulPartitionedCall:output:34^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_34?
Identity_35Identity!StatefulPartitionedCall:output:35^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_35?
Identity_36Identity!StatefulPartitionedCall:output:36^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_36?
Identity_37Identity!StatefulPartitionedCall:output:37^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_37?
Identity_38Identity!StatefulPartitionedCall:output:38^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_38?
Identity_39Identity!StatefulPartitionedCall:output:39^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_39?
Identity_40Identity!StatefulPartitionedCall:output:40^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_40?
Identity_41Identity!StatefulPartitionedCall:output:41^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_41?
Identity_42Identity!StatefulPartitionedCall:output:42^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_42?
Identity_43Identity!StatefulPartitionedCall:output:43^StatefulPartitionedCall*
T0*
_output_shapes
:~2
Identity_43"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_2Identity_2:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_3Identity_3:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_4Identity_4:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????d:?????????d:~:~:::::?:~:~:::::?22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????d
!
_user_specified_name	input_2:!

_output_shapes	
:?:!

_output_shapes	
:?
??
?
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_122102407
inputs_0
inputs_1
blocklstm_cs_prev
blocklstm_h_prev%
!blocklstm_readvariableop_resource'
#blocklstm_readvariableop_1_resource'
#blocklstm_readvariableop_2_resource'
#blocklstm_readvariableop_3_resource
blocklstm_b
identity

identity_1?s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   2
Reshape/shapel
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*"
_output_shapes
:d2	
Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      d   2
Reshape_1/shaper
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*"
_output_shapes
:d2
	Reshape_1p
BlockLSTM/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM/seq_len_max?
BlockLSTM/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM/ReadVariableOp?
BlockLSTM/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_1?
BlockLSTM/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_2?
BlockLSTM/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM/ReadVariableOp_3?
	BlockLSTM	BlockLSTMBlockLSTM/seq_len_max:output:0Reshape:output:0blocklstm_cs_prevblocklstm_h_prev BlockLSTM/ReadVariableOp:value:0"BlockLSTM/ReadVariableOp_1:value:0"BlockLSTM/ReadVariableOp_2:value:0"BlockLSTM/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
	BlockLSTMt
BlockLSTM_1/seq_len_maxConst*
_output_shapes
: *
dtype0	*
value	B	 R2
BlockLSTM_1/seq_len_max?
BlockLSTM_1/ReadVariableOpReadVariableOp!blocklstm_readvariableop_resource* 
_output_shapes
:
??*
dtype02
BlockLSTM_1/ReadVariableOp?
BlockLSTM_1/ReadVariableOp_1ReadVariableOp#blocklstm_readvariableop_1_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_1?
BlockLSTM_1/ReadVariableOp_2ReadVariableOp#blocklstm_readvariableop_2_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_2?
BlockLSTM_1/ReadVariableOp_3ReadVariableOp#blocklstm_readvariableop_3_resource*
_output_shapes
:~*
dtype02
BlockLSTM_1/ReadVariableOp_3?
BlockLSTM_1	BlockLSTM BlockLSTM_1/seq_len_max:output:0Reshape_1:output:0blocklstm_cs_prevblocklstm_h_prev"BlockLSTM_1/ReadVariableOp:value:0$BlockLSTM_1/ReadVariableOp_1:value:0$BlockLSTM_1/ReadVariableOp_2:value:0$BlockLSTM_1/ReadVariableOp_3:value:0blocklstm_b*
T0*v
_output_shapesd
b:~:~:~:~:~:~:~2
BlockLSTM_1w
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_2/shapew
	Reshape_2ReshapeBlockLSTM:h:0Reshape_2/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_2w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ~   2
Reshape_3/shapey
	Reshape_3ReshapeBlockLSTM_1:h:0Reshape_3/shape:output:0*
T0*"
_output_shapes
:~2
	Reshape_3t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceReshape_2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReshape_3:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:~*
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_2p
strided_slice_3/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack?
 strided_slice_3/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_1/values_0?
strided_slice_3/stack_1Pack)strided_slice_3/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_1t
strided_slice_3/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_3/stack_2?
 strided_slice_3/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_3/stack_3/values_0?
strided_slice_3/stack_3Pack)strided_slice_3/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_3|
strided_slice_3/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_4?
strided_slice_3StridedSlicestrided_slice_1:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_3:output:0 strided_slice_3/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:?2
concats
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_4/shapev
	Reshape_4Reshapeconcat:output:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_4p
strided_slice_5/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack?
 strided_slice_5/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_1/values_0?
strided_slice_5/stack_1Pack)strided_slice_5/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_1t
strided_slice_5/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_5/stack_2?
 strided_slice_5/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_5/stack_3/values_0?
strided_slice_5/stack_3Pack)strided_slice_5/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_3|
strided_slice_5/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_4?
strided_slice_5StridedSlicestrided_slice_1:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_3:output:0 strided_slice_5/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_5`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2strided_slice_4:output:0strided_slice_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_1s
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_5/shapex
	Reshape_5Reshapeconcat_1:output:0Reshape_5/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_6p
strided_slice_7/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack?
 strided_slice_7/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_1/values_0?
strided_slice_7/stack_1Pack)strided_slice_7/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_1t
strided_slice_7/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_7/stack_2?
 strided_slice_7/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_7/stack_3/values_0?
strided_slice_7/stack_3Pack)strided_slice_7/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_7/stack_3|
strided_slice_7/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_4?
strided_slice_7StridedSlicestrided_slice_1:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_3:output:0 strided_slice_7/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_7`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis?
concat_2ConcatV2strided_slice_6:output:0strided_slice_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_2s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_6/shapex
	Reshape_6Reshapeconcat_2:output:0Reshape_6/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_6x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_8p
strided_slice_9/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack?
 strided_slice_9/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_1/values_0?
strided_slice_9/stack_1Pack)strided_slice_9/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_1t
strided_slice_9/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_9/stack_2?
 strided_slice_9/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2"
 strided_slice_9/stack_3/values_0?
strided_slice_9/stack_3Pack)strided_slice_9/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_9/stack_3|
strided_slice_9/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_4?
strided_slice_9StridedSlicestrided_slice_1:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_3:output:0 strided_slice_9/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_9`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis?
concat_3ConcatV2strided_slice_8:output:0strided_slice_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_3s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_7/shapex
	Reshape_7Reshapeconcat_3:output:0Reshape_7/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_7z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_10r
strided_slice_11/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack?
!strided_slice_11/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_1/values_0?
strided_slice_11/stack_1Pack*strided_slice_11/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_1v
strided_slice_11/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_11/stack_2?
!strided_slice_11/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_11/stack_3/values_0?
strided_slice_11/stack_3Pack*strided_slice_11/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_11/stack_3~
strided_slice_11/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_4?
strided_slice_11StridedSlicestrided_slice_1:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_3:output:0!strided_slice_11/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_11`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis?
concat_4ConcatV2strided_slice_10:output:0strided_slice_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_4s
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_8/shapex
	Reshape_8Reshapeconcat_4:output:0Reshape_8/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_8z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_12r
strided_slice_13/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack?
!strided_slice_13/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_1/values_0?
strided_slice_13/stack_1Pack*strided_slice_13/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_1v
strided_slice_13/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_13/stack_2?
!strided_slice_13/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_13/stack_3/values_0?
strided_slice_13/stack_3Pack*strided_slice_13/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_13/stack_3~
strided_slice_13/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_4?
strided_slice_13StridedSlicestrided_slice_1:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_3:output:0!strided_slice_13/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_13`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis?
concat_5ConcatV2strided_slice_12:output:0strided_slice_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_5s
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_9/shapex
	Reshape_9Reshapeconcat_5:output:0Reshape_9/shape:output:0*
T0*
_output_shapes
:	?2
	Reshape_9z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_14r
strided_slice_15/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack?
!strided_slice_15/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_1/values_0?
strided_slice_15/stack_1Pack*strided_slice_15/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_1v
strided_slice_15/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_15/stack_2?
!strided_slice_15/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_15/stack_3/values_0?
strided_slice_15/stack_3Pack*strided_slice_15/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_15/stack_3~
strided_slice_15/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_4?
strided_slice_15StridedSlicestrided_slice_1:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_3:output:0!strided_slice_15/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_15`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_6/axis?
concat_6ConcatV2strided_slice_14:output:0strided_slice_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_6u
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_10/shape{

Reshape_10Reshapeconcat_6:output:0Reshape_10/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_10z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_16r
strided_slice_17/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack?
!strided_slice_17/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_1/values_0?
strided_slice_17/stack_1Pack*strided_slice_17/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_1v
strided_slice_17/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_17/stack_2?
!strided_slice_17/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_17/stack_3/values_0?
strided_slice_17/stack_3Pack*strided_slice_17/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_17/stack_3~
strided_slice_17/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_17/stack_4?
strided_slice_17StridedSlicestrided_slice_1:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_3:output:0!strided_slice_17/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_17`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_7/axis?
concat_7ConcatV2strided_slice_16:output:0strided_slice_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_7u
Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_11/shape{

Reshape_11Reshapeconcat_7:output:0Reshape_11/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_11z
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack~
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_18/stack_1~
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_18r
strided_slice_19/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack?
!strided_slice_19/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_1/values_0?
strided_slice_19/stack_1Pack*strided_slice_19/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_1v
strided_slice_19/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_19/stack_2?
!strided_slice_19/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_19/stack_3/values_0?
strided_slice_19/stack_3Pack*strided_slice_19/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_19/stack_3~
strided_slice_19/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_19/stack_4?
strided_slice_19StridedSlicestrided_slice_1:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_3:output:0!strided_slice_19/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_19`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_8/axis?
concat_8ConcatV2strided_slice_18:output:0strided_slice_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_8u
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_12/shape{

Reshape_12Reshapeconcat_8:output:0Reshape_12/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_12z
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_20/stack~
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_20/stack_1~
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_20r
strided_slice_21/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack?
!strided_slice_21/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_1/values_0?
strided_slice_21/stack_1Pack*strided_slice_21/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_1v
strided_slice_21/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_21/stack_2?
!strided_slice_21/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_21/stack_3/values_0?
strided_slice_21/stack_3Pack*strided_slice_21/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_21/stack_3~
strided_slice_21/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_21/stack_4?
strided_slice_21StridedSlicestrided_slice_1:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_3:output:0!strided_slice_21/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_21`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_9/axis?
concat_9ConcatV2strided_slice_20:output:0strided_slice_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes	
:?2

concat_9u
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_13/shape{

Reshape_13Reshapeconcat_9:output:0Reshape_13/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_13z
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_22/stack~
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_1~
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_22r
strided_slice_23/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_23/stack?
!strided_slice_23/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_23/stack_1/values_0?
strided_slice_23/stack_1Pack*strided_slice_23/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_1v
strided_slice_23/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_23/stack_2?
!strided_slice_23/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_23/stack_3/values_0?
strided_slice_23/stack_3Pack*strided_slice_23/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_23/stack_3~
strided_slice_23/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_23/stack_4?
strided_slice_23StridedSlicestrided_slice_1:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_3:output:0!strided_slice_23/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_23b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_10/axis?
	concat_10ConcatV2strided_slice_22:output:0strided_slice_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_10u
Reshape_14/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_14/shape|

Reshape_14Reshapeconcat_10:output:0Reshape_14/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_14z
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack~
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_1~
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_24r
strided_slice_25/stackConst*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_25/stack?
!strided_slice_25/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_25/stack_1/values_0?
strided_slice_25/stack_1Pack*strided_slice_25/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_1v
strided_slice_25/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_25/stack_2?
!strided_slice_25/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_25/stack_3/values_0?
strided_slice_25/stack_3Pack*strided_slice_25/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_25/stack_3~
strided_slice_25/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_25/stack_4?
strided_slice_25StridedSlicestrided_slice_1:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_3:output:0!strided_slice_25/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_25b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_11/axis?
	concat_11ConcatV2strided_slice_24:output:0strided_slice_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_11u
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_15/shape|

Reshape_15Reshapeconcat_11:output:0Reshape_15/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_15z
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack~
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_1~
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_26r
strided_slice_27/stackConst*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_27/stack?
!strided_slice_27/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_27/stack_1/values_0?
strided_slice_27/stack_1Pack*strided_slice_27/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_1v
strided_slice_27/stack_2Const*
_output_shapes
: *
dtype0*
value	B :
2
strided_slice_27/stack_2?
!strided_slice_27/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :
2#
!strided_slice_27/stack_3/values_0?
strided_slice_27/stack_3Pack*strided_slice_27/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_27/stack_3~
strided_slice_27/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_27/stack_4?
strided_slice_27StridedSlicestrided_slice_1:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_3:output:0!strided_slice_27/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_27b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_12/axis?
	concat_12ConcatV2strided_slice_26:output:0strided_slice_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_12u
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_16/shape|

Reshape_16Reshapeconcat_12:output:0Reshape_16/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_16z
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack~
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_1~
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_28/stack_2?
strided_slice_28StridedSlicestrided_slice:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_28r
strided_slice_29/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_29/stack?
!strided_slice_29/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_29/stack_1/values_0?
strided_slice_29/stack_1Pack*strided_slice_29/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_1v
strided_slice_29/stack_2Const*
_output_shapes
: *
dtype0*
value	B :	2
strided_slice_29/stack_2?
!strided_slice_29/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :	2#
!strided_slice_29/stack_3/values_0?
strided_slice_29/stack_3Pack*strided_slice_29/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_29/stack_3~
strided_slice_29/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_29/stack_4?
strided_slice_29StridedSlicestrided_slice_1:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_3:output:0!strided_slice_29/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_29b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_13/axis?
	concat_13ConcatV2strided_slice_28:output:0strided_slice_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_13u
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_17/shape|

Reshape_17Reshapeconcat_13:output:0Reshape_17/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_17z
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack~
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_1~
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_30/stack_2?
strided_slice_30StridedSlicestrided_slice:output:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_30r
strided_slice_31/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack?
!strided_slice_31/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_1/values_0?
strided_slice_31/stack_1Pack*strided_slice_31/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_1v
strided_slice_31/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_31/stack_2?
!strided_slice_31/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_31/stack_3/values_0?
strided_slice_31/stack_3Pack*strided_slice_31/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_31/stack_3~
strided_slice_31/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_31/stack_4?
strided_slice_31StridedSlicestrided_slice_1:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_3:output:0!strided_slice_31/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_31b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_14/axis?
	concat_14ConcatV2strided_slice_30:output:0strided_slice_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_14u
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_18/shape|

Reshape_18Reshapeconcat_14:output:0Reshape_18/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_18z
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack~
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_1~
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_32/stack_2?
strided_slice_32StridedSlicestrided_slice:output:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_32r
strided_slice_33/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack?
!strided_slice_33/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_1/values_0?
strided_slice_33/stack_1Pack*strided_slice_33/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_1v
strided_slice_33/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_33/stack_2?
!strided_slice_33/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_33/stack_3/values_0?
strided_slice_33/stack_3Pack*strided_slice_33/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_33/stack_3~
strided_slice_33/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_33/stack_4?
strided_slice_33StridedSlicestrided_slice_1:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_3:output:0!strided_slice_33/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_33b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_15/axis?
	concat_15ConcatV2strided_slice_32:output:0strided_slice_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_15u
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_19/shape|

Reshape_19Reshapeconcat_15:output:0Reshape_19/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_19z
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack~
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_1~
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_34/stack_2?
strided_slice_34StridedSlicestrided_slice:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_34r
strided_slice_35/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack?
!strided_slice_35/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_1/values_0?
strided_slice_35/stack_1Pack*strided_slice_35/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_1v
strided_slice_35/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_35/stack_2?
!strided_slice_35/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_35/stack_3/values_0?
strided_slice_35/stack_3Pack*strided_slice_35/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_35/stack_3~
strided_slice_35/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_35/stack_4?
strided_slice_35StridedSlicestrided_slice_1:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_3:output:0!strided_slice_35/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_35b
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_16/axis?
	concat_16ConcatV2strided_slice_34:output:0strided_slice_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_16u
Reshape_20/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_20/shape|

Reshape_20Reshapeconcat_16:output:0Reshape_20/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_20z
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack~
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_1~
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_36/stack_2?
strided_slice_36StridedSlicestrided_slice:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_36r
strided_slice_37/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack?
!strided_slice_37/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_1/values_0?
strided_slice_37/stack_1Pack*strided_slice_37/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_1v
strided_slice_37/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_37/stack_2?
!strided_slice_37/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_37/stack_3/values_0?
strided_slice_37/stack_3Pack*strided_slice_37/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_37/stack_3~
strided_slice_37/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_37/stack_4?
strided_slice_37StridedSlicestrided_slice_1:output:0!strided_slice_37/stack_1:output:0!strided_slice_37/stack_3:output:0!strided_slice_37/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_37b
concat_17/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_17/axis?
	concat_17ConcatV2strided_slice_36:output:0strided_slice_37:output:0concat_17/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_17u
Reshape_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_21/shape|

Reshape_21Reshapeconcat_17:output:0Reshape_21/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_21z
strided_slice_38/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack~
strided_slice_38/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_1~
strided_slice_38/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_38/stack_2?
strided_slice_38StridedSlicestrided_slice:output:0strided_slice_38/stack:output:0!strided_slice_38/stack_1:output:0!strided_slice_38/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_38r
strided_slice_39/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack?
!strided_slice_39/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_1/values_0?
strided_slice_39/stack_1Pack*strided_slice_39/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_1v
strided_slice_39/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_39/stack_2?
!strided_slice_39/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_39/stack_3/values_0?
strided_slice_39/stack_3Pack*strided_slice_39/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_39/stack_3~
strided_slice_39/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_39/stack_4?
strided_slice_39StridedSlicestrided_slice_1:output:0!strided_slice_39/stack_1:output:0!strided_slice_39/stack_3:output:0!strided_slice_39/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_39b
concat_18/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_18/axis?
	concat_18ConcatV2strided_slice_38:output:0strided_slice_39:output:0concat_18/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_18u
Reshape_22/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_22/shape|

Reshape_22Reshapeconcat_18:output:0Reshape_22/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_22z
strided_slice_40/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack~
strided_slice_40/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_1~
strided_slice_40/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_40/stack_2?
strided_slice_40StridedSlicestrided_slice:output:0strided_slice_40/stack:output:0!strided_slice_40/stack_1:output:0!strided_slice_40/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_40r
strided_slice_41/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack?
!strided_slice_41/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_1/values_0?
strided_slice_41/stack_1Pack*strided_slice_41/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_1v
strided_slice_41/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_41/stack_2?
!strided_slice_41/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_41/stack_3/values_0?
strided_slice_41/stack_3Pack*strided_slice_41/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_41/stack_3~
strided_slice_41/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_41/stack_4?
strided_slice_41StridedSlicestrided_slice_1:output:0!strided_slice_41/stack_1:output:0!strided_slice_41/stack_3:output:0!strided_slice_41/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_41b
concat_19/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_19/axis?
	concat_19ConcatV2strided_slice_40:output:0strided_slice_41:output:0concat_19/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_19u
Reshape_23/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_23/shape|

Reshape_23Reshapeconcat_19:output:0Reshape_23/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_23z
strided_slice_42/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack~
strided_slice_42/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_1~
strided_slice_42/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_42/stack_2?
strided_slice_42StridedSlicestrided_slice:output:0strided_slice_42/stack:output:0!strided_slice_42/stack_1:output:0!strided_slice_42/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_42r
strided_slice_43/stackConst*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_43/stack?
!strided_slice_43/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_43/stack_1/values_0?
strided_slice_43/stack_1Pack*strided_slice_43/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_1v
strided_slice_43/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_43/stack_2?
!strided_slice_43/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_43/stack_3/values_0?
strided_slice_43/stack_3Pack*strided_slice_43/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_43/stack_3~
strided_slice_43/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_43/stack_4?
strided_slice_43StridedSlicestrided_slice_1:output:0!strided_slice_43/stack_1:output:0!strided_slice_43/stack_3:output:0!strided_slice_43/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_43b
concat_20/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_20/axis?
	concat_20ConcatV2strided_slice_42:output:0strided_slice_43:output:0concat_20/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_20u
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_24/shape|

Reshape_24Reshapeconcat_20:output:0Reshape_24/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_24z
strided_slice_44/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack~
strided_slice_44/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack_1~
strided_slice_44/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_44/stack_2?
strided_slice_44StridedSlicestrided_slice:output:0strided_slice_44/stack:output:0!strided_slice_44/stack_1:output:0!strided_slice_44/stack_2:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_44r
strided_slice_45/stackConst*
_output_shapes
: *
dtype0*
value	B : 2
strided_slice_45/stack?
!strided_slice_45/stack_1/values_0Const*
_output_shapes
: *
dtype0*
value	B : 2#
!strided_slice_45/stack_1/values_0?
strided_slice_45/stack_1Pack*strided_slice_45/stack_1/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_45/stack_1v
strided_slice_45/stack_2Const*
_output_shapes
: *
dtype0*
value	B :2
strided_slice_45/stack_2?
!strided_slice_45/stack_3/values_0Const*
_output_shapes
: *
dtype0*
value	B :2#
!strided_slice_45/stack_3/values_0?
strided_slice_45/stack_3Pack*strided_slice_45/stack_3/values_0:output:0*
N*
T0*
_output_shapes
:2
strided_slice_45/stack_3~
strided_slice_45/stack_4Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_45/stack_4?
strided_slice_45StridedSlicestrided_slice_1:output:0!strided_slice_45/stack_1:output:0!strided_slice_45/stack_3:output:0!strided_slice_45/stack_4:output:0*
Index0*
T0*
_output_shapes
:~*
shrink_axis_mask2
strided_slice_45b
concat_21/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_21/axis?
	concat_21ConcatV2strided_slice_44:output:0strided_slice_45:output:0concat_21/axis:output:0*
N*
T0*
_output_shapes	
:?2
	concat_21u
Reshape_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ?   2
Reshape_25/shape|

Reshape_25Reshapeconcat_21:output:0Reshape_25/shape:output:0*
T0*
_output_shapes
:	?2

Reshape_25b
concat_22/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_22/axis?
	concat_22ConcatV2Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0Reshape_16:output:0Reshape_17:output:0Reshape_18:output:0Reshape_19:output:0Reshape_20:output:0Reshape_21:output:0Reshape_22:output:0Reshape_23:output:0Reshape_24:output:0Reshape_25:output:0concat_22/axis:output:0*
N*
T0*
_output_shapes
:	?2
	concat_22y
Reshape_26/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_26/shape?

Reshape_26Reshapeconcat_22:output:0Reshape_26/shape:output:0*
T0*#
_output_shapes
:?2

Reshape_26b
concat_23/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_23/axis?
	concat_23ConcatV2Reshape_25:output:0Reshape_24:output:0Reshape_23:output:0Reshape_22:output:0Reshape_21:output:0Reshape_20:output:0Reshape_19:output:0Reshape_18:output:0Reshape_17:output:0Reshape_16:output:0Reshape_15:output:0Reshape_14:output:0Reshape_13:output:0Reshape_12:output:0Reshape_11:output:0Reshape_10:output:0Reshape_9:output:0Reshape_8:output:0Reshape_7:output:0Reshape_6:output:0Reshape_5:output:0Reshape_4:output:0concat_23/axis:output:0*
N*
T0*
_output_shapes
:	?2
	concat_23y
Reshape_27/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Reshape_27/shape?

Reshape_27Reshapeconcat_23:output:0Reshape_27/shape:output:0*
T0*#
_output_shapes
:?2

Reshape_27c
IdentityIdentityReshape_26:output:0*
T0*#
_output_shapes
:?2

Identityg

Identity_1IdentityReshape_27:output:0*
T0*#
_output_shapes
:?2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*l
_input_shapes[
Y:?????????d:?????????d:~:~:::::?:U Q
+
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????d
"
_user_specified_name
inputs/1:!

_output_shapes	
:?
?

?
1__inference_NextBlockLSTM_layer_call_fn_122102496
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:~:~*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_1221013992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:~2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*"
_output_shapes
:~2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*\
_input_shapesK
I:?:?:~:~:::::?22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?
"
_user_specified_name
inputs/1:!

_output_shapes	
:?"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????d
?
input_24
serving_default_input_2:0?????????d2
output_10_1#
StatefulPartitionedCall:0~2
output_10_2#
StatefulPartitionedCall:1~2
output_11_1#
StatefulPartitionedCall:2~2
output_11_2#
StatefulPartitionedCall:3~2
output_12_1#
StatefulPartitionedCall:4~2
output_12_2#
StatefulPartitionedCall:5~2
output_13_1#
StatefulPartitionedCall:6~2
output_13_2#
StatefulPartitionedCall:7~2
output_14_1#
StatefulPartitionedCall:8~2
output_14_2#
StatefulPartitionedCall:9~3
output_15_1$
StatefulPartitionedCall:10~3
output_15_2$
StatefulPartitionedCall:11~3
output_16_1$
StatefulPartitionedCall:12~3
output_16_2$
StatefulPartitionedCall:13~3
output_17_1$
StatefulPartitionedCall:14~3
output_17_2$
StatefulPartitionedCall:15~3
output_18_1$
StatefulPartitionedCall:16~3
output_18_2$
StatefulPartitionedCall:17~3
output_19_1$
StatefulPartitionedCall:18~3
output_19_2$
StatefulPartitionedCall:19~2

output_1_1$
StatefulPartitionedCall:20~2

output_1_2$
StatefulPartitionedCall:21~3
output_20_1$
StatefulPartitionedCall:22~3
output_20_2$
StatefulPartitionedCall:23~3
output_21_1$
StatefulPartitionedCall:24~3
output_21_2$
StatefulPartitionedCall:25~3
output_22_1$
StatefulPartitionedCall:26~3
output_22_2$
StatefulPartitionedCall:27~2

output_2_1$
StatefulPartitionedCall:28~2

output_2_2$
StatefulPartitionedCall:29~2

output_3_1$
StatefulPartitionedCall:30~2

output_3_2$
StatefulPartitionedCall:31~2

output_4_1$
StatefulPartitionedCall:32~2

output_4_2$
StatefulPartitionedCall:33~2

output_5_1$
StatefulPartitionedCall:34~2

output_5_2$
StatefulPartitionedCall:35~2

output_6_1$
StatefulPartitionedCall:36~2

output_6_2$
StatefulPartitionedCall:37~2

output_7_1$
StatefulPartitionedCall:38~2

output_7_2$
StatefulPartitionedCall:39~2

output_8_1$
StatefulPartitionedCall:40~2

output_8_2$
StatefulPartitionedCall:41~2

output_9_1$
StatefulPartitionedCall:42~2

output_9_2$
StatefulPartitionedCall:43~tensorflow/serving/predict:?j
?
	blockLstm
nextBlockLstm
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*'&call_and_return_all_conditional_losses
(__call__
)_default_save_signature"?
_tf_keras_model?{"class_name": "BiLSTMModel", "name": "bi_lstm_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BiLSTMModel"}}
?
weight_matrix
	weight_input_gate

weight_forget_gate
weight_output_gate
	variables
regularization_losses
trainable_variables
	keras_api
**&call_and_return_all_conditional_losses
+__call__"?
_tf_keras_layer?{"class_name": "FirstBlockLSTMModule", "name": "FirstBlockLSTMModule", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 22, 100]}, {"class_name": "TensorShape", "items": [1, 22, 100]}]}
?
weight_matrix
weight_input_gate
weight_forget_gate
weight_output_gate
	variables
regularization_losses
trainable_variables
	keras_api
*,&call_and_return_all_conditional_losses
-__call__"?
_tf_keras_layer?{"class_name": "NextBlockLSTM", "name": "NextBlockLSTM", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 22, 252]}, {"class_name": "TensorShape", "items": [1, 22, 252]}]}
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
?
layer_regularization_losses
	variables
metrics

layers
regularization_losses
trainable_variables
non_trainable_variables
layer_metrics
(__call__
)_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
,
.serving_default"
signature_map
C:A
??2/bi_lstm_model/FirstBlockLSTMModule/w_first_lstm
?:=~21bi_lstm_model/FirstBlockLSTMModule/wig_first_lstm
?:=~21bi_lstm_model/FirstBlockLSTMModule/wfg_first_lstm
?:=~21bi_lstm_model/FirstBlockLSTMModule/wog_first_lstm
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
?
layer_regularization_losses
	variables
metrics

layers
regularization_losses
trainable_variables
 non_trainable_variables
!layer_metrics
+__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
;:9
??2'bi_lstm_model/NextBlockLSTM/w_next_lstm
7:5~2)bi_lstm_model/NextBlockLSTM/wig_next_lstm
7:5~2)bi_lstm_model/NextBlockLSTM/wfg_next_lstm
7:5~2)bi_lstm_model/NextBlockLSTM/wog_next_lstm
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
"layer_regularization_losses
	variables
#metrics

$layers
regularization_losses
trainable_variables
%non_trainable_variables
&layer_metrics
-__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
L__inference_bi_lstm_model_layer_call_and_return_conditional_losses_122101751?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
?2?
1__inference_bi_lstm_model_layer_call_fn_122101872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
?2?
$__inference__wrapped_model_122100896?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
?2?
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_122102407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_FirstBlockLSTMModule_layer_call_fn_122102429?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_122102474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_NextBlockLSTM_layer_call_fn_122102496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
=B;
'__inference_signature_wrapper_122101994input_1input_2
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5?
S__inference_FirstBlockLSTMModule_layer_call_and_return_conditional_losses_122102407?/0	
1b?_
X?U
S?P
&?#
inputs/0?????????d
&?#
inputs/1?????????d
? "C?@
9?6
?
0/0?
?
0/1?
? ?
8__inference_FirstBlockLSTMModule_layer_call_fn_122102429?/0	
1b?_
X?U
S?P
&?#
inputs/0?????????d
&?#
inputs/1?????????d
? "5?2
?
0?
?
1??
L__inference_NextBlockLSTM_layer_call_and_return_conditional_losses_122102474?234R?O
H?E
C?@
?
inputs/0?
?
inputs/1?
? "A?>
7?4
?
0/0~
?
0/1~
? ?
1__inference_NextBlockLSTM_layer_call_fn_122102496?234R?O
H?E
C?@
?
inputs/0?
?
inputs/1?
? "3?0
?
0~
?
1~?
$__inference__wrapped_model_122100896?/0	
1234`?]
V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
? "???
'
output_10_1?
output_10_1~
'
output_10_2?
output_10_2~
'
output_11_1?
output_11_1~
'
output_11_2?
output_11_2~
'
output_12_1?
output_12_1~
'
output_12_2?
output_12_2~
'
output_13_1?
output_13_1~
'
output_13_2?
output_13_2~
'
output_14_1?
output_14_1~
'
output_14_2?
output_14_2~
'
output_15_1?
output_15_1~
'
output_15_2?
output_15_2~
'
output_16_1?
output_16_1~
'
output_16_2?
output_16_2~
'
output_17_1?
output_17_1~
'
output_17_2?
output_17_2~
'
output_18_1?
output_18_1~
'
output_18_2?
output_18_2~
'
output_19_1?
output_19_1~
'
output_19_2?
output_19_2~
%

output_1_1?

output_1_1~
%

output_1_2?

output_1_2~
'
output_20_1?
output_20_1~
'
output_20_2?
output_20_2~
'
output_21_1?
output_21_1~
'
output_21_2?
output_21_2~
'
output_22_1?
output_22_1~
'
output_22_2?
output_22_2~
%

output_2_1?

output_2_1~
%

output_2_2?

output_2_2~
%

output_3_1?

output_3_1~
%

output_3_2?

output_3_2~
%

output_4_1?

output_4_1~
%

output_4_2?

output_4_2~
%

output_5_1?

output_5_1~
%

output_5_2?

output_5_2~
%

output_6_1?

output_6_1~
%

output_6_2?

output_6_2~
%

output_7_1?

output_7_1~
%

output_7_2?

output_7_2~
%

output_8_1?

output_8_1~
%

output_8_2?

output_8_2~
%

output_9_1?

output_9_1~
%

output_9_2?

output_9_2~?	
L__inference_bi_lstm_model_layer_call_and_return_conditional_losses_122101751?/0	
1234`?]
V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
? "???
???
+?(
?
0/0/0~
?
0/0/1~
+?(
?
0/1/0~
?
0/1/1~
+?(
?
0/2/0~
?
0/2/1~
+?(
?
0/3/0~
?
0/3/1~
+?(
?
0/4/0~
?
0/4/1~
+?(
?
0/5/0~
?
0/5/1~
+?(
?
0/6/0~
?
0/6/1~
+?(
?
0/7/0~
?
0/7/1~
+?(
?
0/8/0~
?
0/8/1~
+?(
?
0/9/0~
?
0/9/1~
-?*
?
0/10/0~
?
0/10/1~
-?*
?
0/11/0~
?
0/11/1~
-?*
?
0/12/0~
?
0/12/1~
-?*
?
0/13/0~
?
0/13/1~
-?*
?
0/14/0~
?
0/14/1~
-?*
?
0/15/0~
?
0/15/1~
-?*
?
0/16/0~
?
0/16/1~
-?*
?
0/17/0~
?
0/17/1~
-?*
?
0/18/0~
?
0/18/1~
-?*
?
0/19/0~
?
0/19/1~
-?*
?
0/20/0~
?
0/20/1~
-?*
?
0/21/0~
?
0/21/1~
? ?
1__inference_bi_lstm_model_layer_call_fn_122101872?/0	
1234`?]
V?S
Q?N
%?"
input_1?????????d
%?"
input_2?????????d
? "???
'?$
?
0/0~
?
0/1~
'?$
?
1/0~
?
1/1~
'?$
?
2/0~
?
2/1~
'?$
?
3/0~
?
3/1~
'?$
?
4/0~
?
4/1~
'?$
?
5/0~
?
5/1~
'?$
?
6/0~
?
6/1~
'?$
?
7/0~
?
7/1~
'?$
?
8/0~
?
8/1~
'?$
?
9/0~
?
9/1~
)?&
?
10/0~
?
10/1~
)?&
?
11/0~
?
11/1~
)?&
?
12/0~
?
12/1~
)?&
?
13/0~
?
13/1~
)?&
?
14/0~
?
14/1~
)?&
?
15/0~
?
15/1~
)?&
?
16/0~
?
16/1~
)?&
?
17/0~
?
17/1~
)?&
?
18/0~
?
18/1~
)?&
?
19/0~
?
19/1~
)?&
?
20/0~
?
20/1~
)?&
?
21/0~
?
21/1~?
'__inference_signature_wrapper_122101994?/0	
1234q?n
? 
g?d
0
input_1%?"
input_1?????????d
0
input_2%?"
input_2?????????d"???
'
output_10_1?
output_10_1~
'
output_10_2?
output_10_2~
'
output_11_1?
output_11_1~
'
output_11_2?
output_11_2~
'
output_12_1?
output_12_1~
'
output_12_2?
output_12_2~
'
output_13_1?
output_13_1~
'
output_13_2?
output_13_2~
'
output_14_1?
output_14_1~
'
output_14_2?
output_14_2~
'
output_15_1?
output_15_1~
'
output_15_2?
output_15_2~
'
output_16_1?
output_16_1~
'
output_16_2?
output_16_2~
'
output_17_1?
output_17_1~
'
output_17_2?
output_17_2~
'
output_18_1?
output_18_1~
'
output_18_2?
output_18_2~
'
output_19_1?
output_19_1~
'
output_19_2?
output_19_2~
%

output_1_1?

output_1_1~
%

output_1_2?

output_1_2~
'
output_20_1?
output_20_1~
'
output_20_2?
output_20_2~
'
output_21_1?
output_21_1~
'
output_21_2?
output_21_2~
'
output_22_1?
output_22_1~
'
output_22_2?
output_22_2~
%

output_2_1?

output_2_1~
%

output_2_2?

output_2_2~
%

output_3_1?

output_3_1~
%

output_3_2?

output_3_2~
%

output_4_1?

output_4_1~
%

output_4_2?

output_4_2~
%

output_5_1?

output_5_1~
%

output_5_2?

output_5_2~
%

output_6_1?

output_6_1~
%

output_6_2?

output_6_2~
%

output_7_1?

output_7_1~
%

output_7_2?

output_7_2~
%

output_8_1?

output_8_1~
%

output_8_2?

output_8_2~
%

output_9_1?

output_9_1~
%

output_9_2?

output_9_2~