??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.8.02v1.8.0-0-g93bc2e2072??	

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
?
global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
v
PlaceholderPlaceholder* 
shape:?????????6*
dtype0*+
_output_shapes
:?????????6
h
strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
j
strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
j
strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
?
strided_sliceStridedSlicePlaceholderstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:?????????6*
T0*
Index0
O
one_hot/ConstConst*
dtype0*
_output_shapes
: *
value	B :
Q
one_hot/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
R
one_hot/on_valueConst*
value	B :*
dtype0*
_output_shapes
: 
S
one_hot/off_valueConst*
value	B : *
dtype0*
_output_shapes
: 
?
one_hotOneHotstrided_sliceone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
axis?????????*
TI0*+
_output_shapes
:?????????6
^
Reshape/shapeConst*
_output_shapes
:*
valueB"????D  *
dtype0
k
ReshapeReshapeone_hotReshape/shape*
T0*
Tshape0*(
_output_shapes
:??????????
W
CastCastReshape*

SrcT0*(
_output_shapes
:??????????*

DstT0
?
6sequential/linear/w/Initializer/truncated_normal/shapeConst*&
_class
loc:@sequential/linear/w*
valueB"D  ?  *
dtype0*
_output_shapes
:
?
5sequential/linear/w/Initializer/truncated_normal/meanConst*&
_class
loc:@sequential/linear/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/linear/w/Initializer/truncated_normal/stddevConst*&
_class
loc:@sequential/linear/w*
valueB
 *9?c=*
dtype0*
_output_shapes
: 
?
@sequential/linear/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6sequential/linear/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??'*

seed *
T0*&
_class
loc:@sequential/linear/w*
seed2 
?
4sequential/linear/w/Initializer/truncated_normal/mulMul@sequential/linear/w/Initializer/truncated_normal/TruncatedNormal7sequential/linear/w/Initializer/truncated_normal/stddev* 
_output_shapes
:
??'*
T0*&
_class
loc:@sequential/linear/w
?
0sequential/linear/w/Initializer/truncated_normalAdd4sequential/linear/w/Initializer/truncated_normal/mul5sequential/linear/w/Initializer/truncated_normal/mean* 
_output_shapes
:
??'*
T0*&
_class
loc:@sequential/linear/w
?
sequential/linear/w
VariableV2*
shared_name *&
_class
loc:@sequential/linear/w*
	container *
shape:
??'*
dtype0* 
_output_shapes
:
??'
?
sequential/linear/w/AssignAssignsequential/linear/w0sequential/linear/w/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@sequential/linear/w*
validate_shape(* 
_output_shapes
:
??'
?
sequential/linear/w/readIdentitysequential/linear/w*&
_class
loc:@sequential/linear/w* 
_output_shapes
:
??'*
T0
?
#sequential/sequential/linear/MatMulMatMulCastsequential/linear/w/read*
T0*(
_output_shapes
:??????????'*
transpose_a( *
transpose_b( 
?
5sequential/linear/b/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@sequential/linear/b*
valueB:?'*
dtype0*
_output_shapes
:
?
+sequential/linear/b/Initializer/zeros/ConstConst*&
_class
loc:@sequential/linear/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
%sequential/linear/b/Initializer/zerosFill5sequential/linear/b/Initializer/zeros/shape_as_tensor+sequential/linear/b/Initializer/zeros/Const*
T0*&
_class
loc:@sequential/linear/b*

index_type0*
_output_shapes	
:?'
?
sequential/linear/b
VariableV2*
shape:?'*
dtype0*
_output_shapes	
:?'*
shared_name *&
_class
loc:@sequential/linear/b*
	container 
?
sequential/linear/b/AssignAssignsequential/linear/b%sequential/linear/b/Initializer/zeros*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0*&
_class
loc:@sequential/linear/b
?
sequential/linear/b/readIdentitysequential/linear/b*
T0*&
_class
loc:@sequential/linear/b*
_output_shapes	
:?'
?
 sequential/sequential/linear/addAdd#sequential/sequential/linear/MatMulsequential/linear/b/read*
T0*(
_output_shapes
:??????????'
?
Esequential/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*7
_class-
+)loc:@sequential/batch_normalization/gamma*
valueB:?'*
dtype0*
_output_shapes
:
?
;sequential/batch_normalization/gamma/Initializer/ones/ConstConst*
_output_shapes
: *7
_class-
+)loc:@sequential/batch_normalization/gamma*
valueB
 *  ??*
dtype0
?
5sequential/batch_normalization/gamma/Initializer/onesFillEsequential/batch_normalization/gamma/Initializer/ones/shape_as_tensor;sequential/batch_normalization/gamma/Initializer/ones/Const*
_output_shapes	
:?'*
T0*7
_class-
+)loc:@sequential/batch_normalization/gamma*

index_type0
?
$sequential/batch_normalization/gamma
VariableV2*
shape:?'*
dtype0*
_output_shapes	
:?'*
shared_name *7
_class-
+)loc:@sequential/batch_normalization/gamma*
	container 
?
+sequential/batch_normalization/gamma/AssignAssign$sequential/batch_normalization/gamma5sequential/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*7
_class-
+)loc:@sequential/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:?'
?
)sequential/batch_normalization/gamma/readIdentity$sequential/batch_normalization/gamma*
T0*7
_class-
+)loc:@sequential/batch_normalization/gamma*
_output_shapes	
:?'
?
Esequential/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*6
_class,
*(loc:@sequential/batch_normalization/beta*
valueB:?'*
dtype0*
_output_shapes
:
?
;sequential/batch_normalization/beta/Initializer/zeros/ConstConst*
_output_shapes
: *6
_class,
*(loc:@sequential/batch_normalization/beta*
valueB
 *    *
dtype0
?
5sequential/batch_normalization/beta/Initializer/zerosFillEsequential/batch_normalization/beta/Initializer/zeros/shape_as_tensor;sequential/batch_normalization/beta/Initializer/zeros/Const*
T0*6
_class,
*(loc:@sequential/batch_normalization/beta*

index_type0*
_output_shapes	
:?'
?
#sequential/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:?'*
shared_name *6
_class,
*(loc:@sequential/batch_normalization/beta*
	container *
shape:?'
?
*sequential/batch_normalization/beta/AssignAssign#sequential/batch_normalization/beta5sequential/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@sequential/batch_normalization/beta*
validate_shape(*
_output_shapes	
:?'
?
(sequential/batch_normalization/beta/readIdentity#sequential/batch_normalization/beta*
T0*6
_class,
*(loc:@sequential/batch_normalization/beta*
_output_shapes	
:?'
?
Lsequential/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*=
_class3
1/loc:@sequential/batch_normalization/moving_mean*
valueB:?'
?
Bsequential/batch_normalization/moving_mean/Initializer/zeros/ConstConst*=
_class3
1/loc:@sequential/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
<sequential/batch_normalization/moving_mean/Initializer/zerosFillLsequential/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorBsequential/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*=
_class3
1/loc:@sequential/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:?'
?
*sequential/batch_normalization/moving_mean
VariableV2*
shape:?'*
dtype0*
_output_shapes	
:?'*
shared_name *=
_class3
1/loc:@sequential/batch_normalization/moving_mean*
	container 
?
1sequential/batch_normalization/moving_mean/AssignAssign*sequential/batch_normalization/moving_mean<sequential/batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0*=
_class3
1/loc:@sequential/batch_normalization/moving_mean
?
/sequential/batch_normalization/moving_mean/readIdentity*sequential/batch_normalization/moving_mean*
T0*=
_class3
1/loc:@sequential/batch_normalization/moving_mean*
_output_shapes	
:?'
?
Osequential/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
valueB:?'*
dtype0*
_output_shapes
:
?
Esequential/batch_normalization/moving_variance/Initializer/ones/ConstConst*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
?sequential/batch_normalization/moving_variance/Initializer/onesFillOsequential/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorEsequential/batch_normalization/moving_variance/Initializer/ones/Const*
T0*A
_class7
53loc:@sequential/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:?'
?
.sequential/batch_normalization/moving_variance
VariableV2*
_output_shapes	
:?'*
shared_name *A
_class7
53loc:@sequential/batch_normalization/moving_variance*
	container *
shape:?'*
dtype0
?
5sequential/batch_normalization/moving_variance/AssignAssign.sequential/batch_normalization/moving_variance?sequential/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:?'
?
3sequential/batch_normalization/moving_variance/readIdentity.sequential/batch_normalization/moving_variance*
T0*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
_output_shapes	
:?'
s
.sequential/batch_normalization/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
,sequential/batch_normalization/batchnorm/addAdd3sequential/batch_normalization/moving_variance/read.sequential/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:?'
?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt,sequential/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:?'
?
,sequential/batch_normalization/batchnorm/mulMul.sequential/batch_normalization/batchnorm/Rsqrt)sequential/batch_normalization/gamma/read*
T0*
_output_shapes	
:?'
?
.sequential/batch_normalization/batchnorm/mul_1Mul sequential/sequential/linear/add,sequential/batch_normalization/batchnorm/mul*(
_output_shapes
:??????????'*
T0
?
.sequential/batch_normalization/batchnorm/mul_2Mul/sequential/batch_normalization/moving_mean/read,sequential/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:?'
?
,sequential/batch_normalization/batchnorm/subSub(sequential/batch_normalization/beta/read.sequential/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:?'
?
.sequential/batch_normalization/batchnorm/add_1Add.sequential/batch_normalization/batchnorm/mul_1,sequential/batch_normalization/batchnorm/sub*(
_output_shapes
:??????????'*
T0
?
sequential/sequential_1/ReluRelu.sequential/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:??????????'
?
8sequential/linear_1/w/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@sequential/linear_1/w*
valueB"?  ?  
?
7sequential/linear_1/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_1/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_1/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_1/w*
valueB
 *j?g<*
dtype0*
_output_shapes
: 
?
Bsequential/linear_1/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_1/w/Initializer/truncated_normal/shape*

seed *
T0*(
_class
loc:@sequential/linear_1/w*
seed2 *
dtype0* 
_output_shapes
:
?'?
?
6sequential/linear_1/w/Initializer/truncated_normal/mulMulBsequential/linear_1/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_1/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_1/w* 
_output_shapes
:
?'?
?
2sequential/linear_1/w/Initializer/truncated_normalAdd6sequential/linear_1/w/Initializer/truncated_normal/mul7sequential/linear_1/w/Initializer/truncated_normal/mean* 
_output_shapes
:
?'?*
T0*(
_class
loc:@sequential/linear_1/w
?
sequential/linear_1/w
VariableV2*
dtype0* 
_output_shapes
:
?'?*
shared_name *(
_class
loc:@sequential/linear_1/w*
	container *
shape:
?'?
?
sequential/linear_1/w/AssignAssignsequential/linear_1/w2sequential/linear_1/w/Initializer/truncated_normal*
use_locking(*
T0*(
_class
loc:@sequential/linear_1/w*
validate_shape(* 
_output_shapes
:
?'?
?
sequential/linear_1/w/readIdentitysequential/linear_1/w*
T0*(
_class
loc:@sequential/linear_1/w* 
_output_shapes
:
?'?
?
'sequential/sequential_2/linear_1/MatMulMatMulsequential/sequential_1/Relusequential/linear_1/w/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( *
T0
?
7sequential/linear_1/b/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*(
_class
loc:@sequential/linear_1/b*
valueB:?*
dtype0
?
-sequential/linear_1/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_1/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_1/b/Initializer/zerosFill7sequential/linear_1/b/Initializer/zeros/shape_as_tensor-sequential/linear_1/b/Initializer/zeros/Const*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_1/b*

index_type0
?
sequential/linear_1/b
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_1/b*
	container *
shape:?
?
sequential/linear_1/b/AssignAssignsequential/linear_1/b'sequential/linear_1/b/Initializer/zeros*(
_class
loc:@sequential/linear_1/b*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
sequential/linear_1/b/readIdentitysequential/linear_1/b*
T0*(
_class
loc:@sequential/linear_1/b*
_output_shapes	
:?
?
$sequential/sequential_2/linear_1/addAdd'sequential/sequential_2/linear_1/MatMulsequential/linear_1/b/read*
T0*(
_output_shapes
:??????????
?
Gsequential/batch_normalization_1/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_1/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_1/gamma/Initializer/onesFillGsequential/batch_normalization_1/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_1/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_1/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
	container *
shape:?
?
-sequential/batch_normalization_1/gamma/AssignAssign&sequential/batch_normalization_1/gamma7sequential/batch_normalization_1/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_1/gamma/readIdentity&sequential/batch_normalization_1/gamma*
T0*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
_output_shapes	
:?
?
Gsequential/batch_normalization_1/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_1/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_1/beta/Initializer/zerosFillGsequential/batch_normalization_1/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_1/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_1/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_1/beta
VariableV2*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_1/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
,sequential/batch_normalization_1/beta/AssignAssign%sequential/batch_normalization_1/beta7sequential/batch_normalization_1/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:?
?
*sequential/batch_normalization_1/beta/readIdentity%sequential/batch_normalization_1/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
valueB:?*
dtype0
?
Dsequential/batch_normalization_1/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_1/moving_mean/Initializer/zerosFillNsequential/batch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_1/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_1/moving_mean
VariableV2*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
	container *
shape:?*
dtype0
?
3sequential/batch_normalization_1/moving_mean/AssignAssign,sequential/batch_normalization_1/moving_mean>sequential/batch_normalization_1/moving_mean/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
validate_shape(
?
1sequential/batch_normalization_1/moving_mean/readIdentity,sequential/batch_normalization_1/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_1/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_1/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
valueB
 *  ??
?
Asequential/batch_normalization_1/moving_variance/Initializer/onesFillQsequential/batch_normalization_1/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_1/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*

index_type0
?
0sequential/batch_normalization_1/moving_variance
VariableV2*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
7sequential/batch_normalization_1/moving_variance/AssignAssign0sequential/batch_normalization_1/moving_varianceAsequential/batch_normalization_1/moving_variance/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes	
:?
?
5sequential/batch_normalization_1/moving_variance/readIdentity0sequential/batch_normalization_1/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_1/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_1/batchnorm/addAdd5sequential/batch_normalization_1/moving_variance/read0sequential/batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_1/batchnorm/RsqrtRsqrt.sequential/batch_normalization_1/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_1/batchnorm/mulMul0sequential/batch_normalization_1/batchnorm/Rsqrt+sequential/batch_normalization_1/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_1/batchnorm/mul_1Mul$sequential/sequential_2/linear_1/add.sequential/batch_normalization_1/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_1/batchnorm/mul_2Mul1sequential/batch_normalization_1/moving_mean/read.sequential/batch_normalization_1/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_1/batchnorm/subSub*sequential/batch_normalization_1/beta/read0sequential/batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_1/batchnorm/add_1Add0sequential/batch_normalization_1/batchnorm/mul_1.sequential/batch_normalization_1/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/sequential_3/ReluRelu0sequential/batch_normalization_1/batchnorm/add_1*
T0*(
_output_shapes
:??????????
?
8sequential/linear_2/w/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@sequential/linear_2/w*
valueB"?  ?  
?
7sequential/linear_2/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_2/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_2/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_2/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_2/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_2/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_2/w*
seed2 
?
6sequential/linear_2/w/Initializer/truncated_normal/mulMulBsequential/linear_2/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_2/w/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_2/w
?
2sequential/linear_2/w/Initializer/truncated_normalAdd6sequential/linear_2/w/Initializer/truncated_normal/mul7sequential/linear_2/w/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_2/w
?
sequential/linear_2/w
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_2/w*
	container 
?
sequential/linear_2/w/AssignAssignsequential/linear_2/w2sequential/linear_2/w/Initializer/truncated_normal*
use_locking(*
T0*(
_class
loc:@sequential/linear_2/w*
validate_shape(* 
_output_shapes
:
??
?
sequential/linear_2/w/readIdentitysequential/linear_2/w*
T0*(
_class
loc:@sequential/linear_2/w* 
_output_shapes
:
??
?
'sequential/sequential_4/linear_2/MatMulMatMulsequential/sequential_3/Relusequential/linear_2/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_2/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_2/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_2/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_2/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_2/b/Initializer/zerosFill7sequential/linear_2/b/Initializer/zeros/shape_as_tensor-sequential/linear_2/b/Initializer/zeros/Const*
T0*(
_class
loc:@sequential/linear_2/b*

index_type0*
_output_shapes	
:?
?
sequential/linear_2/b
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_2/b*
	container 
?
sequential/linear_2/b/AssignAssignsequential/linear_2/b'sequential/linear_2/b/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*(
_class
loc:@sequential/linear_2/b
?
sequential/linear_2/b/readIdentitysequential/linear_2/b*
T0*(
_class
loc:@sequential/linear_2/b*
_output_shapes	
:?
?
$sequential/sequential_4/linear_2/addAdd'sequential/sequential_4/linear_2/MatMulsequential/linear_2/b/read*
T0*(
_output_shapes
:??????????
?
Gsequential/batch_normalization_2/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@sequential/batch_normalization_2/gamma*
valueB:?
?
=sequential/batch_normalization_2/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@sequential/batch_normalization_2/gamma*
valueB
 *  ??
?
7sequential/batch_normalization_2/gamma/Initializer/onesFillGsequential/batch_normalization_2/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_2/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_2/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_2/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_2/gamma
?
-sequential/batch_normalization_2/gamma/AssignAssign&sequential/batch_normalization_2/gamma7sequential/batch_normalization_2/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_2/gamma/readIdentity&sequential/batch_normalization_2/gamma*
_output_shapes	
:?*
T0*9
_class/
-+loc:@sequential/batch_normalization_2/gamma
?
Gsequential/batch_normalization_2/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_2/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_2/beta/Initializer/zerosFillGsequential/batch_normalization_2/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_2/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_2/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_2/beta
VariableV2*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_2/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
,sequential/batch_normalization_2/beta/AssignAssign%sequential/batch_normalization_2/beta7sequential/batch_normalization_2/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
validate_shape(
?
*sequential/batch_normalization_2/beta/readIdentity%sequential/batch_normalization_2/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_2/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
valueB
 *    
?
>sequential/batch_normalization_2/moving_mean/Initializer/zerosFillNsequential/batch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_2/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*

index_type0
?
,sequential/batch_normalization_2/moving_mean
VariableV2*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
	container *
shape:?*
dtype0
?
3sequential/batch_normalization_2/moving_mean/AssignAssign,sequential/batch_normalization_2/moving_mean>sequential/batch_normalization_2/moving_mean/Initializer/zeros*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
1sequential/batch_normalization_2/moving_mean/readIdentity,sequential/batch_normalization_2/moving_mean*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
_output_shapes	
:?*
T0
?
Qsequential/batch_normalization_2/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_2/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_2/moving_variance/Initializer/onesFillQsequential/batch_normalization_2/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_2/moving_variance/Initializer/ones/Const*
T0*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*

index_type0*
_output_shapes	
:?
?
0sequential/batch_normalization_2/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
	container *
shape:?
?
7sequential/batch_normalization_2/moving_variance/AssignAssign0sequential/batch_normalization_2/moving_varianceAsequential/batch_normalization_2/moving_variance/Initializer/ones*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
5sequential/batch_normalization_2/moving_variance/readIdentity0sequential/batch_normalization_2/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_2/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_2/batchnorm/addAdd5sequential/batch_normalization_2/moving_variance/read0sequential/batch_normalization_2/batchnorm/add/y*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_2/batchnorm/RsqrtRsqrt.sequential/batch_normalization_2/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_2/batchnorm/mulMul0sequential/batch_normalization_2/batchnorm/Rsqrt+sequential/batch_normalization_2/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_2/batchnorm/mul_1Mul$sequential/sequential_4/linear_2/add.sequential/batch_normalization_2/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_2/batchnorm/mul_2Mul1sequential/batch_normalization_2/moving_mean/read.sequential/batch_normalization_2/batchnorm/mul*
_output_shapes	
:?*
T0
?
.sequential/batch_normalization_2/batchnorm/subSub*sequential/batch_normalization_2/beta/read0sequential/batch_normalization_2/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_2/batchnorm/add_1Add0sequential/batch_normalization_2/batchnorm/mul_1.sequential/batch_normalization_2/batchnorm/sub*(
_output_shapes
:??????????*
T0
?
sequential/sequential_5/ReluRelu0sequential/batch_normalization_2/batchnorm/add_1*(
_output_shapes
:??????????*
T0
?
8sequential/linear_3/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_3/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_3/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_3/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_3/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_3/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_3/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_3/w/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_3/w
?
6sequential/linear_3/w/Initializer/truncated_normal/mulMulBsequential/linear_3/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_3/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_3/w* 
_output_shapes
:
??
?
2sequential/linear_3/w/Initializer/truncated_normalAdd6sequential/linear_3/w/Initializer/truncated_normal/mul7sequential/linear_3/w/Initializer/truncated_normal/mean*
T0*(
_class
loc:@sequential/linear_3/w* 
_output_shapes
:
??
?
sequential/linear_3/w
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_3/w*
	container *
shape:
??
?
sequential/linear_3/w/AssignAssignsequential/linear_3/w2sequential/linear_3/w/Initializer/truncated_normal*
use_locking(*
T0*(
_class
loc:@sequential/linear_3/w*
validate_shape(* 
_output_shapes
:
??
?
sequential/linear_3/w/readIdentitysequential/linear_3/w*
T0*(
_class
loc:@sequential/linear_3/w* 
_output_shapes
:
??
?
'sequential/sequential_6/linear_3/MatMulMatMulsequential/sequential_5/Relusequential/linear_3/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_3/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_3/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_3/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_3/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_3/b/Initializer/zerosFill7sequential/linear_3/b/Initializer/zeros/shape_as_tensor-sequential/linear_3/b/Initializer/zeros/Const*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_3/b*

index_type0
?
sequential/linear_3/b
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_3/b*
	container *
shape:?
?
sequential/linear_3/b/AssignAssignsequential/linear_3/b'sequential/linear_3/b/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@sequential/linear_3/b*
validate_shape(*
_output_shapes	
:?
?
sequential/linear_3/b/readIdentitysequential/linear_3/b*
T0*(
_class
loc:@sequential/linear_3/b*
_output_shapes	
:?
?
$sequential/sequential_6/linear_3/addAdd'sequential/sequential_6/linear_3/MatMulsequential/linear_3/b/read*
T0*(
_output_shapes
:??????????
?
Gsequential/batch_normalization_3/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_3/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_3/gamma/Initializer/onesFillGsequential/batch_normalization_3/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_3/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_3/gamma
VariableV2*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
	container *
shape:?*
dtype0
?
-sequential/batch_normalization_3/gamma/AssignAssign&sequential/batch_normalization_3/gamma7sequential/batch_normalization_3/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_3/gamma/readIdentity&sequential/batch_normalization_3/gamma*
_output_shapes	
:?*
T0*9
_class/
-+loc:@sequential/batch_normalization_3/gamma
?
Gsequential/batch_normalization_3/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_3/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_3/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_3/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_3/beta/Initializer/zerosFillGsequential/batch_normalization_3/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_3/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_3/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_3/beta
VariableV2*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_3/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
,sequential/batch_normalization_3/beta/AssignAssign%sequential/batch_normalization_3/beta7sequential/batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:?
?
*sequential/batch_normalization_3/beta/readIdentity%sequential/batch_normalization_3/beta*
_output_shapes	
:?*
T0*8
_class.
,*loc:@sequential/batch_normalization_3/beta
?
Nsequential/batch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_3/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_3/moving_mean/Initializer/zerosFillNsequential/batch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_3/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_3/moving_mean
VariableV2*
shared_name *?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3sequential/batch_normalization_3/moving_mean/AssignAssign,sequential/batch_normalization_3/moving_mean>sequential/batch_normalization_3/moving_mean/Initializer/zeros*
T0*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1sequential/batch_normalization_3/moving_mean/readIdentity,sequential/batch_normalization_3/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_3/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
valueB:?
?
Gsequential/batch_normalization_3/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_3/moving_variance/Initializer/onesFillQsequential/batch_normalization_3/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_3/moving_variance/Initializer/ones/Const*
T0*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*

index_type0*
_output_shapes	
:?
?
0sequential/batch_normalization_3/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
	container *
shape:?
?
7sequential/batch_normalization_3/moving_variance/AssignAssign0sequential/batch_normalization_3/moving_varianceAsequential/batch_normalization_3/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
validate_shape(
?
5sequential/batch_normalization_3/moving_variance/readIdentity0sequential/batch_normalization_3/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_3/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_3/batchnorm/addAdd5sequential/batch_normalization_3/moving_variance/read0sequential/batch_normalization_3/batchnorm/add/y*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_3/batchnorm/RsqrtRsqrt.sequential/batch_normalization_3/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_3/batchnorm/mulMul0sequential/batch_normalization_3/batchnorm/Rsqrt+sequential/batch_normalization_3/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_3/batchnorm/mul_1Mul$sequential/sequential_6/linear_3/add.sequential/batch_normalization_3/batchnorm/mul*(
_output_shapes
:??????????*
T0
?
0sequential/batch_normalization_3/batchnorm/mul_2Mul1sequential/batch_normalization_3/moving_mean/read.sequential/batch_normalization_3/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_3/batchnorm/subSub*sequential/batch_normalization_3/beta/read0sequential/batch_normalization_3/batchnorm/mul_2*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_3/batchnorm/add_1Add0sequential/batch_normalization_3/batchnorm/mul_1.sequential/batch_normalization_3/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/AddAdd0sequential/batch_normalization_3/batchnorm/add_1sequential/sequential_3/Relu*
T0*(
_output_shapes
:??????????
g
sequential/sequential_8/ReluRelusequential/Add*
T0*(
_output_shapes
:??????????
?
8sequential/linear_4/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_4/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_4/w/Initializer/truncated_normal/meanConst*
_output_shapes
: *(
_class
loc:@sequential/linear_4/w*
valueB
 *    *
dtype0
?
9sequential/linear_4/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_4/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_4/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_4/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_4/w*
seed2 
?
6sequential/linear_4/w/Initializer/truncated_normal/mulMulBsequential/linear_4/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_4/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_4/w* 
_output_shapes
:
??
?
2sequential/linear_4/w/Initializer/truncated_normalAdd6sequential/linear_4/w/Initializer/truncated_normal/mul7sequential/linear_4/w/Initializer/truncated_normal/mean*
T0*(
_class
loc:@sequential/linear_4/w* 
_output_shapes
:
??
?
sequential/linear_4/w
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_4/w*
	container 
?
sequential/linear_4/w/AssignAssignsequential/linear_4/w2sequential/linear_4/w/Initializer/truncated_normal*
use_locking(*
T0*(
_class
loc:@sequential/linear_4/w*
validate_shape(* 
_output_shapes
:
??
?
sequential/linear_4/w/readIdentitysequential/linear_4/w* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_4/w
?
'sequential/sequential_9/linear_4/MatMulMatMulsequential/sequential_8/Relusequential/linear_4/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_4/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_4/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_4/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_4/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_4/b/Initializer/zerosFill7sequential/linear_4/b/Initializer/zeros/shape_as_tensor-sequential/linear_4/b/Initializer/zeros/Const*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_4/b*

index_type0
?
sequential/linear_4/b
VariableV2*
shared_name *(
_class
loc:@sequential/linear_4/b*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
sequential/linear_4/b/AssignAssignsequential/linear_4/b'sequential/linear_4/b/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@sequential/linear_4/b*
validate_shape(*
_output_shapes	
:?
?
sequential/linear_4/b/readIdentitysequential/linear_4/b*
T0*(
_class
loc:@sequential/linear_4/b*
_output_shapes	
:?
?
$sequential/sequential_9/linear_4/addAdd'sequential/sequential_9/linear_4/MatMulsequential/linear_4/b/read*(
_output_shapes
:??????????*
T0
?
Gsequential/batch_normalization_4/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_4/gamma/Initializer/ones/ConstConst*
_output_shapes
: *9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
valueB
 *  ??*
dtype0
?
7sequential/batch_normalization_4/gamma/Initializer/onesFillGsequential/batch_normalization_4/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_4/gamma/Initializer/ones/Const*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*

index_type0*
_output_shapes	
:?*
T0
?
&sequential/batch_normalization_4/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
	container *
shape:?
?
-sequential/batch_normalization_4/gamma/AssignAssign&sequential/batch_normalization_4/gamma7sequential/batch_normalization_4/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_4/gamma/readIdentity&sequential/batch_normalization_4/gamma*
T0*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
_output_shapes	
:?
?
Gsequential/batch_normalization_4/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_4/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_4/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@sequential/batch_normalization_4/beta*
valueB
 *    
?
7sequential/batch_normalization_4/beta/Initializer/zerosFillGsequential/batch_normalization_4/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_4/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_4/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_4/beta
VariableV2*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_4/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
,sequential/batch_normalization_4/beta/AssignAssign%sequential/batch_normalization_4/beta7sequential/batch_normalization_4/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_4/beta
?
*sequential/batch_normalization_4/beta/readIdentity%sequential/batch_normalization_4/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_4/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_4/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
valueB
 *    *
dtype0
?
>sequential/batch_normalization_4/moving_mean/Initializer/zerosFillNsequential/batch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_4/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_4/moving_mean
VariableV2*
shared_name *?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3sequential/batch_normalization_4/moving_mean/AssignAssign,sequential/batch_normalization_4/moving_mean>sequential/batch_normalization_4/moving_mean/Initializer/zeros*
T0*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1sequential/batch_normalization_4/moving_mean/readIdentity,sequential/batch_normalization_4/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_4/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*
valueB:?
?
Gsequential/batch_normalization_4/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*
valueB
 *  ??
?
Asequential/batch_normalization_4/moving_variance/Initializer/onesFillQsequential/batch_normalization_4/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_4/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*

index_type0
?
0sequential/batch_normalization_4/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@sequential/batch_normalization_4/moving_variance
?
7sequential/batch_normalization_4/moving_variance/AssignAssign0sequential/batch_normalization_4/moving_varianceAsequential/batch_normalization_4/moving_variance/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes	
:?
?
5sequential/batch_normalization_4/moving_variance/readIdentity0sequential/batch_normalization_4/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_4/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_4/batchnorm/addAdd5sequential/batch_normalization_4/moving_variance/read0sequential/batch_normalization_4/batchnorm/add/y*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_4/batchnorm/RsqrtRsqrt.sequential/batch_normalization_4/batchnorm/add*
_output_shapes	
:?*
T0
?
.sequential/batch_normalization_4/batchnorm/mulMul0sequential/batch_normalization_4/batchnorm/Rsqrt+sequential/batch_normalization_4/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_4/batchnorm/mul_1Mul$sequential/sequential_9/linear_4/add.sequential/batch_normalization_4/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_4/batchnorm/mul_2Mul1sequential/batch_normalization_4/moving_mean/read.sequential/batch_normalization_4/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_4/batchnorm/subSub*sequential/batch_normalization_4/beta/read0sequential/batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_4/batchnorm/add_1Add0sequential/batch_normalization_4/batchnorm/mul_1.sequential/batch_normalization_4/batchnorm/sub*(
_output_shapes
:??????????*
T0
?
sequential/sequential_10/ReluRelu0sequential/batch_normalization_4/batchnorm/add_1*(
_output_shapes
:??????????*
T0
?
8sequential/linear_5/w/Initializer/truncated_normal/shapeConst*
_output_shapes
:*(
_class
loc:@sequential/linear_5/w*
valueB"?  ?  *
dtype0
?
7sequential/linear_5/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_5/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_5/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_5/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_5/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_5/w/Initializer/truncated_normal/shape*(
_class
loc:@sequential/linear_5/w*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
6sequential/linear_5/w/Initializer/truncated_normal/mulMulBsequential/linear_5/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_5/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_5/w* 
_output_shapes
:
??
?
2sequential/linear_5/w/Initializer/truncated_normalAdd6sequential/linear_5/w/Initializer/truncated_normal/mul7sequential/linear_5/w/Initializer/truncated_normal/mean*
T0*(
_class
loc:@sequential/linear_5/w* 
_output_shapes
:
??
?
sequential/linear_5/w
VariableV2*
shared_name *(
_class
loc:@sequential/linear_5/w*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
sequential/linear_5/w/AssignAssignsequential/linear_5/w2sequential/linear_5/w/Initializer/truncated_normal*
T0*(
_class
loc:@sequential/linear_5/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
sequential/linear_5/w/readIdentitysequential/linear_5/w*
T0*(
_class
loc:@sequential/linear_5/w* 
_output_shapes
:
??
?
(sequential/sequential_11/linear_5/MatMulMatMulsequential/sequential_10/Relusequential/linear_5/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_5/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_5/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_5/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_5/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_5/b/Initializer/zerosFill7sequential/linear_5/b/Initializer/zeros/shape_as_tensor-sequential/linear_5/b/Initializer/zeros/Const*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_5/b*

index_type0
?
sequential/linear_5/b
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_5/b*
	container 
?
sequential/linear_5/b/AssignAssignsequential/linear_5/b'sequential/linear_5/b/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@sequential/linear_5/b*
validate_shape(*
_output_shapes	
:?
?
sequential/linear_5/b/readIdentitysequential/linear_5/b*
T0*(
_class
loc:@sequential/linear_5/b*
_output_shapes	
:?
?
%sequential/sequential_11/linear_5/addAdd(sequential/sequential_11/linear_5/MatMulsequential/linear_5/b/read*(
_output_shapes
:??????????*
T0
?
Gsequential/batch_normalization_5/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_5/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_5/gamma/Initializer/onesFillGsequential/batch_normalization_5/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_5/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_5/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
	container *
shape:?
?
-sequential/batch_normalization_5/gamma/AssignAssign&sequential/batch_normalization_5/gamma7sequential/batch_normalization_5/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_5/gamma/readIdentity&sequential/batch_normalization_5/gamma*
T0*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
_output_shapes	
:?
?
Gsequential/batch_normalization_5/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_5/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_5/beta/Initializer/zerosFillGsequential/batch_normalization_5/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_5/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_5/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_5/beta
VariableV2*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_5/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
,sequential/batch_normalization_5/beta/AssignAssign%sequential/batch_normalization_5/beta7sequential/batch_normalization_5/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:?
?
*sequential/batch_normalization_5/beta/readIdentity%sequential/batch_normalization_5/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_5/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_5/moving_mean/Initializer/zerosFillNsequential/batch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_5/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_5/moving_mean
VariableV2*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
	container *
shape:?*
dtype0
?
3sequential/batch_normalization_5/moving_mean/AssignAssign,sequential/batch_normalization_5/moving_mean>sequential/batch_normalization_5/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean
?
1sequential/batch_normalization_5/moving_mean/readIdentity,sequential/batch_normalization_5/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_5/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_5/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_5/moving_variance/Initializer/onesFillQsequential/batch_normalization_5/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_5/moving_variance/Initializer/ones/Const*
T0*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*

index_type0*
_output_shapes	
:?
?
0sequential/batch_normalization_5/moving_variance
VariableV2*
shared_name *C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
7sequential/batch_normalization_5/moving_variance/AssignAssign0sequential/batch_normalization_5/moving_varianceAsequential/batch_normalization_5/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
validate_shape(
?
5sequential/batch_normalization_5/moving_variance/readIdentity0sequential/batch_normalization_5/moving_variance*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance
u
0sequential/batch_normalization_5/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_5/batchnorm/addAdd5sequential/batch_normalization_5/moving_variance/read0sequential/batch_normalization_5/batchnorm/add/y*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_5/batchnorm/RsqrtRsqrt.sequential/batch_normalization_5/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_5/batchnorm/mulMul0sequential/batch_normalization_5/batchnorm/Rsqrt+sequential/batch_normalization_5/gamma/read*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_5/batchnorm/mul_1Mul%sequential/sequential_11/linear_5/add.sequential/batch_normalization_5/batchnorm/mul*(
_output_shapes
:??????????*
T0
?
0sequential/batch_normalization_5/batchnorm/mul_2Mul1sequential/batch_normalization_5/moving_mean/read.sequential/batch_normalization_5/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_5/batchnorm/subSub*sequential/batch_normalization_5/beta/read0sequential/batch_normalization_5/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_5/batchnorm/add_1Add0sequential/batch_normalization_5/batchnorm/mul_1.sequential/batch_normalization_5/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/Add_1Add0sequential/batch_normalization_5/batchnorm/add_1sequential/sequential_8/Relu*
T0*(
_output_shapes
:??????????
j
sequential/sequential_13/ReluRelusequential/Add_1*
T0*(
_output_shapes
:??????????
?
8sequential/linear_6/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_6/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_6/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_6/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_6/w/Initializer/truncated_normal/stddevConst*
_output_shapes
: *(
_class
loc:@sequential/linear_6/w*
valueB
 *??=*
dtype0
?
Bsequential/linear_6/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_6/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_6/w*
seed2 
?
6sequential/linear_6/w/Initializer/truncated_normal/mulMulBsequential/linear_6/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_6/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_6/w* 
_output_shapes
:
??
?
2sequential/linear_6/w/Initializer/truncated_normalAdd6sequential/linear_6/w/Initializer/truncated_normal/mul7sequential/linear_6/w/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_6/w
?
sequential/linear_6/w
VariableV2*
shared_name *(
_class
loc:@sequential/linear_6/w*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
sequential/linear_6/w/AssignAssignsequential/linear_6/w2sequential/linear_6/w/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*(
_class
loc:@sequential/linear_6/w
?
sequential/linear_6/w/readIdentitysequential/linear_6/w*
T0*(
_class
loc:@sequential/linear_6/w* 
_output_shapes
:
??
?
(sequential/sequential_14/linear_6/MatMulMatMulsequential/sequential_13/Relusequential/linear_6/w/read*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( *
T0
?
7sequential/linear_6/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_6/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_6/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_6/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_6/b/Initializer/zerosFill7sequential/linear_6/b/Initializer/zeros/shape_as_tensor-sequential/linear_6/b/Initializer/zeros/Const*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_6/b*

index_type0
?
sequential/linear_6/b
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_6/b*
	container *
shape:?
?
sequential/linear_6/b/AssignAssignsequential/linear_6/b'sequential/linear_6/b/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@sequential/linear_6/b*
validate_shape(*
_output_shapes	
:?
?
sequential/linear_6/b/readIdentitysequential/linear_6/b*
T0*(
_class
loc:@sequential/linear_6/b*
_output_shapes	
:?
?
%sequential/sequential_14/linear_6/addAdd(sequential/sequential_14/linear_6/MatMulsequential/linear_6/b/read*(
_output_shapes
:??????????*
T0
?
Gsequential/batch_normalization_6/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_6/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_6/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_6/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_6/gamma/Initializer/onesFillGsequential/batch_normalization_6/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_6/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_6/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_6/gamma
VariableV2*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_6/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-sequential/batch_normalization_6/gamma/AssignAssign&sequential/batch_normalization_6/gamma7sequential/batch_normalization_6/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_6/gamma
?
+sequential/batch_normalization_6/gamma/readIdentity&sequential/batch_normalization_6/gamma*
T0*9
_class/
-+loc:@sequential/batch_normalization_6/gamma*
_output_shapes	
:?
?
Gsequential/batch_normalization_6/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_6/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_6/beta/Initializer/zerosFillGsequential/batch_normalization_6/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_6/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_6/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_6/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_6/beta*
	container 
?
,sequential/batch_normalization_6/beta/AssignAssign%sequential/batch_normalization_6/beta7sequential/batch_normalization_6/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
validate_shape(
?
*sequential/batch_normalization_6/beta/readIdentity%sequential/batch_normalization_6/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
valueB:?
?
Dsequential/batch_normalization_6/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_6/moving_mean/Initializer/zerosFillNsequential/batch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_6/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_6/moving_mean
VariableV2*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
3sequential/batch_normalization_6/moving_mean/AssignAssign,sequential/batch_normalization_6/moving_mean>sequential/batch_normalization_6/moving_mean/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
validate_shape(*
_output_shapes	
:?
?
1sequential/batch_normalization_6/moving_mean/readIdentity,sequential/batch_normalization_6/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_6/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_6/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_6/moving_variance/Initializer/onesFillQsequential/batch_normalization_6/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_6/moving_variance/Initializer/ones/Const*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*

index_type0*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_6/moving_variance
VariableV2*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
7sequential/batch_normalization_6/moving_variance/AssignAssign0sequential/batch_normalization_6/moving_varianceAsequential/batch_normalization_6/moving_variance/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:?
?
5sequential/batch_normalization_6/moving_variance/readIdentity0sequential/batch_normalization_6/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_6/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_6/batchnorm/addAdd5sequential/batch_normalization_6/moving_variance/read0sequential/batch_normalization_6/batchnorm/add/y*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_6/batchnorm/RsqrtRsqrt.sequential/batch_normalization_6/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_6/batchnorm/mulMul0sequential/batch_normalization_6/batchnorm/Rsqrt+sequential/batch_normalization_6/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_6/batchnorm/mul_1Mul%sequential/sequential_14/linear_6/add.sequential/batch_normalization_6/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_6/batchnorm/mul_2Mul1sequential/batch_normalization_6/moving_mean/read.sequential/batch_normalization_6/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_6/batchnorm/subSub*sequential/batch_normalization_6/beta/read0sequential/batch_normalization_6/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_6/batchnorm/add_1Add0sequential/batch_normalization_6/batchnorm/mul_1.sequential/batch_normalization_6/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/sequential_15/ReluRelu0sequential/batch_normalization_6/batchnorm/add_1*(
_output_shapes
:??????????*
T0
?
8sequential/linear_7/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_7/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_7/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_7/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_7/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_7/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_7/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_7/w/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_7/w
?
6sequential/linear_7/w/Initializer/truncated_normal/mulMulBsequential/linear_7/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_7/w/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_7/w
?
2sequential/linear_7/w/Initializer/truncated_normalAdd6sequential/linear_7/w/Initializer/truncated_normal/mul7sequential/linear_7/w/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_7/w
?
sequential/linear_7/w
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_7/w*
	container 
?
sequential/linear_7/w/AssignAssignsequential/linear_7/w2sequential/linear_7/w/Initializer/truncated_normal*(
_class
loc:@sequential/linear_7/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
sequential/linear_7/w/readIdentitysequential/linear_7/w* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_7/w
?
(sequential/sequential_16/linear_7/MatMulMatMulsequential/sequential_15/Relusequential/linear_7/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_7/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_7/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_7/b/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *(
_class
loc:@sequential/linear_7/b*
valueB
 *    
?
'sequential/linear_7/b/Initializer/zerosFill7sequential/linear_7/b/Initializer/zeros/shape_as_tensor-sequential/linear_7/b/Initializer/zeros/Const*
T0*(
_class
loc:@sequential/linear_7/b*

index_type0*
_output_shapes	
:?
?
sequential/linear_7/b
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_7/b*
	container *
shape:?
?
sequential/linear_7/b/AssignAssignsequential/linear_7/b'sequential/linear_7/b/Initializer/zeros*
T0*(
_class
loc:@sequential/linear_7/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
sequential/linear_7/b/readIdentitysequential/linear_7/b*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_7/b
?
%sequential/sequential_16/linear_7/addAdd(sequential/sequential_16/linear_7/MatMulsequential/linear_7/b/read*
T0*(
_output_shapes
:??????????
?
Gsequential/batch_normalization_7/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
valueB:?
?
=sequential/batch_normalization_7/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_7/gamma/Initializer/onesFillGsequential/batch_normalization_7/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_7/gamma/Initializer/ones/Const*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*

index_type0*
_output_shapes	
:?*
T0
?
&sequential/batch_normalization_7/gamma
VariableV2*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-sequential/batch_normalization_7/gamma/AssignAssign&sequential/batch_normalization_7/gamma7sequential/batch_normalization_7/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_7/gamma/readIdentity&sequential/batch_normalization_7/gamma*
_output_shapes	
:?*
T0*9
_class/
-+loc:@sequential/batch_normalization_7/gamma
?
Gsequential/batch_normalization_7/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_7/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_7/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_7/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_7/beta/Initializer/zerosFillGsequential/batch_normalization_7/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_7/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_7/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_7/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_7/beta*
	container 
?
,sequential/batch_normalization_7/beta/AssignAssign%sequential/batch_normalization_7/beta7sequential/batch_normalization_7/beta/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:?
?
*sequential/batch_normalization_7/beta/readIdentity%sequential/batch_normalization_7/beta*
_output_shapes	
:?*
T0*8
_class.
,*loc:@sequential/batch_normalization_7/beta
?
Nsequential/batch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_7/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_7/moving_mean/Initializer/zerosFillNsequential/batch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_7/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_7/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
	container *
shape:?
?
3sequential/batch_normalization_7/moving_mean/AssignAssign,sequential/batch_normalization_7/moving_mean>sequential/batch_normalization_7/moving_mean/Initializer/zeros*
T0*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1sequential/batch_normalization_7/moving_mean/readIdentity,sequential/batch_normalization_7/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_7/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_7/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_7/moving_variance/Initializer/onesFillQsequential/batch_normalization_7/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_7/moving_variance/Initializer/ones/Const*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*

index_type0*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_7/moving_variance
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*
	container 
?
7sequential/batch_normalization_7/moving_variance/AssignAssign0sequential/batch_normalization_7/moving_varianceAsequential/batch_normalization_7/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*
validate_shape(
?
5sequential/batch_normalization_7/moving_variance/readIdentity0sequential/batch_normalization_7/moving_variance*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance
u
0sequential/batch_normalization_7/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_7/batchnorm/addAdd5sequential/batch_normalization_7/moving_variance/read0sequential/batch_normalization_7/batchnorm/add/y*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_7/batchnorm/RsqrtRsqrt.sequential/batch_normalization_7/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_7/batchnorm/mulMul0sequential/batch_normalization_7/batchnorm/Rsqrt+sequential/batch_normalization_7/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_7/batchnorm/mul_1Mul%sequential/sequential_16/linear_7/add.sequential/batch_normalization_7/batchnorm/mul*(
_output_shapes
:??????????*
T0
?
0sequential/batch_normalization_7/batchnorm/mul_2Mul1sequential/batch_normalization_7/moving_mean/read.sequential/batch_normalization_7/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_7/batchnorm/subSub*sequential/batch_normalization_7/beta/read0sequential/batch_normalization_7/batchnorm/mul_2*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_7/batchnorm/add_1Add0sequential/batch_normalization_7/batchnorm/mul_1.sequential/batch_normalization_7/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/Add_2Add0sequential/batch_normalization_7/batchnorm/add_1sequential/sequential_13/Relu*
T0*(
_output_shapes
:??????????
j
sequential/sequential_18/ReluRelusequential/Add_2*(
_output_shapes
:??????????*
T0
?
8sequential/linear_8/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_8/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_8/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_8/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_8/w/Initializer/truncated_normal/stddevConst*(
_class
loc:@sequential/linear_8/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Bsequential/linear_8/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_8/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_8/w*
seed2 
?
6sequential/linear_8/w/Initializer/truncated_normal/mulMulBsequential/linear_8/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_8/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_8/w* 
_output_shapes
:
??
?
2sequential/linear_8/w/Initializer/truncated_normalAdd6sequential/linear_8/w/Initializer/truncated_normal/mul7sequential/linear_8/w/Initializer/truncated_normal/mean*
T0*(
_class
loc:@sequential/linear_8/w* 
_output_shapes
:
??
?
sequential/linear_8/w
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_8/w*
	container *
shape:
??
?
sequential/linear_8/w/AssignAssignsequential/linear_8/w2sequential/linear_8/w/Initializer/truncated_normal*
T0*(
_class
loc:@sequential/linear_8/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
sequential/linear_8/w/readIdentitysequential/linear_8/w* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_8/w
?
(sequential/sequential_19/linear_8/MatMulMatMulsequential/sequential_18/Relusequential/linear_8/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_8/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_8/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_8/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_8/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_8/b/Initializer/zerosFill7sequential/linear_8/b/Initializer/zeros/shape_as_tensor-sequential/linear_8/b/Initializer/zeros/Const*(
_class
loc:@sequential/linear_8/b*

index_type0*
_output_shapes	
:?*
T0
?
sequential/linear_8/b
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *(
_class
loc:@sequential/linear_8/b*
	container *
shape:?
?
sequential/linear_8/b/AssignAssignsequential/linear_8/b'sequential/linear_8/b/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*(
_class
loc:@sequential/linear_8/b
?
sequential/linear_8/b/readIdentitysequential/linear_8/b*
_output_shapes	
:?*
T0*(
_class
loc:@sequential/linear_8/b
?
%sequential/sequential_19/linear_8/addAdd(sequential/sequential_19/linear_8/MatMulsequential/linear_8/b/read*
T0*(
_output_shapes
:??????????
?
Gsequential/batch_normalization_8/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*9
_class/
-+loc:@sequential/batch_normalization_8/gamma*
valueB:?
?
=sequential/batch_normalization_8/gamma/Initializer/ones/ConstConst*
_output_shapes
: *9
_class/
-+loc:@sequential/batch_normalization_8/gamma*
valueB
 *  ??*
dtype0
?
7sequential/batch_normalization_8/gamma/Initializer/onesFillGsequential/batch_normalization_8/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_8/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_8/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_8/gamma
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_8/gamma*
	container 
?
-sequential/batch_normalization_8/gamma/AssignAssign&sequential/batch_normalization_8/gamma7sequential/batch_normalization_8/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_8/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_8/gamma/readIdentity&sequential/batch_normalization_8/gamma*
_output_shapes	
:?*
T0*9
_class/
-+loc:@sequential/batch_normalization_8/gamma
?
Gsequential/batch_normalization_8/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_8/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_8/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_8/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_8/beta/Initializer/zerosFillGsequential/batch_normalization_8/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_8/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_8/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_8/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_8/beta*
	container *
shape:?
?
,sequential/batch_normalization_8/beta/AssignAssign%sequential/batch_normalization_8/beta7sequential/batch_normalization_8/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_8/beta
?
*sequential/batch_normalization_8/beta/readIdentity%sequential/batch_normalization_8/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_8/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
valueB:?*
dtype0
?
Dsequential/batch_normalization_8/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_8/moving_mean/Initializer/zerosFillNsequential/batch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_8/moving_mean/Initializer/zeros/Const*
T0*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*

index_type0*
_output_shapes	
:?
?
,sequential/batch_normalization_8/moving_mean
VariableV2*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
	container *
shape:?*
dtype0
?
3sequential/batch_normalization_8/moving_mean/AssignAssign,sequential/batch_normalization_8/moving_mean>sequential/batch_normalization_8/moving_mean/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
validate_shape(
?
1sequential/batch_normalization_8/moving_mean/readIdentity,sequential/batch_normalization_8/moving_mean*
T0*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
_output_shapes	
:?
?
Qsequential/batch_normalization_8/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_8/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_8/moving_variance/Initializer/onesFillQsequential/batch_normalization_8/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_8/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*

index_type0
?
0sequential/batch_normalization_8/moving_variance
VariableV2*
shared_name *C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
7sequential/batch_normalization_8/moving_variance/AssignAssign0sequential/batch_normalization_8/moving_varianceAsequential/batch_normalization_8/moving_variance/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes	
:?
?
5sequential/batch_normalization_8/moving_variance/readIdentity0sequential/batch_normalization_8/moving_variance*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
_output_shapes	
:?*
T0
u
0sequential/batch_normalization_8/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_8/batchnorm/addAdd5sequential/batch_normalization_8/moving_variance/read0sequential/batch_normalization_8/batchnorm/add/y*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_8/batchnorm/RsqrtRsqrt.sequential/batch_normalization_8/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_8/batchnorm/mulMul0sequential/batch_normalization_8/batchnorm/Rsqrt+sequential/batch_normalization_8/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_8/batchnorm/mul_1Mul%sequential/sequential_19/linear_8/add.sequential/batch_normalization_8/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_8/batchnorm/mul_2Mul1sequential/batch_normalization_8/moving_mean/read.sequential/batch_normalization_8/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_8/batchnorm/subSub*sequential/batch_normalization_8/beta/read0sequential/batch_normalization_8/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_8/batchnorm/add_1Add0sequential/batch_normalization_8/batchnorm/mul_1.sequential/batch_normalization_8/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/sequential_20/ReluRelu0sequential/batch_normalization_8/batchnorm/add_1*
T0*(
_output_shapes
:??????????
?
8sequential/linear_9/w/Initializer/truncated_normal/shapeConst*(
_class
loc:@sequential/linear_9/w*
valueB"?  ?  *
dtype0*
_output_shapes
:
?
7sequential/linear_9/w/Initializer/truncated_normal/meanConst*(
_class
loc:@sequential/linear_9/w*
valueB
 *    *
dtype0*
_output_shapes
: 
?
9sequential/linear_9/w/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *(
_class
loc:@sequential/linear_9/w*
valueB
 *??=
?
Bsequential/linear_9/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal8sequential/linear_9/w/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*(
_class
loc:@sequential/linear_9/w*
seed2 
?
6sequential/linear_9/w/Initializer/truncated_normal/mulMulBsequential/linear_9/w/Initializer/truncated_normal/TruncatedNormal9sequential/linear_9/w/Initializer/truncated_normal/stddev*
T0*(
_class
loc:@sequential/linear_9/w* 
_output_shapes
:
??
?
2sequential/linear_9/w/Initializer/truncated_normalAdd6sequential/linear_9/w/Initializer/truncated_normal/mul7sequential/linear_9/w/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*(
_class
loc:@sequential/linear_9/w
?
sequential/linear_9/w
VariableV2* 
_output_shapes
:
??*
shared_name *(
_class
loc:@sequential/linear_9/w*
	container *
shape:
??*
dtype0
?
sequential/linear_9/w/AssignAssignsequential/linear_9/w2sequential/linear_9/w/Initializer/truncated_normal*
use_locking(*
T0*(
_class
loc:@sequential/linear_9/w*
validate_shape(* 
_output_shapes
:
??
?
sequential/linear_9/w/readIdentitysequential/linear_9/w*
T0*(
_class
loc:@sequential/linear_9/w* 
_output_shapes
:
??
?
(sequential/sequential_21/linear_9/MatMulMatMulsequential/sequential_20/Relusequential/linear_9/w/read*
T0*(
_output_shapes
:??????????*
transpose_a( *
transpose_b( 
?
7sequential/linear_9/b/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@sequential/linear_9/b*
valueB:?*
dtype0*
_output_shapes
:
?
-sequential/linear_9/b/Initializer/zeros/ConstConst*(
_class
loc:@sequential/linear_9/b*
valueB
 *    *
dtype0*
_output_shapes
: 
?
'sequential/linear_9/b/Initializer/zerosFill7sequential/linear_9/b/Initializer/zeros/shape_as_tensor-sequential/linear_9/b/Initializer/zeros/Const*(
_class
loc:@sequential/linear_9/b*

index_type0*
_output_shapes	
:?*
T0
?
sequential/linear_9/b
VariableV2*
shared_name *(
_class
loc:@sequential/linear_9/b*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
sequential/linear_9/b/AssignAssignsequential/linear_9/b'sequential/linear_9/b/Initializer/zeros*
T0*(
_class
loc:@sequential/linear_9/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
sequential/linear_9/b/readIdentitysequential/linear_9/b*
T0*(
_class
loc:@sequential/linear_9/b*
_output_shapes	
:?
?
%sequential/sequential_21/linear_9/addAdd(sequential/sequential_21/linear_9/MatMulsequential/linear_9/b/read*(
_output_shapes
:??????????*
T0
?
Gsequential/batch_normalization_9/gamma/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_9/gamma/Initializer/ones/ConstConst*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_9/gamma/Initializer/onesFillGsequential/batch_normalization_9/gamma/Initializer/ones/shape_as_tensor=sequential/batch_normalization_9/gamma/Initializer/ones/Const*
T0*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*

index_type0*
_output_shapes	
:?
?
&sequential/batch_normalization_9/gamma
VariableV2*
shared_name *9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-sequential/batch_normalization_9/gamma/AssignAssign&sequential/batch_normalization_9/gamma7sequential/batch_normalization_9/gamma/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
validate_shape(*
_output_shapes	
:?
?
+sequential/batch_normalization_9/gamma/readIdentity&sequential/batch_normalization_9/gamma*
T0*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
_output_shapes	
:?
?
Gsequential/batch_normalization_9/beta/Initializer/zeros/shape_as_tensorConst*8
_class.
,*loc:@sequential/batch_normalization_9/beta*
valueB:?*
dtype0*
_output_shapes
:
?
=sequential/batch_normalization_9/beta/Initializer/zeros/ConstConst*8
_class.
,*loc:@sequential/batch_normalization_9/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7sequential/batch_normalization_9/beta/Initializer/zerosFillGsequential/batch_normalization_9/beta/Initializer/zeros/shape_as_tensor=sequential/batch_normalization_9/beta/Initializer/zeros/Const*
T0*8
_class.
,*loc:@sequential/batch_normalization_9/beta*

index_type0*
_output_shapes	
:?
?
%sequential/batch_normalization_9/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *8
_class.
,*loc:@sequential/batch_normalization_9/beta
?
,sequential/batch_normalization_9/beta/AssignAssign%sequential/batch_normalization_9/beta7sequential/batch_normalization_9/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_9/beta*
validate_shape(
?
*sequential/batch_normalization_9/beta/readIdentity%sequential/batch_normalization_9/beta*
T0*8
_class.
,*loc:@sequential/batch_normalization_9/beta*
_output_shapes	
:?
?
Nsequential/batch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
Dsequential/batch_normalization_9/moving_mean/Initializer/zeros/ConstConst*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>sequential/batch_normalization_9/moving_mean/Initializer/zerosFillNsequential/batch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensorDsequential/batch_normalization_9/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*

index_type0
?
,sequential/batch_normalization_9/moving_mean
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
	container 
?
3sequential/batch_normalization_9/moving_mean/AssignAssign,sequential/batch_normalization_9/moving_mean>sequential/batch_normalization_9/moving_mean/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
validate_shape(
?
1sequential/batch_normalization_9/moving_mean/readIdentity,sequential/batch_normalization_9/moving_mean*
_output_shapes	
:?*
T0*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean
?
Qsequential/batch_normalization_9/moving_variance/Initializer/ones/shape_as_tensorConst*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Gsequential/batch_normalization_9/moving_variance/Initializer/ones/ConstConst*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Asequential/batch_normalization_9/moving_variance/Initializer/onesFillQsequential/batch_normalization_9/moving_variance/Initializer/ones/shape_as_tensorGsequential/batch_normalization_9/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance*

index_type0
?
0sequential/batch_normalization_9/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@sequential/batch_normalization_9/moving_variance
?
7sequential/batch_normalization_9/moving_variance/AssignAssign0sequential/batch_normalization_9/moving_varianceAsequential/batch_normalization_9/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance
?
5sequential/batch_normalization_9/moving_variance/readIdentity0sequential/batch_normalization_9/moving_variance*
T0*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance*
_output_shapes	
:?
u
0sequential/batch_normalization_9/batchnorm/add/yConst*
valueB
 *??'7*
dtype0*
_output_shapes
: 
?
.sequential/batch_normalization_9/batchnorm/addAdd5sequential/batch_normalization_9/moving_variance/read0sequential/batch_normalization_9/batchnorm/add/y*
_output_shapes	
:?*
T0
?
0sequential/batch_normalization_9/batchnorm/RsqrtRsqrt.sequential/batch_normalization_9/batchnorm/add*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_9/batchnorm/mulMul0sequential/batch_normalization_9/batchnorm/Rsqrt+sequential/batch_normalization_9/gamma/read*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_9/batchnorm/mul_1Mul%sequential/sequential_21/linear_9/add.sequential/batch_normalization_9/batchnorm/mul*
T0*(
_output_shapes
:??????????
?
0sequential/batch_normalization_9/batchnorm/mul_2Mul1sequential/batch_normalization_9/moving_mean/read.sequential/batch_normalization_9/batchnorm/mul*
T0*
_output_shapes	
:?
?
.sequential/batch_normalization_9/batchnorm/subSub*sequential/batch_normalization_9/beta/read0sequential/batch_normalization_9/batchnorm/mul_2*
T0*
_output_shapes	
:?
?
0sequential/batch_normalization_9/batchnorm/add_1Add0sequential/batch_normalization_9/batchnorm/mul_1.sequential/batch_normalization_9/batchnorm/sub*
T0*(
_output_shapes
:??????????
?
sequential/Add_3Add0sequential/batch_normalization_9/batchnorm/add_1sequential/sequential_18/Relu*
T0*(
_output_shapes
:??????????
j
sequential/sequential_23/ReluRelusequential/Add_3*
T0*(
_output_shapes
:??????????
a
sequential/dropout/keep_probConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
9sequential/linear_10/w/Initializer/truncated_normal/shapeConst*)
_class
loc:@sequential/linear_10/w*
valueB"?     *
dtype0*
_output_shapes
:
?
8sequential/linear_10/w/Initializer/truncated_normal/meanConst*
_output_shapes
: *)
_class
loc:@sequential/linear_10/w*
valueB
 *    *
dtype0
?
:sequential/linear_10/w/Initializer/truncated_normal/stddevConst*)
_class
loc:@sequential/linear_10/w*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
Csequential/linear_10/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9sequential/linear_10/w/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes
:	?*

seed *
T0*)
_class
loc:@sequential/linear_10/w
?
7sequential/linear_10/w/Initializer/truncated_normal/mulMulCsequential/linear_10/w/Initializer/truncated_normal/TruncatedNormal:sequential/linear_10/w/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@sequential/linear_10/w*
_output_shapes
:	?
?
3sequential/linear_10/w/Initializer/truncated_normalAdd7sequential/linear_10/w/Initializer/truncated_normal/mul8sequential/linear_10/w/Initializer/truncated_normal/mean*
T0*)
_class
loc:@sequential/linear_10/w*
_output_shapes
:	?
?
sequential/linear_10/w
VariableV2*
shape:	?*
dtype0*
_output_shapes
:	?*
shared_name *)
_class
loc:@sequential/linear_10/w*
	container 
?
sequential/linear_10/w/AssignAssignsequential/linear_10/w3sequential/linear_10/w/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@sequential/linear_10/w*
validate_shape(*
_output_shapes
:	?
?
sequential/linear_10/w/readIdentitysequential/linear_10/w*
T0*)
_class
loc:@sequential/linear_10/w*
_output_shapes
:	?
?
)sequential/sequential_24/linear_10/MatMulMatMulsequential/sequential_23/Relusequential/linear_10/w/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
?
(sequential/linear_10/b/Initializer/zerosConst*)
_class
loc:@sequential/linear_10/b*
valueB*    *
dtype0*
_output_shapes
:
?
sequential/linear_10/b
VariableV2*
shared_name *)
_class
loc:@sequential/linear_10/b*
	container *
shape:*
dtype0*
_output_shapes
:
?
sequential/linear_10/b/AssignAssignsequential/linear_10/b(sequential/linear_10/b/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@sequential/linear_10/b*
validate_shape(*
_output_shapes
:
?
sequential/linear_10/b/readIdentitysequential/linear_10/b*
T0*)
_class
loc:@sequential/linear_10/b*
_output_shapes
:
?
&sequential/sequential_24/linear_10/addAdd)sequential/sequential_24/linear_10/MatMulsequential/linear_10/b/read*
T0*'
_output_shapes
:?????????
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_05ba5210cda34863b12b7aa4567aa655/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B??Bglobal_stepB#sequential/batch_normalization/betaB$sequential/batch_normalization/gammaB*sequential/batch_normalization/moving_meanB.sequential/batch_normalization/moving_varianceB%sequential/batch_normalization_1/betaB&sequential/batch_normalization_1/gammaB,sequential/batch_normalization_1/moving_meanB0sequential/batch_normalization_1/moving_varianceB%sequential/batch_normalization_2/betaB&sequential/batch_normalization_2/gammaB,sequential/batch_normalization_2/moving_meanB0sequential/batch_normalization_2/moving_varianceB%sequential/batch_normalization_3/betaB&sequential/batch_normalization_3/gammaB,sequential/batch_normalization_3/moving_meanB0sequential/batch_normalization_3/moving_varianceB%sequential/batch_normalization_4/betaB&sequential/batch_normalization_4/gammaB,sequential/batch_normalization_4/moving_meanB0sequential/batch_normalization_4/moving_varianceB%sequential/batch_normalization_5/betaB&sequential/batch_normalization_5/gammaB,sequential/batch_normalization_5/moving_meanB0sequential/batch_normalization_5/moving_varianceB%sequential/batch_normalization_6/betaB&sequential/batch_normalization_6/gammaB,sequential/batch_normalization_6/moving_meanB0sequential/batch_normalization_6/moving_varianceB%sequential/batch_normalization_7/betaB&sequential/batch_normalization_7/gammaB,sequential/batch_normalization_7/moving_meanB0sequential/batch_normalization_7/moving_varianceB%sequential/batch_normalization_8/betaB&sequential/batch_normalization_8/gammaB,sequential/batch_normalization_8/moving_meanB0sequential/batch_normalization_8/moving_varianceB%sequential/batch_normalization_9/betaB&sequential/batch_normalization_9/gammaB,sequential/batch_normalization_9/moving_meanB0sequential/batch_normalization_9/moving_varianceBsequential/linear/bBsequential/linear/wBsequential/linear_1/bBsequential/linear_1/wBsequential/linear_10/bBsequential/linear_10/wBsequential/linear_2/bBsequential/linear_2/wBsequential/linear_3/bBsequential/linear_3/wBsequential/linear_4/bBsequential/linear_4/wBsequential/linear_5/bBsequential/linear_5/wBsequential/linear_6/bBsequential/linear_6/wBsequential/linear_7/bBsequential/linear_7/wBsequential/linear_8/bBsequential/linear_8/wBsequential/linear_9/bBsequential/linear_9/w*
dtype0*
_output_shapes
:?
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step#sequential/batch_normalization/beta$sequential/batch_normalization/gamma*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variance%sequential/batch_normalization_1/beta&sequential/batch_normalization_1/gamma,sequential/batch_normalization_1/moving_mean0sequential/batch_normalization_1/moving_variance%sequential/batch_normalization_2/beta&sequential/batch_normalization_2/gamma,sequential/batch_normalization_2/moving_mean0sequential/batch_normalization_2/moving_variance%sequential/batch_normalization_3/beta&sequential/batch_normalization_3/gamma,sequential/batch_normalization_3/moving_mean0sequential/batch_normalization_3/moving_variance%sequential/batch_normalization_4/beta&sequential/batch_normalization_4/gamma,sequential/batch_normalization_4/moving_mean0sequential/batch_normalization_4/moving_variance%sequential/batch_normalization_5/beta&sequential/batch_normalization_5/gamma,sequential/batch_normalization_5/moving_mean0sequential/batch_normalization_5/moving_variance%sequential/batch_normalization_6/beta&sequential/batch_normalization_6/gamma,sequential/batch_normalization_6/moving_mean0sequential/batch_normalization_6/moving_variance%sequential/batch_normalization_7/beta&sequential/batch_normalization_7/gamma,sequential/batch_normalization_7/moving_mean0sequential/batch_normalization_7/moving_variance%sequential/batch_normalization_8/beta&sequential/batch_normalization_8/gamma,sequential/batch_normalization_8/moving_mean0sequential/batch_normalization_8/moving_variance%sequential/batch_normalization_9/beta&sequential/batch_normalization_9/gamma,sequential/batch_normalization_9/moving_mean0sequential/batch_normalization_9/moving_variancesequential/linear/bsequential/linear/wsequential/linear_1/bsequential/linear_1/wsequential/linear_10/bsequential/linear_10/wsequential/linear_2/bsequential/linear_2/wsequential/linear_3/bsequential/linear_3/wsequential/linear_4/bsequential/linear_4/wsequential/linear_5/bsequential/linear_5/wsequential/linear_6/bsequential/linear_6/wsequential/linear_7/bsequential/linear_7/wsequential/linear_8/bsequential/linear_8/wsequential/linear_9/bsequential/linear_9/w"/device:CPU:0*M
dtypesC
A2?	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B??Bglobal_stepB#sequential/batch_normalization/betaB$sequential/batch_normalization/gammaB*sequential/batch_normalization/moving_meanB.sequential/batch_normalization/moving_varianceB%sequential/batch_normalization_1/betaB&sequential/batch_normalization_1/gammaB,sequential/batch_normalization_1/moving_meanB0sequential/batch_normalization_1/moving_varianceB%sequential/batch_normalization_2/betaB&sequential/batch_normalization_2/gammaB,sequential/batch_normalization_2/moving_meanB0sequential/batch_normalization_2/moving_varianceB%sequential/batch_normalization_3/betaB&sequential/batch_normalization_3/gammaB,sequential/batch_normalization_3/moving_meanB0sequential/batch_normalization_3/moving_varianceB%sequential/batch_normalization_4/betaB&sequential/batch_normalization_4/gammaB,sequential/batch_normalization_4/moving_meanB0sequential/batch_normalization_4/moving_varianceB%sequential/batch_normalization_5/betaB&sequential/batch_normalization_5/gammaB,sequential/batch_normalization_5/moving_meanB0sequential/batch_normalization_5/moving_varianceB%sequential/batch_normalization_6/betaB&sequential/batch_normalization_6/gammaB,sequential/batch_normalization_6/moving_meanB0sequential/batch_normalization_6/moving_varianceB%sequential/batch_normalization_7/betaB&sequential/batch_normalization_7/gammaB,sequential/batch_normalization_7/moving_meanB0sequential/batch_normalization_7/moving_varianceB%sequential/batch_normalization_8/betaB&sequential/batch_normalization_8/gammaB,sequential/batch_normalization_8/moving_meanB0sequential/batch_normalization_8/moving_varianceB%sequential/batch_normalization_9/betaB&sequential/batch_normalization_9/gammaB,sequential/batch_normalization_9/moving_meanB0sequential/batch_normalization_9/moving_varianceBsequential/linear/bBsequential/linear/wBsequential/linear_1/bBsequential/linear_1/wBsequential/linear_10/bBsequential/linear_10/wBsequential/linear_2/bBsequential/linear_2/wBsequential/linear_3/bBsequential/linear_3/wBsequential/linear_4/bBsequential/linear_4/wBsequential/linear_5/bBsequential/linear_5/wBsequential/linear_6/bBsequential/linear_6/wBsequential/linear_7/bBsequential/linear_7/wBsequential/linear_8/bBsequential/linear_8/wBsequential/linear_9/bBsequential/linear_9/w*
dtype0*
_output_shapes
:?
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:?*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	
?
save/AssignAssignglobal_stepsave/RestoreV2*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(
?
save/Assign_1Assign#sequential/batch_normalization/betasave/RestoreV2:1*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0*6
_class,
*(loc:@sequential/batch_normalization/beta
?
save/Assign_2Assign$sequential/batch_normalization/gammasave/RestoreV2:2*
use_locking(*
T0*7
_class-
+)loc:@sequential/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:?'
?
save/Assign_3Assign*sequential/batch_normalization/moving_meansave/RestoreV2:3*
use_locking(*
T0*=
_class3
1/loc:@sequential/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:?'
?
save/Assign_4Assign.sequential/batch_normalization/moving_variancesave/RestoreV2:4*
T0*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:?'*
use_locking(
?
save/Assign_5Assign%sequential/batch_normalization_1/betasave/RestoreV2:5*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_6Assign&sequential/batch_normalization_1/gammasave/RestoreV2:6*
T0*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_7Assign,sequential/batch_normalization_1/moving_meansave/RestoreV2:7*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_8Assign0sequential/batch_normalization_1/moving_variancesave/RestoreV2:8*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_9Assign%sequential/batch_normalization_2/betasave/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_10Assign&sequential/batch_normalization_2/gammasave/RestoreV2:10*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_11Assign,sequential/batch_normalization_2/moving_meansave/RestoreV2:11*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_12Assign0sequential/batch_normalization_2/moving_variancesave/RestoreV2:12*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_13Assign%sequential/batch_normalization_3/betasave/RestoreV2:13*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_3/beta*
validate_shape(
?
save/Assign_14Assign&sequential/batch_normalization_3/gammasave/RestoreV2:14*
T0*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_15Assign,sequential/batch_normalization_3/moving_meansave/RestoreV2:15*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_16Assign0sequential/batch_normalization_3/moving_variancesave/RestoreV2:16*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_17Assign%sequential/batch_normalization_4/betasave/RestoreV2:17*
T0*8
_class.
,*loc:@sequential/batch_normalization_4/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_18Assign&sequential/batch_normalization_4/gammasave/RestoreV2:18*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_19Assign,sequential/batch_normalization_4/moving_meansave/RestoreV2:19*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean
?
save/Assign_20Assign0sequential/batch_normalization_4/moving_variancesave/RestoreV2:20*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance*
validate_shape(
?
save/Assign_21Assign%sequential/batch_normalization_5/betasave/RestoreV2:21*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_22Assign&sequential/batch_normalization_5/gammasave/RestoreV2:22*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_23Assign,sequential/batch_normalization_5/moving_meansave/RestoreV2:23*
T0*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_24Assign0sequential/batch_normalization_5/moving_variancesave/RestoreV2:24*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_25Assign%sequential/batch_normalization_6/betasave/RestoreV2:25*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_26Assign&sequential/batch_normalization_6/gammasave/RestoreV2:26*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_6/gamma
?
save/Assign_27Assign,sequential/batch_normalization_6/moving_meansave/RestoreV2:27*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean
?
save/Assign_28Assign0sequential/batch_normalization_6/moving_variancesave/RestoreV2:28*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_29Assign%sequential/batch_normalization_7/betasave/RestoreV2:29*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_30Assign&sequential/batch_normalization_7/gammasave/RestoreV2:30*
T0*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_31Assign,sequential/batch_normalization_7/moving_meansave/RestoreV2:31*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean
?
save/Assign_32Assign0sequential/batch_normalization_7/moving_variancesave/RestoreV2:32*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance
?
save/Assign_33Assign%sequential/batch_normalization_8/betasave/RestoreV2:33*
T0*8
_class.
,*loc:@sequential/batch_normalization_8/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_34Assign&sequential/batch_normalization_8/gammasave/RestoreV2:34*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_8/gamma*
validate_shape(
?
save/Assign_35Assign,sequential/batch_normalization_8/moving_meansave/RestoreV2:35*
T0*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_36Assign0sequential/batch_normalization_8/moving_variancesave/RestoreV2:36*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance
?
save/Assign_37Assign%sequential/batch_normalization_9/betasave/RestoreV2:37*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_9/beta
?
save/Assign_38Assign&sequential/batch_normalization_9/gammasave/RestoreV2:38*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
validate_shape(
?
save/Assign_39Assign,sequential/batch_normalization_9/moving_meansave/RestoreV2:39*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_40Assign0sequential/batch_normalization_9/moving_variancesave/RestoreV2:40*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_41Assignsequential/linear/bsave/RestoreV2:41*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0*&
_class
loc:@sequential/linear/b
?
save/Assign_42Assignsequential/linear/wsave/RestoreV2:42*
use_locking(*
T0*&
_class
loc:@sequential/linear/w*
validate_shape(* 
_output_shapes
:
??'
?
save/Assign_43Assignsequential/linear_1/bsave/RestoreV2:43*
use_locking(*
T0*(
_class
loc:@sequential/linear_1/b*
validate_shape(*
_output_shapes	
:?
?
save/Assign_44Assignsequential/linear_1/wsave/RestoreV2:44*
use_locking(*
T0*(
_class
loc:@sequential/linear_1/w*
validate_shape(* 
_output_shapes
:
?'?
?
save/Assign_45Assignsequential/linear_10/bsave/RestoreV2:45*
T0*)
_class
loc:@sequential/linear_10/b*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_46Assignsequential/linear_10/wsave/RestoreV2:46*
use_locking(*
T0*)
_class
loc:@sequential/linear_10/w*
validate_shape(*
_output_shapes
:	?
?
save/Assign_47Assignsequential/linear_2/bsave/RestoreV2:47*
T0*(
_class
loc:@sequential/linear_2/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_48Assignsequential/linear_2/wsave/RestoreV2:48*
T0*(
_class
loc:@sequential/linear_2/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_49Assignsequential/linear_3/bsave/RestoreV2:49*
T0*(
_class
loc:@sequential/linear_3/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_50Assignsequential/linear_3/wsave/RestoreV2:50*
use_locking(*
T0*(
_class
loc:@sequential/linear_3/w*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_51Assignsequential/linear_4/bsave/RestoreV2:51*
T0*(
_class
loc:@sequential/linear_4/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_52Assignsequential/linear_4/wsave/RestoreV2:52*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*(
_class
loc:@sequential/linear_4/w
?
save/Assign_53Assignsequential/linear_5/bsave/RestoreV2:53*
T0*(
_class
loc:@sequential/linear_5/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_54Assignsequential/linear_5/wsave/RestoreV2:54*
use_locking(*
T0*(
_class
loc:@sequential/linear_5/w*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_55Assignsequential/linear_6/bsave/RestoreV2:55*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*(
_class
loc:@sequential/linear_6/b
?
save/Assign_56Assignsequential/linear_6/wsave/RestoreV2:56*
use_locking(*
T0*(
_class
loc:@sequential/linear_6/w*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_57Assignsequential/linear_7/bsave/RestoreV2:57*
use_locking(*
T0*(
_class
loc:@sequential/linear_7/b*
validate_shape(*
_output_shapes	
:?
?
save/Assign_58Assignsequential/linear_7/wsave/RestoreV2:58*
use_locking(*
T0*(
_class
loc:@sequential/linear_7/w*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_59Assignsequential/linear_8/bsave/RestoreV2:59*
use_locking(*
T0*(
_class
loc:@sequential/linear_8/b*
validate_shape(*
_output_shapes	
:?
?
save/Assign_60Assignsequential/linear_8/wsave/RestoreV2:60*(
_class
loc:@sequential/linear_8/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_61Assignsequential/linear_9/bsave/RestoreV2:61*
use_locking(*
T0*(
_class
loc:@sequential/linear_9/b*
validate_shape(*
_output_shapes	
:?
?
save/Assign_62Assignsequential/linear_9/wsave/RestoreV2:62*
use_locking(*
T0*(
_class
loc:@sequential/linear_9/w*
validate_shape(* 
_output_shapes
:
??
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_881b8db030d34c5f953fcbe2cb9ba825/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*?
value?B??Bglobal_stepB#sequential/batch_normalization/betaB$sequential/batch_normalization/gammaB*sequential/batch_normalization/moving_meanB.sequential/batch_normalization/moving_varianceB%sequential/batch_normalization_1/betaB&sequential/batch_normalization_1/gammaB,sequential/batch_normalization_1/moving_meanB0sequential/batch_normalization_1/moving_varianceB%sequential/batch_normalization_2/betaB&sequential/batch_normalization_2/gammaB,sequential/batch_normalization_2/moving_meanB0sequential/batch_normalization_2/moving_varianceB%sequential/batch_normalization_3/betaB&sequential/batch_normalization_3/gammaB,sequential/batch_normalization_3/moving_meanB0sequential/batch_normalization_3/moving_varianceB%sequential/batch_normalization_4/betaB&sequential/batch_normalization_4/gammaB,sequential/batch_normalization_4/moving_meanB0sequential/batch_normalization_4/moving_varianceB%sequential/batch_normalization_5/betaB&sequential/batch_normalization_5/gammaB,sequential/batch_normalization_5/moving_meanB0sequential/batch_normalization_5/moving_varianceB%sequential/batch_normalization_6/betaB&sequential/batch_normalization_6/gammaB,sequential/batch_normalization_6/moving_meanB0sequential/batch_normalization_6/moving_varianceB%sequential/batch_normalization_7/betaB&sequential/batch_normalization_7/gammaB,sequential/batch_normalization_7/moving_meanB0sequential/batch_normalization_7/moving_varianceB%sequential/batch_normalization_8/betaB&sequential/batch_normalization_8/gammaB,sequential/batch_normalization_8/moving_meanB0sequential/batch_normalization_8/moving_varianceB%sequential/batch_normalization_9/betaB&sequential/batch_normalization_9/gammaB,sequential/batch_normalization_9/moving_meanB0sequential/batch_normalization_9/moving_varianceBsequential/linear/bBsequential/linear/wBsequential/linear_1/bBsequential/linear_1/wBsequential/linear_10/bBsequential/linear_10/wBsequential/linear_2/bBsequential/linear_2/wBsequential/linear_3/bBsequential/linear_3/wBsequential/linear_4/bBsequential/linear_4/wBsequential/linear_5/bBsequential/linear_5/wBsequential/linear_6/bBsequential/linear_6/wBsequential/linear_7/bBsequential/linear_7/wBsequential/linear_8/bBsequential/linear_8/wBsequential/linear_9/bBsequential/linear_9/w*
dtype0
?
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesglobal_step#sequential/batch_normalization/beta$sequential/batch_normalization/gamma*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variance%sequential/batch_normalization_1/beta&sequential/batch_normalization_1/gamma,sequential/batch_normalization_1/moving_mean0sequential/batch_normalization_1/moving_variance%sequential/batch_normalization_2/beta&sequential/batch_normalization_2/gamma,sequential/batch_normalization_2/moving_mean0sequential/batch_normalization_2/moving_variance%sequential/batch_normalization_3/beta&sequential/batch_normalization_3/gamma,sequential/batch_normalization_3/moving_mean0sequential/batch_normalization_3/moving_variance%sequential/batch_normalization_4/beta&sequential/batch_normalization_4/gamma,sequential/batch_normalization_4/moving_mean0sequential/batch_normalization_4/moving_variance%sequential/batch_normalization_5/beta&sequential/batch_normalization_5/gamma,sequential/batch_normalization_5/moving_mean0sequential/batch_normalization_5/moving_variance%sequential/batch_normalization_6/beta&sequential/batch_normalization_6/gamma,sequential/batch_normalization_6/moving_mean0sequential/batch_normalization_6/moving_variance%sequential/batch_normalization_7/beta&sequential/batch_normalization_7/gamma,sequential/batch_normalization_7/moving_mean0sequential/batch_normalization_7/moving_variance%sequential/batch_normalization_8/beta&sequential/batch_normalization_8/gamma,sequential/batch_normalization_8/moving_mean0sequential/batch_normalization_8/moving_variance%sequential/batch_normalization_9/beta&sequential/batch_normalization_9/gamma,sequential/batch_normalization_9/moving_mean0sequential/batch_normalization_9/moving_variancesequential/linear/bsequential/linear/wsequential/linear_1/bsequential/linear_1/wsequential/linear_10/bsequential/linear_10/wsequential/linear_2/bsequential/linear_2/wsequential/linear_3/bsequential/linear_3/wsequential/linear_4/bsequential/linear_4/wsequential/linear_5/bsequential/linear_5/wsequential/linear_6/bsequential/linear_6/wsequential/linear_7/bsequential/linear_7/wsequential/linear_8/bsequential/linear_8/wsequential/linear_9/bsequential/linear_9/w"/device:CPU:0*M
dtypesC
A2?	
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*)
_class
loc:@save_1/ShardedFilename
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B??Bglobal_stepB#sequential/batch_normalization/betaB$sequential/batch_normalization/gammaB*sequential/batch_normalization/moving_meanB.sequential/batch_normalization/moving_varianceB%sequential/batch_normalization_1/betaB&sequential/batch_normalization_1/gammaB,sequential/batch_normalization_1/moving_meanB0sequential/batch_normalization_1/moving_varianceB%sequential/batch_normalization_2/betaB&sequential/batch_normalization_2/gammaB,sequential/batch_normalization_2/moving_meanB0sequential/batch_normalization_2/moving_varianceB%sequential/batch_normalization_3/betaB&sequential/batch_normalization_3/gammaB,sequential/batch_normalization_3/moving_meanB0sequential/batch_normalization_3/moving_varianceB%sequential/batch_normalization_4/betaB&sequential/batch_normalization_4/gammaB,sequential/batch_normalization_4/moving_meanB0sequential/batch_normalization_4/moving_varianceB%sequential/batch_normalization_5/betaB&sequential/batch_normalization_5/gammaB,sequential/batch_normalization_5/moving_meanB0sequential/batch_normalization_5/moving_varianceB%sequential/batch_normalization_6/betaB&sequential/batch_normalization_6/gammaB,sequential/batch_normalization_6/moving_meanB0sequential/batch_normalization_6/moving_varianceB%sequential/batch_normalization_7/betaB&sequential/batch_normalization_7/gammaB,sequential/batch_normalization_7/moving_meanB0sequential/batch_normalization_7/moving_varianceB%sequential/batch_normalization_8/betaB&sequential/batch_normalization_8/gammaB,sequential/batch_normalization_8/moving_meanB0sequential/batch_normalization_8/moving_varianceB%sequential/batch_normalization_9/betaB&sequential/batch_normalization_9/gammaB,sequential/batch_normalization_9/moving_meanB0sequential/batch_normalization_9/moving_varianceBsequential/linear/bBsequential/linear/wBsequential/linear_1/bBsequential/linear_1/wBsequential/linear_10/bBsequential/linear_10/wBsequential/linear_2/bBsequential/linear_2/wBsequential/linear_3/bBsequential/linear_3/wBsequential/linear_4/bBsequential/linear_4/wBsequential/linear_5/bBsequential/linear_5/wBsequential/linear_6/bBsequential/linear_6/wBsequential/linear_7/bBsequential/linear_7/wBsequential/linear_8/bBsequential/linear_8/wBsequential/linear_9/bBsequential/linear_9/w*
dtype0*
_output_shapes
:?
?
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*M
dtypesC
A2?	*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_1/AssignAssignglobal_stepsave_1/RestoreV2*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_1Assign#sequential/batch_normalization/betasave_1/RestoreV2:1*
use_locking(*
T0*6
_class,
*(loc:@sequential/batch_normalization/beta*
validate_shape(*
_output_shapes	
:?'
?
save_1/Assign_2Assign$sequential/batch_normalization/gammasave_1/RestoreV2:2*
_output_shapes	
:?'*
use_locking(*
T0*7
_class-
+)loc:@sequential/batch_normalization/gamma*
validate_shape(
?
save_1/Assign_3Assign*sequential/batch_normalization/moving_meansave_1/RestoreV2:3*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0*=
_class3
1/loc:@sequential/batch_normalization/moving_mean
?
save_1/Assign_4Assign.sequential/batch_normalization/moving_variancesave_1/RestoreV2:4*A
_class7
53loc:@sequential/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0
?
save_1/Assign_5Assign%sequential/batch_normalization_1/betasave_1/RestoreV2:5*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_6Assign&sequential/batch_normalization_1/gammasave_1/RestoreV2:6*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_7Assign,sequential/batch_normalization_1/moving_meansave_1/RestoreV2:7*
T0*?
_class5
31loc:@sequential/batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_8Assign0sequential/batch_normalization_1/moving_variancesave_1/RestoreV2:8*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_1/moving_variance*
validate_shape(
?
save_1/Assign_9Assign%sequential/batch_normalization_2/betasave_1/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_10Assign&sequential/batch_normalization_2/gammasave_1/RestoreV2:10*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_2/gamma
?
save_1/Assign_11Assign,sequential/batch_normalization_2/moving_meansave_1/RestoreV2:11*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_2/moving_mean
?
save_1/Assign_12Assign0sequential/batch_normalization_2/moving_variancesave_1/RestoreV2:12*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_2/moving_variance*
validate_shape(
?
save_1/Assign_13Assign%sequential/batch_normalization_3/betasave_1/RestoreV2:13*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_14Assign&sequential/batch_normalization_3/gammasave_1/RestoreV2:14*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_15Assign,sequential/batch_normalization_3/moving_meansave_1/RestoreV2:15*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_3/moving_mean
?
save_1/Assign_16Assign0sequential/batch_normalization_3/moving_variancesave_1/RestoreV2:16*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_17Assign%sequential/batch_normalization_4/betasave_1/RestoreV2:17*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_4/beta
?
save_1/Assign_18Assign&sequential/batch_normalization_4/gammasave_1/RestoreV2:18*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_19Assign,sequential/batch_normalization_4/moving_meansave_1/RestoreV2:19*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_20Assign0sequential/batch_normalization_4/moving_variancesave_1/RestoreV2:20*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_4/moving_variance
?
save_1/Assign_21Assign%sequential/batch_normalization_5/betasave_1/RestoreV2:21*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_22Assign&sequential/batch_normalization_5/gammasave_1/RestoreV2:22*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_23Assign,sequential/batch_normalization_5/moving_meansave_1/RestoreV2:23*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_24Assign0sequential/batch_normalization_5/moving_variancesave_1/RestoreV2:24*
T0*C
_class9
75loc:@sequential/batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_25Assign%sequential/batch_normalization_6/betasave_1/RestoreV2:25*
T0*8
_class.
,*loc:@sequential/batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_26Assign&sequential/batch_normalization_6/gammasave_1/RestoreV2:26*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_27Assign,sequential/batch_normalization_6/moving_meansave_1/RestoreV2:27*
T0*?
_class5
31loc:@sequential/batch_normalization_6/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_28Assign0sequential/batch_normalization_6/moving_variancesave_1/RestoreV2:28*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_6/moving_variance*
validate_shape(
?
save_1/Assign_29Assign%sequential/batch_normalization_7/betasave_1/RestoreV2:29*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_7/beta
?
save_1/Assign_30Assign&sequential/batch_normalization_7/gammasave_1/RestoreV2:30*
T0*9
_class/
-+loc:@sequential/batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_31Assign,sequential/batch_normalization_7/moving_meansave_1/RestoreV2:31*?
_class5
31loc:@sequential/batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_32Assign0sequential/batch_normalization_7/moving_variancesave_1/RestoreV2:32*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_33Assign%sequential/batch_normalization_8/betasave_1/RestoreV2:33*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_8/beta
?
save_1/Assign_34Assign&sequential/batch_normalization_8/gammasave_1/RestoreV2:34*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@sequential/batch_normalization_8/gamma
?
save_1/Assign_35Assign,sequential/batch_normalization_8/moving_meansave_1/RestoreV2:35*
use_locking(*
T0*?
_class5
31loc:@sequential/batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_36Assign0sequential/batch_normalization_8/moving_variancesave_1/RestoreV2:36*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_37Assign%sequential/batch_normalization_9/betasave_1/RestoreV2:37*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@sequential/batch_normalization_9/beta
?
save_1/Assign_38Assign&sequential/batch_normalization_9/gammasave_1/RestoreV2:38*
T0*9
_class/
-+loc:@sequential/batch_normalization_9/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_39Assign,sequential/batch_normalization_9/moving_meansave_1/RestoreV2:39*
T0*?
_class5
31loc:@sequential/batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_40Assign0sequential/batch_normalization_9/moving_variancesave_1/RestoreV2:40*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@sequential/batch_normalization_9/moving_variance
?
save_1/Assign_41Assignsequential/linear/bsave_1/RestoreV2:41*&
_class
loc:@sequential/linear/b*
validate_shape(*
_output_shapes	
:?'*
use_locking(*
T0
?
save_1/Assign_42Assignsequential/linear/wsave_1/RestoreV2:42*
use_locking(*
T0*&
_class
loc:@sequential/linear/w*
validate_shape(* 
_output_shapes
:
??'
?
save_1/Assign_43Assignsequential/linear_1/bsave_1/RestoreV2:43*
use_locking(*
T0*(
_class
loc:@sequential/linear_1/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_44Assignsequential/linear_1/wsave_1/RestoreV2:44*
use_locking(*
T0*(
_class
loc:@sequential/linear_1/w*
validate_shape(* 
_output_shapes
:
?'?
?
save_1/Assign_45Assignsequential/linear_10/bsave_1/RestoreV2:45*
use_locking(*
T0*)
_class
loc:@sequential/linear_10/b*
validate_shape(*
_output_shapes
:
?
save_1/Assign_46Assignsequential/linear_10/wsave_1/RestoreV2:46*
use_locking(*
T0*)
_class
loc:@sequential/linear_10/w*
validate_shape(*
_output_shapes
:	?
?
save_1/Assign_47Assignsequential/linear_2/bsave_1/RestoreV2:47*
use_locking(*
T0*(
_class
loc:@sequential/linear_2/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_48Assignsequential/linear_2/wsave_1/RestoreV2:48*
use_locking(*
T0*(
_class
loc:@sequential/linear_2/w*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_49Assignsequential/linear_3/bsave_1/RestoreV2:49*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*(
_class
loc:@sequential/linear_3/b
?
save_1/Assign_50Assignsequential/linear_3/wsave_1/RestoreV2:50*
use_locking(*
T0*(
_class
loc:@sequential/linear_3/w*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_51Assignsequential/linear_4/bsave_1/RestoreV2:51*
use_locking(*
T0*(
_class
loc:@sequential/linear_4/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_52Assignsequential/linear_4/wsave_1/RestoreV2:52*
use_locking(*
T0*(
_class
loc:@sequential/linear_4/w*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_53Assignsequential/linear_5/bsave_1/RestoreV2:53*
use_locking(*
T0*(
_class
loc:@sequential/linear_5/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_54Assignsequential/linear_5/wsave_1/RestoreV2:54*
use_locking(*
T0*(
_class
loc:@sequential/linear_5/w*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_55Assignsequential/linear_6/bsave_1/RestoreV2:55*
T0*(
_class
loc:@sequential/linear_6/b*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_56Assignsequential/linear_6/wsave_1/RestoreV2:56*
T0*(
_class
loc:@sequential/linear_6/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_57Assignsequential/linear_7/bsave_1/RestoreV2:57*
use_locking(*
T0*(
_class
loc:@sequential/linear_7/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_58Assignsequential/linear_7/wsave_1/RestoreV2:58*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*(
_class
loc:@sequential/linear_7/w
?
save_1/Assign_59Assignsequential/linear_8/bsave_1/RestoreV2:59*
use_locking(*
T0*(
_class
loc:@sequential/linear_8/b*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_60Assignsequential/linear_8/wsave_1/RestoreV2:60*
use_locking(*
T0*(
_class
loc:@sequential/linear_8/w*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_61Assignsequential/linear_9/bsave_1/RestoreV2:61*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*(
_class
loc:@sequential/linear_9/b
?
save_1/Assign_62Assignsequential/linear_9/wsave_1/RestoreV2:62*(
_class
loc:@sequential/linear_9/w*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?	
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"?Z
	variables?Z?Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
?
sequential/linear/w:0sequential/linear/w/Assignsequential/linear/w/read:022sequential/linear/w/Initializer/truncated_normal:0
x
sequential/linear/b:0sequential/linear/b/Assignsequential/linear/b/read:02'sequential/linear/b/Initializer/zeros:0
?
&sequential/batch_normalization/gamma:0+sequential/batch_normalization/gamma/Assign+sequential/batch_normalization/gamma/read:027sequential/batch_normalization/gamma/Initializer/ones:0
?
%sequential/batch_normalization/beta:0*sequential/batch_normalization/beta/Assign*sequential/batch_normalization/beta/read:027sequential/batch_normalization/beta/Initializer/zeros:0
?
,sequential/batch_normalization/moving_mean:01sequential/batch_normalization/moving_mean/Assign1sequential/batch_normalization/moving_mean/read:02>sequential/batch_normalization/moving_mean/Initializer/zeros:0
?
0sequential/batch_normalization/moving_variance:05sequential/batch_normalization/moving_variance/Assign5sequential/batch_normalization/moving_variance/read:02Asequential/batch_normalization/moving_variance/Initializer/ones:0
?
sequential/linear_1/w:0sequential/linear_1/w/Assignsequential/linear_1/w/read:024sequential/linear_1/w/Initializer/truncated_normal:0
?
sequential/linear_1/b:0sequential/linear_1/b/Assignsequential/linear_1/b/read:02)sequential/linear_1/b/Initializer/zeros:0
?
(sequential/batch_normalization_1/gamma:0-sequential/batch_normalization_1/gamma/Assign-sequential/batch_normalization_1/gamma/read:029sequential/batch_normalization_1/gamma/Initializer/ones:0
?
'sequential/batch_normalization_1/beta:0,sequential/batch_normalization_1/beta/Assign,sequential/batch_normalization_1/beta/read:029sequential/batch_normalization_1/beta/Initializer/zeros:0
?
.sequential/batch_normalization_1/moving_mean:03sequential/batch_normalization_1/moving_mean/Assign3sequential/batch_normalization_1/moving_mean/read:02@sequential/batch_normalization_1/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_1/moving_variance:07sequential/batch_normalization_1/moving_variance/Assign7sequential/batch_normalization_1/moving_variance/read:02Csequential/batch_normalization_1/moving_variance/Initializer/ones:0
?
sequential/linear_2/w:0sequential/linear_2/w/Assignsequential/linear_2/w/read:024sequential/linear_2/w/Initializer/truncated_normal:0
?
sequential/linear_2/b:0sequential/linear_2/b/Assignsequential/linear_2/b/read:02)sequential/linear_2/b/Initializer/zeros:0
?
(sequential/batch_normalization_2/gamma:0-sequential/batch_normalization_2/gamma/Assign-sequential/batch_normalization_2/gamma/read:029sequential/batch_normalization_2/gamma/Initializer/ones:0
?
'sequential/batch_normalization_2/beta:0,sequential/batch_normalization_2/beta/Assign,sequential/batch_normalization_2/beta/read:029sequential/batch_normalization_2/beta/Initializer/zeros:0
?
.sequential/batch_normalization_2/moving_mean:03sequential/batch_normalization_2/moving_mean/Assign3sequential/batch_normalization_2/moving_mean/read:02@sequential/batch_normalization_2/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_2/moving_variance:07sequential/batch_normalization_2/moving_variance/Assign7sequential/batch_normalization_2/moving_variance/read:02Csequential/batch_normalization_2/moving_variance/Initializer/ones:0
?
sequential/linear_3/w:0sequential/linear_3/w/Assignsequential/linear_3/w/read:024sequential/linear_3/w/Initializer/truncated_normal:0
?
sequential/linear_3/b:0sequential/linear_3/b/Assignsequential/linear_3/b/read:02)sequential/linear_3/b/Initializer/zeros:0
?
(sequential/batch_normalization_3/gamma:0-sequential/batch_normalization_3/gamma/Assign-sequential/batch_normalization_3/gamma/read:029sequential/batch_normalization_3/gamma/Initializer/ones:0
?
'sequential/batch_normalization_3/beta:0,sequential/batch_normalization_3/beta/Assign,sequential/batch_normalization_3/beta/read:029sequential/batch_normalization_3/beta/Initializer/zeros:0
?
.sequential/batch_normalization_3/moving_mean:03sequential/batch_normalization_3/moving_mean/Assign3sequential/batch_normalization_3/moving_mean/read:02@sequential/batch_normalization_3/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_3/moving_variance:07sequential/batch_normalization_3/moving_variance/Assign7sequential/batch_normalization_3/moving_variance/read:02Csequential/batch_normalization_3/moving_variance/Initializer/ones:0
?
sequential/linear_4/w:0sequential/linear_4/w/Assignsequential/linear_4/w/read:024sequential/linear_4/w/Initializer/truncated_normal:0
?
sequential/linear_4/b:0sequential/linear_4/b/Assignsequential/linear_4/b/read:02)sequential/linear_4/b/Initializer/zeros:0
?
(sequential/batch_normalization_4/gamma:0-sequential/batch_normalization_4/gamma/Assign-sequential/batch_normalization_4/gamma/read:029sequential/batch_normalization_4/gamma/Initializer/ones:0
?
'sequential/batch_normalization_4/beta:0,sequential/batch_normalization_4/beta/Assign,sequential/batch_normalization_4/beta/read:029sequential/batch_normalization_4/beta/Initializer/zeros:0
?
.sequential/batch_normalization_4/moving_mean:03sequential/batch_normalization_4/moving_mean/Assign3sequential/batch_normalization_4/moving_mean/read:02@sequential/batch_normalization_4/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_4/moving_variance:07sequential/batch_normalization_4/moving_variance/Assign7sequential/batch_normalization_4/moving_variance/read:02Csequential/batch_normalization_4/moving_variance/Initializer/ones:0
?
sequential/linear_5/w:0sequential/linear_5/w/Assignsequential/linear_5/w/read:024sequential/linear_5/w/Initializer/truncated_normal:0
?
sequential/linear_5/b:0sequential/linear_5/b/Assignsequential/linear_5/b/read:02)sequential/linear_5/b/Initializer/zeros:0
?
(sequential/batch_normalization_5/gamma:0-sequential/batch_normalization_5/gamma/Assign-sequential/batch_normalization_5/gamma/read:029sequential/batch_normalization_5/gamma/Initializer/ones:0
?
'sequential/batch_normalization_5/beta:0,sequential/batch_normalization_5/beta/Assign,sequential/batch_normalization_5/beta/read:029sequential/batch_normalization_5/beta/Initializer/zeros:0
?
.sequential/batch_normalization_5/moving_mean:03sequential/batch_normalization_5/moving_mean/Assign3sequential/batch_normalization_5/moving_mean/read:02@sequential/batch_normalization_5/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_5/moving_variance:07sequential/batch_normalization_5/moving_variance/Assign7sequential/batch_normalization_5/moving_variance/read:02Csequential/batch_normalization_5/moving_variance/Initializer/ones:0
?
sequential/linear_6/w:0sequential/linear_6/w/Assignsequential/linear_6/w/read:024sequential/linear_6/w/Initializer/truncated_normal:0
?
sequential/linear_6/b:0sequential/linear_6/b/Assignsequential/linear_6/b/read:02)sequential/linear_6/b/Initializer/zeros:0
?
(sequential/batch_normalization_6/gamma:0-sequential/batch_normalization_6/gamma/Assign-sequential/batch_normalization_6/gamma/read:029sequential/batch_normalization_6/gamma/Initializer/ones:0
?
'sequential/batch_normalization_6/beta:0,sequential/batch_normalization_6/beta/Assign,sequential/batch_normalization_6/beta/read:029sequential/batch_normalization_6/beta/Initializer/zeros:0
?
.sequential/batch_normalization_6/moving_mean:03sequential/batch_normalization_6/moving_mean/Assign3sequential/batch_normalization_6/moving_mean/read:02@sequential/batch_normalization_6/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_6/moving_variance:07sequential/batch_normalization_6/moving_variance/Assign7sequential/batch_normalization_6/moving_variance/read:02Csequential/batch_normalization_6/moving_variance/Initializer/ones:0
?
sequential/linear_7/w:0sequential/linear_7/w/Assignsequential/linear_7/w/read:024sequential/linear_7/w/Initializer/truncated_normal:0
?
sequential/linear_7/b:0sequential/linear_7/b/Assignsequential/linear_7/b/read:02)sequential/linear_7/b/Initializer/zeros:0
?
(sequential/batch_normalization_7/gamma:0-sequential/batch_normalization_7/gamma/Assign-sequential/batch_normalization_7/gamma/read:029sequential/batch_normalization_7/gamma/Initializer/ones:0
?
'sequential/batch_normalization_7/beta:0,sequential/batch_normalization_7/beta/Assign,sequential/batch_normalization_7/beta/read:029sequential/batch_normalization_7/beta/Initializer/zeros:0
?
.sequential/batch_normalization_7/moving_mean:03sequential/batch_normalization_7/moving_mean/Assign3sequential/batch_normalization_7/moving_mean/read:02@sequential/batch_normalization_7/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_7/moving_variance:07sequential/batch_normalization_7/moving_variance/Assign7sequential/batch_normalization_7/moving_variance/read:02Csequential/batch_normalization_7/moving_variance/Initializer/ones:0
?
sequential/linear_8/w:0sequential/linear_8/w/Assignsequential/linear_8/w/read:024sequential/linear_8/w/Initializer/truncated_normal:0
?
sequential/linear_8/b:0sequential/linear_8/b/Assignsequential/linear_8/b/read:02)sequential/linear_8/b/Initializer/zeros:0
?
(sequential/batch_normalization_8/gamma:0-sequential/batch_normalization_8/gamma/Assign-sequential/batch_normalization_8/gamma/read:029sequential/batch_normalization_8/gamma/Initializer/ones:0
?
'sequential/batch_normalization_8/beta:0,sequential/batch_normalization_8/beta/Assign,sequential/batch_normalization_8/beta/read:029sequential/batch_normalization_8/beta/Initializer/zeros:0
?
.sequential/batch_normalization_8/moving_mean:03sequential/batch_normalization_8/moving_mean/Assign3sequential/batch_normalization_8/moving_mean/read:02@sequential/batch_normalization_8/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_8/moving_variance:07sequential/batch_normalization_8/moving_variance/Assign7sequential/batch_normalization_8/moving_variance/read:02Csequential/batch_normalization_8/moving_variance/Initializer/ones:0
?
sequential/linear_9/w:0sequential/linear_9/w/Assignsequential/linear_9/w/read:024sequential/linear_9/w/Initializer/truncated_normal:0
?
sequential/linear_9/b:0sequential/linear_9/b/Assignsequential/linear_9/b/read:02)sequential/linear_9/b/Initializer/zeros:0
?
(sequential/batch_normalization_9/gamma:0-sequential/batch_normalization_9/gamma/Assign-sequential/batch_normalization_9/gamma/read:029sequential/batch_normalization_9/gamma/Initializer/ones:0
?
'sequential/batch_normalization_9/beta:0,sequential/batch_normalization_9/beta/Assign,sequential/batch_normalization_9/beta/read:029sequential/batch_normalization_9/beta/Initializer/zeros:0
?
.sequential/batch_normalization_9/moving_mean:03sequential/batch_normalization_9/moving_mean/Assign3sequential/batch_normalization_9/moving_mean/read:02@sequential/batch_normalization_9/moving_mean/Initializer/zeros:0
?
2sequential/batch_normalization_9/moving_variance:07sequential/batch_normalization_9/moving_variance/Assign7sequential/batch_normalization_9/moving_variance/read:02Csequential/batch_normalization_9/moving_variance/Initializer/ones:0
?
sequential/linear_10/w:0sequential/linear_10/w/Assignsequential/linear_10/w/read:025sequential/linear_10/w/Initializer/truncated_normal:0
?
sequential/linear_10/b:0sequential/linear_10/b/Assignsequential/linear_10/b/read:02*sequential/linear_10/b/Initializer/zeros:0" 
legacy_init_op


group_deps"?=
sonnet?=?=
?

sequential
sequentialR

sequential"

args


Cast:0*
(sequential/sequential_24/linear_10/add:0"+sonnet.python.modules.sequential.Sequential
?

sequentialsequential/sequentialW
sequential/sequential"

args


Cast:0$
"sequential/sequential/linear/add:0"+sonnet.python.modules.sequential.Sequential
?
linearsequential/linear\
sequential/sequential/linear"

inputs
Cast:0$
"sequential/sequential/linear/add:0""sonnet.python.modules.basic.Linear
?
sequential_1sequential/sequential_1
sequential/sequential_1B"@
>
args64
2
0sequential/batch_normalization/batchnorm/add_1:0 
sequential/sequential_1/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_2sequential/sequential_2u
sequential/sequential_20".
,
args$"
 
sequential/sequential_1/Relu:0(
&sequential/sequential_2/linear_1/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_1sequential/linear_1|
 sequential/sequential_2/linear_1.",
*
inputs 
sequential/sequential_1/Relu:0(
&sequential/sequential_2/linear_1/add:0""sonnet.python.modules.basic.Linear
?
sequential_3sequential/sequential_3?
sequential/sequential_3D"B
@
args86
4
2sequential/batch_normalization_1/batchnorm/add_1:0 
sequential/sequential_3/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_4sequential/sequential_4u
sequential/sequential_40".
,
args$"
 
sequential/sequential_3/Relu:0(
&sequential/sequential_4/linear_2/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_2sequential/linear_2|
 sequential/sequential_4/linear_2.",
*
inputs 
sequential/sequential_3/Relu:0(
&sequential/sequential_4/linear_2/add:0""sonnet.python.modules.basic.Linear
?
sequential_5sequential/sequential_5?
sequential/sequential_5D"B
@
args86
4
2sequential/batch_normalization_2/batchnorm/add_1:0 
sequential/sequential_5/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_6sequential/sequential_6u
sequential/sequential_60".
,
args$"
 
sequential/sequential_5/Relu:0(
&sequential/sequential_6/linear_3/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_3sequential/linear_3|
 sequential/sequential_6/linear_3.",
*
inputs 
sequential/sequential_5/Relu:0(
&sequential/sequential_6/linear_3/add:0""sonnet.python.modules.basic.Linear
?
sequential_7sequential/sequential_7?
sequential/sequential_7D"B
@
args86
4
2sequential/batch_normalization_3/batchnorm/add_1:04
2sequential/batch_normalization_3/batchnorm/add_1:0"+sonnet.python.modules.sequential.Sequential
?
sequential_8sequential/sequential_8_
sequential/sequential_8"" 

args

sequential/Add:0 
sequential/sequential_8/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_9sequential/sequential_9u
sequential/sequential_90".
,
args$"
 
sequential/sequential_8/Relu:0(
&sequential/sequential_9/linear_4/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_4sequential/linear_4|
 sequential/sequential_9/linear_4.",
*
inputs 
sequential/sequential_8/Relu:0(
&sequential/sequential_9/linear_4/add:0""sonnet.python.modules.basic.Linear
?
sequential_10sequential/sequential_10?
sequential/sequential_10D"B
@
args86
4
2sequential/batch_normalization_4/batchnorm/add_1:0!
sequential/sequential_10/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_11sequential/sequential_11x
sequential/sequential_111"/
-
args%#
!
sequential/sequential_10/Relu:0)
'sequential/sequential_11/linear_5/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_5sequential/linear_5
!sequential/sequential_11/linear_5/"-
+
inputs!
sequential/sequential_10/Relu:0)
'sequential/sequential_11/linear_5/add:0""sonnet.python.modules.basic.Linear
?
sequential_12sequential/sequential_12?
sequential/sequential_12D"B
@
args86
4
2sequential/batch_normalization_5/batchnorm/add_1:04
2sequential/batch_normalization_5/batchnorm/add_1:0"+sonnet.python.modules.sequential.Sequential
?
sequential_13sequential/sequential_13c
sequential/sequential_13$""
 
args

sequential/Add_1:0!
sequential/sequential_13/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_14sequential/sequential_14x
sequential/sequential_141"/
-
args%#
!
sequential/sequential_13/Relu:0)
'sequential/sequential_14/linear_6/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_6sequential/linear_6
!sequential/sequential_14/linear_6/"-
+
inputs!
sequential/sequential_13/Relu:0)
'sequential/sequential_14/linear_6/add:0""sonnet.python.modules.basic.Linear
?
sequential_15sequential/sequential_15?
sequential/sequential_15D"B
@
args86
4
2sequential/batch_normalization_6/batchnorm/add_1:0!
sequential/sequential_15/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_16sequential/sequential_16x
sequential/sequential_161"/
-
args%#
!
sequential/sequential_15/Relu:0)
'sequential/sequential_16/linear_7/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_7sequential/linear_7
!sequential/sequential_16/linear_7/"-
+
inputs!
sequential/sequential_15/Relu:0)
'sequential/sequential_16/linear_7/add:0""sonnet.python.modules.basic.Linear
?
sequential_17sequential/sequential_17?
sequential/sequential_17D"B
@
args86
4
2sequential/batch_normalization_7/batchnorm/add_1:04
2sequential/batch_normalization_7/batchnorm/add_1:0"+sonnet.python.modules.sequential.Sequential
?
sequential_18sequential/sequential_18c
sequential/sequential_18$""
 
args

sequential/Add_2:0!
sequential/sequential_18/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_19sequential/sequential_19x
sequential/sequential_191"/
-
args%#
!
sequential/sequential_18/Relu:0)
'sequential/sequential_19/linear_8/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_8sequential/linear_8
!sequential/sequential_19/linear_8/"-
+
inputs!
sequential/sequential_18/Relu:0)
'sequential/sequential_19/linear_8/add:0""sonnet.python.modules.basic.Linear
?
sequential_20sequential/sequential_20?
sequential/sequential_20D"B
@
args86
4
2sequential/batch_normalization_8/batchnorm/add_1:0!
sequential/sequential_20/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_21sequential/sequential_21x
sequential/sequential_211"/
-
args%#
!
sequential/sequential_20/Relu:0)
'sequential/sequential_21/linear_9/add:0"+sonnet.python.modules.sequential.Sequential
?
linear_9sequential/linear_9
!sequential/sequential_21/linear_9/"-
+
inputs!
sequential/sequential_20/Relu:0)
'sequential/sequential_21/linear_9/add:0""sonnet.python.modules.basic.Linear
?
sequential_22sequential/sequential_22?
sequential/sequential_22D"B
@
args86
4
2sequential/batch_normalization_9/batchnorm/add_1:04
2sequential/batch_normalization_9/batchnorm/add_1:0"+sonnet.python.modules.sequential.Sequential
?
sequential_23sequential/sequential_23c
sequential/sequential_23$""
 
args

sequential/Add_3:0!
sequential/sequential_23/Relu:0"+sonnet.python.modules.sequential.Sequential
?
sequential_24sequential/sequential_24y
sequential/sequential_241"/
-
args%#
!
sequential/sequential_23/Relu:0*
(sequential/sequential_24/linear_10/add:0"+sonnet.python.modules.sequential.Sequential
?
	linear_10sequential/linear_10?
"sequential/sequential_24/linear_10/"-
+
inputs!
sequential/sequential_23/Relu:0*
(sequential/sequential_24/linear_10/add:0""sonnet.python.modules.basic.Linear
?
sequential_25sequential/sequential_25?
sequential/sequential_25:"8
6
args.,
*
(sequential/sequential_24/linear_10/add:0*
(sequential/sequential_24/linear_10/add:0"+sonnet.python.modules.sequential.Sequential"?6
trainable_variables?5?5
?
sequential/linear/w:0sequential/linear/w/Assignsequential/linear/w/read:022sequential/linear/w/Initializer/truncated_normal:0
x
sequential/linear/b:0sequential/linear/b/Assignsequential/linear/b/read:02'sequential/linear/b/Initializer/zeros:0
?
&sequential/batch_normalization/gamma:0+sequential/batch_normalization/gamma/Assign+sequential/batch_normalization/gamma/read:027sequential/batch_normalization/gamma/Initializer/ones:0
?
%sequential/batch_normalization/beta:0*sequential/batch_normalization/beta/Assign*sequential/batch_normalization/beta/read:027sequential/batch_normalization/beta/Initializer/zeros:0
?
sequential/linear_1/w:0sequential/linear_1/w/Assignsequential/linear_1/w/read:024sequential/linear_1/w/Initializer/truncated_normal:0
?
sequential/linear_1/b:0sequential/linear_1/b/Assignsequential/linear_1/b/read:02)sequential/linear_1/b/Initializer/zeros:0
?
(sequential/batch_normalization_1/gamma:0-sequential/batch_normalization_1/gamma/Assign-sequential/batch_normalization_1/gamma/read:029sequential/batch_normalization_1/gamma/Initializer/ones:0
?
'sequential/batch_normalization_1/beta:0,sequential/batch_normalization_1/beta/Assign,sequential/batch_normalization_1/beta/read:029sequential/batch_normalization_1/beta/Initializer/zeros:0
?
sequential/linear_2/w:0sequential/linear_2/w/Assignsequential/linear_2/w/read:024sequential/linear_2/w/Initializer/truncated_normal:0
?
sequential/linear_2/b:0sequential/linear_2/b/Assignsequential/linear_2/b/read:02)sequential/linear_2/b/Initializer/zeros:0
?
(sequential/batch_normalization_2/gamma:0-sequential/batch_normalization_2/gamma/Assign-sequential/batch_normalization_2/gamma/read:029sequential/batch_normalization_2/gamma/Initializer/ones:0
?
'sequential/batch_normalization_2/beta:0,sequential/batch_normalization_2/beta/Assign,sequential/batch_normalization_2/beta/read:029sequential/batch_normalization_2/beta/Initializer/zeros:0
?
sequential/linear_3/w:0sequential/linear_3/w/Assignsequential/linear_3/w/read:024sequential/linear_3/w/Initializer/truncated_normal:0
?
sequential/linear_3/b:0sequential/linear_3/b/Assignsequential/linear_3/b/read:02)sequential/linear_3/b/Initializer/zeros:0
?
(sequential/batch_normalization_3/gamma:0-sequential/batch_normalization_3/gamma/Assign-sequential/batch_normalization_3/gamma/read:029sequential/batch_normalization_3/gamma/Initializer/ones:0
?
'sequential/batch_normalization_3/beta:0,sequential/batch_normalization_3/beta/Assign,sequential/batch_normalization_3/beta/read:029sequential/batch_normalization_3/beta/Initializer/zeros:0
?
sequential/linear_4/w:0sequential/linear_4/w/Assignsequential/linear_4/w/read:024sequential/linear_4/w/Initializer/truncated_normal:0
?
sequential/linear_4/b:0sequential/linear_4/b/Assignsequential/linear_4/b/read:02)sequential/linear_4/b/Initializer/zeros:0
?
(sequential/batch_normalization_4/gamma:0-sequential/batch_normalization_4/gamma/Assign-sequential/batch_normalization_4/gamma/read:029sequential/batch_normalization_4/gamma/Initializer/ones:0
?
'sequential/batch_normalization_4/beta:0,sequential/batch_normalization_4/beta/Assign,sequential/batch_normalization_4/beta/read:029sequential/batch_normalization_4/beta/Initializer/zeros:0
?
sequential/linear_5/w:0sequential/linear_5/w/Assignsequential/linear_5/w/read:024sequential/linear_5/w/Initializer/truncated_normal:0
?
sequential/linear_5/b:0sequential/linear_5/b/Assignsequential/linear_5/b/read:02)sequential/linear_5/b/Initializer/zeros:0
?
(sequential/batch_normalization_5/gamma:0-sequential/batch_normalization_5/gamma/Assign-sequential/batch_normalization_5/gamma/read:029sequential/batch_normalization_5/gamma/Initializer/ones:0
?
'sequential/batch_normalization_5/beta:0,sequential/batch_normalization_5/beta/Assign,sequential/batch_normalization_5/beta/read:029sequential/batch_normalization_5/beta/Initializer/zeros:0
?
sequential/linear_6/w:0sequential/linear_6/w/Assignsequential/linear_6/w/read:024sequential/linear_6/w/Initializer/truncated_normal:0
?
sequential/linear_6/b:0sequential/linear_6/b/Assignsequential/linear_6/b/read:02)sequential/linear_6/b/Initializer/zeros:0
?
(sequential/batch_normalization_6/gamma:0-sequential/batch_normalization_6/gamma/Assign-sequential/batch_normalization_6/gamma/read:029sequential/batch_normalization_6/gamma/Initializer/ones:0
?
'sequential/batch_normalization_6/beta:0,sequential/batch_normalization_6/beta/Assign,sequential/batch_normalization_6/beta/read:029sequential/batch_normalization_6/beta/Initializer/zeros:0
?
sequential/linear_7/w:0sequential/linear_7/w/Assignsequential/linear_7/w/read:024sequential/linear_7/w/Initializer/truncated_normal:0
?
sequential/linear_7/b:0sequential/linear_7/b/Assignsequential/linear_7/b/read:02)sequential/linear_7/b/Initializer/zeros:0
?
(sequential/batch_normalization_7/gamma:0-sequential/batch_normalization_7/gamma/Assign-sequential/batch_normalization_7/gamma/read:029sequential/batch_normalization_7/gamma/Initializer/ones:0
?
'sequential/batch_normalization_7/beta:0,sequential/batch_normalization_7/beta/Assign,sequential/batch_normalization_7/beta/read:029sequential/batch_normalization_7/beta/Initializer/zeros:0
?
sequential/linear_8/w:0sequential/linear_8/w/Assignsequential/linear_8/w/read:024sequential/linear_8/w/Initializer/truncated_normal:0
?
sequential/linear_8/b:0sequential/linear_8/b/Assignsequential/linear_8/b/read:02)sequential/linear_8/b/Initializer/zeros:0
?
(sequential/batch_normalization_8/gamma:0-sequential/batch_normalization_8/gamma/Assign-sequential/batch_normalization_8/gamma/read:029sequential/batch_normalization_8/gamma/Initializer/ones:0
?
'sequential/batch_normalization_8/beta:0,sequential/batch_normalization_8/beta/Assign,sequential/batch_normalization_8/beta/read:029sequential/batch_normalization_8/beta/Initializer/zeros:0
?
sequential/linear_9/w:0sequential/linear_9/w/Assignsequential/linear_9/w/read:024sequential/linear_9/w/Initializer/truncated_normal:0
?
sequential/linear_9/b:0sequential/linear_9/b/Assignsequential/linear_9/b/read:02)sequential/linear_9/b/Initializer/zeros:0
?
(sequential/batch_normalization_9/gamma:0-sequential/batch_normalization_9/gamma/Assign-sequential/batch_normalization_9/gamma/read:029sequential/batch_normalization_9/gamma/Initializer/ones:0
?
'sequential/batch_normalization_9/beta:0,sequential/batch_normalization_9/beta/Assign,sequential/batch_normalization_9/beta/read:029sequential/batch_normalization_9/beta/Initializer/zeros:0
?
sequential/linear_10/w:0sequential/linear_10/w/Assignsequential/linear_10/w/read:025sequential/linear_10/w/Initializer/truncated_normal:0
?
sequential/linear_10/b:0sequential/linear_10/b/Assignsequential/linear_10/b/read:02*sequential/linear_10/b/Initializer/zeros:0*?
y?
-
x(
Placeholder:0?????????6I
output?
(sequential/sequential_24/linear_10/add:0?????????tensorflow/serving/predict*?
serving_default?
-
x(
Placeholder:0?????????6I
output?
(sequential/sequential_24/linear_10/add:0?????????tensorflow/serving/predict