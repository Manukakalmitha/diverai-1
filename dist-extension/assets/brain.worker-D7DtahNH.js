(function(){"use strict";function Wy(n,t){return t.forEach(function(e){e&&typeof e!="string"&&!Array.isArray(e)&&Object.keys(e).forEach(function(s){if(s!=="default"&&!(s in n)){var o=Object.getOwnPropertyDescriptor(e,s);Object.defineProperty(n,s,o.get?o:{enumerable:!0,get:function(){return e[s]}})}})}),Object.freeze(n)}const Uy=1e-7,Gy=1e-4;class gp{constructor(t,e){this.backend=t,this.dataMover=e,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,e){this.dataIdsCount++,this.data.set(t,e)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class $c{refCount(t){return We("refCount")}incRef(t){return We("incRef")}timerAvailable(){return!0}time(t){return We("time")}read(t){return We("read")}readSync(t){return We("readSync")}readToGPU(t,e){return We("readToGPU")}numDataIds(){return We("numDataIds")}disposeData(t,e){return We("disposeData")}write(t,e,s){return We("write")}move(t,e,s,o,r){return We("move")}createTensorFromGPUData(t,e,s){return We("createTensorFromGPUData")}memory(){return We("memory")}floatPrecision(){return We("floatPrecision")}epsilon(){return this.floatPrecision()===32?Uy:Gy}dispose(){return We("dispose")}}function We(n){throw new Error(`'${n}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}function Hy(n){let t=n.length,e=0;for(;t>0;)e=Math.random()*t|0,t--,go(n,t,e)}function As(n,t,e){return Math.max(n,Math.min(t,e))}function kc(n){return n%2===0?n:n+1}function go(n,t,e){const s=n[t];n[t]=n[e],n[e]=s}function qy(n){let t=0;for(let e=0;e<n.length;e++)t+=n[e];return t}function S(n,t){if(!n)throw new Error(typeof t=="string"?t:t())}function vc(n,t,e=""){S(Et(n,t),()=>e+` Shapes ${n} and ${t} must match`)}function Sc(n){S(n!=null,()=>"The input to the tensor constructor must be a non-null value.")}function q(n){if(n.length===0)return 1;let t=n[0];for(let e=1;e<n.length;e++)t*=n[e];return t}function Et(n,t){if(n===t)return!0;if(n==null||t==null||n.length!==t.length)return!1;for(let e=0;e<n.length;e++)if(n[e]!==t[e])return!1;return!0}function xo(n){return n%1===0}function Nc(n){const t=Math.ceil(Math.sqrt(n));return[t,Math.ceil(n/t)]}function bo(n,t){return t<=n.length?n:n+" ".repeat(t-n.length)}function xp(n,t=o=>0,e,s){return new Promise((o,r)=>{let i=0;const a=()=>{if(n()){o();return}i++;const l=t(i);if(e!=null&&i>=e){r();return}s!=null?s(a,l):setTimeout(a,l)};a()})}function bp(n,t){let e=1,s=-1;for(let r=0;r<n.length;++r)if(n[r]>=0)e*=n[r];else if(n[r]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${r}`);s=r}else if(n[r]<0)throw Error(`Shapes can not be < 0. Found ${n[r]} at dim ${r}`);if(s===-1){if(t>0&&t!==e)throw Error(`Size(${t}) must match the product of shape ${n}`);return n}if(e===0)throw Error(`Cannot infer the missing size in [${n}] when there are 0 elements`);if(t%e!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${e}`);const o=n.slice();return o[s]=t/e,o}function yt(n,t){const e=t.length;return n=n==null?t.map((s,o)=>o):[].concat(n),S(n.every(s=>s>=-e&&s<e),()=>`All values in axis param must be in range [-${e}, ${e}) but got axis ${n}`),S(n.every(s=>xo(s)),()=>`All values in axis param must be integers but got axis ${n}`),n.map(s=>s<0?e+s:s)}function ss(n,t){const e=[],s=[],o=t!=null&&Array.isArray(t)&&t.length===0,r=t==null||o?null:yt(t,n).sort();let i=0;for(let a=0;a<n.length;++a){if(r!=null){if(r[i]===a&&n[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${n[a]}' is not 1`);(r[i]==null||r[i]>a)&&n[a]===1&&(e.push(n[a]),s.push(a)),r[i]<=a&&i++}n[a]!==1&&(e.push(n[a]),s.push(a))}return{newShape:e,keptDims:s}}function Ce(n,t){return Yt(n,t)}function Yt(n,t){let e=null;if(n==null||n==="float32")e=new Float32Array(t);else if(n==="int32")e=new Int32Array(t);else if(n==="bool")e=new Uint8Array(t);else if(n==="string")e=new Array(t);else throw new Error(`Unknown data type ${n}`);return e}function Ky(n,t){for(let e=0;e<n.length;e++){const s=n[e];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function jy(n){return n==="bool"||n==="complex64"||n==="float32"||n==="int32"||n==="string"}function yp(n,t){return!(t==="complex64"||t==="float32"&&n!=="complex64"||t==="int32"&&n!=="float32"&&n!=="complex64"||t==="bool"&&n==="bool")}function Yi(n){if(n==="float32"||n==="int32")return 4;if(n==="complex64")return 8;if(n==="bool")return 1;throw new Error(`Unknown dtype ${n}`)}function Xy(n){if(n==null)return 0;let t=0;return n.forEach(e=>t+=e.length),t}function er(n){return typeof n=="string"||n instanceof String}function Yy(n){return typeof n=="boolean"}function Tc(n){return typeof n=="number"}function yo(n){return Array.isArray(n)?yo(n[0]):n instanceof Float32Array?"float32":n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray?"int32":Tc(n)?"float32":er(n)?"string":Yy(n)?"bool":"float32"}function Ec(n){return!!(n&&n.constructor&&n.call&&n.apply)}function Rc(n,t){for(let e=t;e<n;++e)if(n%e===0)return e;return n}function lt(n){const t=n.length;if(t<2)return[];const e=new Array(t-1);e[t-2]=n[t-1];for(let s=t-3;s>=0;--s)e[s]=e[s+1]*n[s+1];return e}function wp(n,t,e,s=!1){const o=new Array;if(t.length===1){const r=t[0]*(s?2:1);for(let i=0;i<r;i++)o[i]=e[n+i]}else{const r=t[0],i=t.slice(1),a=i.reduce((l,c)=>l*c)*(s?2:1);for(let l=0;l<r;l++)o[l]=wp(n+l*a,i,e,s)}return o}function pn(n,t,e=!1){if(n.length===0)return t[0];const s=n.reduce((o,r)=>o*r)*(e?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${n}] does not match the input size ${t.length}${e?" for a complex tensor":""}.`);return wp(0,n,t,e)}function Zy(n,t){if(Array.isArray(n))return n;if(t==="float32")return n instanceof Float32Array?n:new Float32Array(n);if(t==="int32")return n instanceof Int32Array?n:new Int32Array(n);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(n));throw new Error(`Unknown dtype ${t}`)}function Ac(n,t){const e=Ie(n,t);for(let s=0;s<e.length;s++)e[s]=1;return e}function Ie(n,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(n);if(t==="int32")return new Int32Array(n);if(t==="bool")return new Uint8Array(n);throw new Error(`Unknown data type ${t}`)}function Cp(n,t){const e=n.reduce((s,o)=>s*o,1);if(t==null||t==="float32")return pn(n,new Float32Array(e));if(t==="int32")return pn(n,new Int32Array(e));if(t==="bool")return pn(n,new Uint8Array(e));throw new Error(`Unknown data type ${t}`)}function Un(n){n.forEach(t=>{S(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${n}].`)})}function Sn(n,t,e){if(t===0)return 0;if(t===1)return n[0];let s=n[n.length-1];for(let o=0;o<n.length-1;++o)s+=e[o]*n[o];return s}function wo(n,t,e){if(t===0)return[];if(t===1)return[n];const s=new Array(t);for(let o=0;o<s.length-1;++o)s[o]=Math.floor(n/e[o]),n-=s[o]*e[o];return s[s.length-1]=n,s}function Dc(n){return n&&n.then&&typeof n.then=="function"}const Ip="tfjsflags";class Jy{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Qy,this.populateURLFlags()}setPlatform(t,e){this.platform!=null&&(U().getBool("IS_TEST")||U().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=e}registerFlag(t,e,s){if(this.flagRegistry[t]={evaluationFn:e,setHook:s},this.urlFlags[t]!=null){const o=this.urlFlags[t];U().getBool("IS_TEST")||U().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${o}.`),this.set(t,o)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const e=this.evaluateFlag(t);if(Dc(e))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=e,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,e){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=e,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(e)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);Ip in t&&t[Ip].split(",").forEach(s=>{const[o,r]=s.split(":");this.urlFlags[o]=ew(o,r)})}}function Qy(n){const t={};return n.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...s)=>(tw(t,s[0],s[1]),s.join("="))),t}function tw(n,t,e){n[decodeURIComponent(t)]=decodeURIComponent(e||"")}function ew(n,t){const e=t.toLowerCase();return e==="true"||e==="false"?e==="true":`${+e}`===e?+e:t}function U(){return $p}let $p=null;function nw(n){$p=n}let Fc;function kp(){if(Fc==null){let n;if(typeof window<"u")n=window;else if(typeof global<"u")n=global;else if(typeof process<"u")n=process;else if(typeof self<"u")n=self;else throw new Error("Could not find a global object");Fc=n}return Fc}function sw(){const n=kp();return n._tfGlobals==null&&(n._tfGlobals=new Map),n._tfGlobals}function Oc(n,t){const e=sw();if(e.has(n))return e.get(n);{const s=t();return e.set(n,s),e.get(n)}}const Zi="Abs",nr="Acos",sr="Acosh",Co="Add",_c="AddN",Lc="All",Mc="Any",Ji="ArgMax",Qi="ArgMin",or="Asin",rr="Asinh",ir="Atan",ar="Atanh",lr="Atan2",ta="AvgPool",Pc="AvgPoolGrad",ea="AvgPool3D",Bc="AvgPool3DGrad",na="BatchMatMul",sa="BatchToSpaceND",zc="Bincount",Vc="BitwiseAnd",ow="BroadcastTo",vp="BroadcastArgs",cr="Cast",ur="Ceil",hr="ClipByValue",Wc="Complex",oa="ComplexAbs",ra="Concat",ia="Conv2D",Uc="Conv2DBackpropFilter",aa="Conv2DBackpropInput",la="Conv3D",Gc="Conv3DBackpropFilterV2",Hc="Conv3DBackpropInputV2",dr="Cos",pr="Cosh",qc="Cumprod",ca="Cumsum",Kc="CropAndResize",jc="DenseBincount",Xc="DepthToSpace",ua="DepthwiseConv2dNative",Yc="DepthwiseConv2dNativeBackpropFilter",Zc="DepthwiseConv2dNativeBackpropInput",Sp="Diag",ha="Dilation2D",Jc="Dilation2DBackpropInput",Qc="Dilation2DBackpropFilter",rw="Draw",fr="RealDiv",tu="Einsum",mr="Elu",eu="EluGrad",gr="Erf",da="Equal",xr="Exp",pa="ExpandDims",br="Expm1",nu="FFT",su="Fill",ou="FlipLeftRight",yr="Floor",wr="FloorDiv",fa="FusedBatchNorm",ma="GatherV2",Np="GatherNd",ga="Greater",Cr="GreaterEqual",Ir="Identity",ru="IFFT",iu="Imag",$r="IsFinite",kr="IsInf",vr="IsNan",xa="LeakyRelu",ba="Less",ya="LessEqual",Tp="LinSpace",Sr="Log",Nr="Log1p",wa="LogicalAnd",Ca="LogicalNot",Ia="LogicalOr",iw="LogSoftmax",$a="LRN",au="LRNGrad",ka="Max",Tr="Maximum",va="MaxPool",lu="MaxPoolGrad",Sa="MaxPool3D",cu="MaxPool3DGrad",Ep="MaxPoolWithArgmax",Na="Mean",Ta="Min",Er="Minimum",Ea="MirrorPad",Rr="Mod",Rp="Multinomial",Ar="Multiply",Ra="Neg",Aa="NotEqual",uu="NonMaxSuppressionV3",hu="NonMaxSuppressionV4",du="NonMaxSuppressionV5",Da="OnesLike",Fa="OneHot",Oa="Pack",_a="PadV2",Dr="Pow",La="Prelu",Ma="Prod",Ap="RaggedGather",Dp="RaggedRange",Fp="RaggedTensorToTensor",pu="Range",fu="Real",Fr="Reciprocal",Or="Relu",Pa="Reshape",Ba="ResizeNearestNeighbor",mu="ResizeNearestNeighborGrad",za="ResizeBilinear",gu="ResizeBilinearGrad",_r="Relu6",Va="Reverse",Lr="Round",Mr="Rsqrt",Op="ScatterNd",_p="TensorScatterUpdate",Lp="SearchSorted",Wa="Select",Pr="Selu",Ua="Slice",Br="Sin",zr="Sinh",Vr="Sign",Wr="Sigmoid",Ur="Softplus",Gr="Sqrt",Ga="Sum",Ha="SpaceToBatchND",qa="SplitV",Ka="Softmax",Mp="SparseFillEmptyRows",Pp="SparseReshape",Bp="SparseSegmentMean",zp="SparseSegmentSum",Vp="SparseToDense",Hr="SquaredDifference",xu="Square",bu="StaticRegexReplace",yu="StridedSlice",Wp="StringNGrams",Up="StringSplit",Gp="StringToHashBucketFast",qr="Sub",Kr="Tan",jr="Tanh",Xr="Tile",wu="TopK",Cu="Transform",Io="Transpose",Iu="Unique",ja="Unpack",Xa="UnsortedSegmentSum",Ya="ZerosLike",Yr="Step",aw="FromPixels",$u="RotateWithOffset",Za="_FusedMatMul",Ja="FusedConv2D",Hp="FusedDepthwiseConv2D";function je(...n){U().getBool("IS_TEST")||U().getBool("PROD")||console.warn(...n)}function lw(...n){U().getBool("IS_TEST")||U().getBool("PROD")||console.log(...n)}const Qa=Oc("kernelRegistry",()=>new Map),ku=Oc("gradRegistry",()=>new Map);function qp(n,t){const e=Yp(n,t);return Qa.get(e)}function Kp(n){return ku.get(n)}function jp(n){const t=Qa.entries(),e=[];for(;;){const{done:s,value:o}=t.next();if(s)break;const[r,i]=o,[a]=r.split("_");a===n&&e.push(i)}return e}function Xp(n){const{kernelName:t,backendName:e}=n,s=Yp(t,e);Qa.has(s)&&je(`The kernel '${t}' for backend '${e}' is already registered`),Qa.set(s,n)}function cw(n){const{kernelName:t}=n;ku.has(t)&&U().getBool("DEBUG")&&je(`Overriding the gradient for '${t}'`),ku.set(t,n)}function Yp(n,t){return`${t}_${n}`}function Zp(n){return n instanceof Float32Array||n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray}function uw(n){return n&&n.__esModule&&Object.prototype.hasOwnProperty.call(n,"default")?n.default:n}function hw(n){if(Object.prototype.hasOwnProperty.call(n,"__esModule"))return n;var t=n.default;if(typeof t=="function"){var e=function s(){var o=!1;try{o=this instanceof s}catch{}return o?Reflect.construct(t,arguments,this.constructor):t.apply(this,arguments)};e.prototype=t.prototype}else e={};return Object.defineProperty(e,"__esModule",{value:!0}),Object.keys(n).forEach(function(s){var o=Object.getOwnPropertyDescriptor(n,s);Object.defineProperty(e,s,o.get?o:{enumerable:!0,get:function(){return n[s]}})}),e}var vu,Jp;function dw(){if(Jp)return vu;Jp=1,vu=t;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function t(v,I,R){this.low=v|0,this.high=I|0,this.unsigned=!!R}t.prototype.__isLong__,Object.defineProperty(t.prototype,"__isLong__",{value:!0});function e(v){return(v&&v.__isLong__)===!0}t.isLong=e;var s={},o={};function r(v,I){var R,A,P;return I?(v>>>=0,(P=0<=v&&v<256)&&(A=o[v],A)?A:(R=a(v,(v|0)<0?-1:0,!0),P&&(o[v]=R),R)):(v|=0,(P=-128<=v&&v<128)&&(A=s[v],A)?A:(R=a(v,v<0?-1:0,!1),P&&(s[v]=R),R))}t.fromInt=r;function i(v,I){if(isNaN(v))return I?b:x;if(I){if(v<0)return b;if(v>=f)return N}else{if(v<=-m)return T;if(v+1>=m)return $}return v<0?i(-v,I).neg():a(v%p|0,v/p|0,I)}t.fromNumber=i;function a(v,I,R){return new t(v,I,R)}t.fromBits=a;var l=Math.pow;function c(v,I,R){if(v.length===0)throw Error("empty string");if(v==="NaN"||v==="Infinity"||v==="+Infinity"||v==="-Infinity")return x;if(typeof I=="number"?(R=I,I=!1):I=!!I,R=R||10,R<2||36<R)throw RangeError("radix");var A;if((A=v.indexOf("-"))>0)throw Error("interior hyphen");if(A===0)return c(v.substring(1),I,R).neg();for(var P=i(l(R,8)),_=x,B=0;B<v.length;B+=8){var z=Math.min(8,v.length-B),W=parseInt(v.substring(B,B+z),R);if(z<8){var H=i(l(R,z));_=_.mul(H).add(i(W))}else _=_.mul(P),_=_.add(i(W))}return _.unsigned=I,_}t.fromString=c;function u(v,I){return typeof v=="number"?i(v,I):typeof v=="string"?c(v,I):a(v.low,v.high,typeof I=="boolean"?I:v.unsigned)}t.fromValue=u;var h=65536,d=1<<24,p=h*h,f=p*p,m=f/2,g=r(d),x=r(0);t.ZERO=x;var b=r(0,!0);t.UZERO=b;var w=r(1);t.ONE=w;var y=r(1,!0);t.UONE=y;var C=r(-1);t.NEG_ONE=C;var $=a(-1,2147483647,!1);t.MAX_VALUE=$;var N=a(-1,-1,!0);t.MAX_UNSIGNED_VALUE=N;var T=a(0,-2147483648,!1);t.MIN_VALUE=T;var k=t.prototype;return k.toInt=function(){return this.unsigned?this.low>>>0:this.low},k.toNumber=function(){return this.unsigned?(this.high>>>0)*p+(this.low>>>0):this.high*p+(this.low>>>0)},k.toString=function(I){if(I=I||10,I<2||36<I)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(T)){var R=i(I),A=this.div(R),P=A.mul(R).sub(this);return A.toString(I)+P.toInt().toString(I)}else return"-"+this.neg().toString(I);for(var _=i(l(I,6),this.unsigned),B=this,z="";;){var W=B.div(_),H=B.sub(W.mul(_)).toInt()>>>0,K=H.toString(I);if(B=W,B.isZero())return K+z;for(;K.length<6;)K="0"+K;z=""+K+z}},k.getHighBits=function(){return this.high},k.getHighBitsUnsigned=function(){return this.high>>>0},k.getLowBits=function(){return this.low},k.getLowBitsUnsigned=function(){return this.low>>>0},k.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var I=this.high!=0?this.high:this.low,R=31;R>0&&(I&1<<R)==0;R--);return this.high!=0?R+33:R+1},k.isZero=function(){return this.high===0&&this.low===0},k.eqz=k.isZero,k.isNegative=function(){return!this.unsigned&&this.high<0},k.isPositive=function(){return this.unsigned||this.high>=0},k.isOdd=function(){return(this.low&1)===1},k.isEven=function(){return(this.low&1)===0},k.equals=function(I){return e(I)||(I=u(I)),this.unsigned!==I.unsigned&&this.high>>>31===1&&I.high>>>31===1?!1:this.high===I.high&&this.low===I.low},k.eq=k.equals,k.notEquals=function(I){return!this.eq(I)},k.neq=k.notEquals,k.ne=k.notEquals,k.lessThan=function(I){return this.comp(I)<0},k.lt=k.lessThan,k.lessThanOrEqual=function(I){return this.comp(I)<=0},k.lte=k.lessThanOrEqual,k.le=k.lessThanOrEqual,k.greaterThan=function(I){return this.comp(I)>0},k.gt=k.greaterThan,k.greaterThanOrEqual=function(I){return this.comp(I)>=0},k.gte=k.greaterThanOrEqual,k.ge=k.greaterThanOrEqual,k.compare=function(I){if(e(I)||(I=u(I)),this.eq(I))return 0;var R=this.isNegative(),A=I.isNegative();return R&&!A?-1:!R&&A?1:this.unsigned?I.high>>>0>this.high>>>0||I.high===this.high&&I.low>>>0>this.low>>>0?-1:1:this.sub(I).isNegative()?-1:1},k.comp=k.compare,k.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(w)},k.neg=k.negate,k.add=function(I){e(I)||(I=u(I));var R=this.high>>>16,A=this.high&65535,P=this.low>>>16,_=this.low&65535,B=I.high>>>16,z=I.high&65535,W=I.low>>>16,H=I.low&65535,K=0,j=0,X=0,Y=0;return Y+=_+H,X+=Y>>>16,Y&=65535,X+=P+W,j+=X>>>16,X&=65535,j+=A+z,K+=j>>>16,j&=65535,K+=R+B,K&=65535,a(X<<16|Y,K<<16|j,this.unsigned)},k.subtract=function(I){return e(I)||(I=u(I)),this.add(I.neg())},k.sub=k.subtract,k.multiply=function(I){if(this.isZero())return x;if(e(I)||(I=u(I)),n){var R=n.mul(this.low,this.high,I.low,I.high);return a(R,n.get_high(),this.unsigned)}if(I.isZero())return x;if(this.eq(T))return I.isOdd()?T:x;if(I.eq(T))return this.isOdd()?T:x;if(this.isNegative())return I.isNegative()?this.neg().mul(I.neg()):this.neg().mul(I).neg();if(I.isNegative())return this.mul(I.neg()).neg();if(this.lt(g)&&I.lt(g))return i(this.toNumber()*I.toNumber(),this.unsigned);var A=this.high>>>16,P=this.high&65535,_=this.low>>>16,B=this.low&65535,z=I.high>>>16,W=I.high&65535,H=I.low>>>16,K=I.low&65535,j=0,X=0,Y=0,tt=0;return tt+=B*K,Y+=tt>>>16,tt&=65535,Y+=_*K,X+=Y>>>16,Y&=65535,Y+=B*H,X+=Y>>>16,Y&=65535,X+=P*K,j+=X>>>16,X&=65535,X+=_*H,j+=X>>>16,X&=65535,X+=B*W,j+=X>>>16,X&=65535,j+=A*K+P*H+_*W+B*z,j&=65535,a(Y<<16|tt,j<<16|X,this.unsigned)},k.mul=k.multiply,k.divide=function(I){if(e(I)||(I=u(I)),I.isZero())throw Error("division by zero");if(n){if(!this.unsigned&&this.high===-2147483648&&I.low===-1&&I.high===-1)return this;var R=(this.unsigned?n.div_u:n.div_s)(this.low,this.high,I.low,I.high);return a(R,n.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:x;var A,P,_;if(this.unsigned){if(I.unsigned||(I=I.toUnsigned()),I.gt(this))return b;if(I.gt(this.shru(1)))return y;_=b}else{if(this.eq(T)){if(I.eq(w)||I.eq(C))return T;if(I.eq(T))return w;var B=this.shr(1);return A=B.div(I).shl(1),A.eq(x)?I.isNegative()?w:C:(P=this.sub(I.mul(A)),_=A.add(P.div(I)),_)}else if(I.eq(T))return this.unsigned?b:x;if(this.isNegative())return I.isNegative()?this.neg().div(I.neg()):this.neg().div(I).neg();if(I.isNegative())return this.div(I.neg()).neg();_=x}for(P=this;P.gte(I);){A=Math.max(1,Math.floor(P.toNumber()/I.toNumber()));for(var z=Math.ceil(Math.log(A)/Math.LN2),W=z<=48?1:l(2,z-48),H=i(A),K=H.mul(I);K.isNegative()||K.gt(P);)A-=W,H=i(A,this.unsigned),K=H.mul(I);H.isZero()&&(H=w),_=_.add(H),P=P.sub(K)}return _},k.div=k.divide,k.modulo=function(I){if(e(I)||(I=u(I)),n){var R=(this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,I.low,I.high);return a(R,n.get_high(),this.unsigned)}return this.sub(this.div(I).mul(I))},k.mod=k.modulo,k.rem=k.modulo,k.not=function(){return a(~this.low,~this.high,this.unsigned)},k.and=function(I){return e(I)||(I=u(I)),a(this.low&I.low,this.high&I.high,this.unsigned)},k.or=function(I){return e(I)||(I=u(I)),a(this.low|I.low,this.high|I.high,this.unsigned)},k.xor=function(I){return e(I)||(I=u(I)),a(this.low^I.low,this.high^I.high,this.unsigned)},k.shiftLeft=function(I){return e(I)&&(I=I.toInt()),(I&=63)===0?this:I<32?a(this.low<<I,this.high<<I|this.low>>>32-I,this.unsigned):a(0,this.low<<I-32,this.unsigned)},k.shl=k.shiftLeft,k.shiftRight=function(I){return e(I)&&(I=I.toInt()),(I&=63)===0?this:I<32?a(this.low>>>I|this.high<<32-I,this.high>>I,this.unsigned):a(this.high>>I-32,this.high>=0?0:-1,this.unsigned)},k.shr=k.shiftRight,k.shiftRightUnsigned=function(I){if(e(I)&&(I=I.toInt()),I&=63,I===0)return this;var R=this.high;if(I<32){var A=this.low;return a(A>>>I|R<<32-I,R>>>I,this.unsigned)}else return I===32?a(R,0,this.unsigned):a(R>>>I-32,0,this.unsigned)},k.shru=k.shiftRightUnsigned,k.shr_u=k.shiftRightUnsigned,k.toSigned=function(){return this.unsigned?a(this.low,this.high,!1):this},k.toUnsigned=function(){return this.unsigned?this:a(this.low,this.high,!0)},k.toBytes=function(I){return I?this.toBytesLE():this.toBytesBE()},k.toBytesLE=function(){var I=this.high,R=this.low;return[R&255,R>>>8&255,R>>>16&255,R>>>24,I&255,I>>>8&255,I>>>16&255,I>>>24]},k.toBytesBE=function(){var I=this.high,R=this.low;return[I>>>24,I>>>16&255,I>>>8&255,I&255,R>>>24,R>>>16&255,R>>>8&255,R&255]},t.fromBytes=function(I,R,A){return A?t.fromBytesLE(I,R):t.fromBytesBE(I,R)},t.fromBytesLE=function(I,R){return new t(I[0]|I[1]<<8|I[2]<<16|I[3]<<24,I[4]|I[5]<<8|I[6]<<16|I[7]<<24,R)},t.fromBytesBE=function(I,R){return new t(I[4]<<24|I[5]<<16|I[6]<<8|I[7],I[0]<<24|I[1]<<16|I[2]<<8|I[3],R)},vu}var Qp=dw(),tf=uw(Qp),pw=Wy({__proto__:null,default:tf},[Qp]);const Ds=tf||pw;function tl(n){return Ds.fromString(n,!0,16)}const ef=tl("c3a5c85c97cb3127"),Fs=tl("b492b66fbe98f273"),Ne=tl("9ae16a3b2f90404f");function Su(n){return n.xor(n.shru(47))}function nf(n,t,e){const s=n.slice(t,t+e);return Ds.fromBytes(Array.from(s),!0,!0)}function Mt(n,t){return nf(n,t,8)}function sf(n,t){return nf(n,t,4)}function he(n,t){return t===0?n:n.shru(t).or(n.shl(64-t))}function os(n,t,e=tl("9ddfea08eb382d69")){let s=n.xor(t).mul(e);s=s.xor(s.shru(47));let o=t.xor(s).mul(e);return o=o.xor(o.shru(47)),o=o.mul(e),o}function fw(n,t,e,s,o,r){o=o.add(n),r=he(r.add(o).add(s),21);const i=o;return o=o.add(t),o=o.add(e),r=r.add(he(o,44)),[o.add(s),r.add(i)]}function el(n,t,e,s){return fw(Mt(n,t),Mt(n,t+8),Mt(n,t+16),Mt(n,t+24),e,s)}function mw(n,t=n.length){if(t>=8){const e=Ne.add(t*2),s=Mt(n,0).add(Ne),o=Mt(n,t-8),r=he(o,37).mul(e).add(s),i=he(s,25).add(o).mul(e);return os(r,i,e)}if(t>=4){const e=Ne.add(t*2),s=sf(n,0);return os(s.shl(3).add(t),sf(n,t-4),e)}if(t>0){const e=n[0],s=n[t>>1],o=n[t-1],r=e+(s<<8),i=t+(o<<2);return Su(Ne.mul(r).xor(ef.mul(i))).mul(Ne)}return Ne}function gw(n,t=n.length){const e=Ne.add(t*2),s=Mt(n,0).mul(Fs),o=Mt(n,8),r=Mt(n,t-8).mul(e),i=Mt(n,t-16).mul(Ne);return os(he(s.add(o),43).add(he(r,30)).add(i),s.add(he(o.add(Ne),18)).add(r),e)}function xw(n,t=n.length){const e=Ne.add(t*2),s=Mt(n,0).mul(Ne),o=Mt(n,8),r=Mt(n,t-8).mul(e),i=Mt(n,t-16).mul(Ne),a=he(s.add(o),43).add(he(r,30)).add(i),l=os(a,s.add(he(o.add(Ne),18)).add(r),e),c=Mt(n,16).mul(e),u=Mt(n,24),h=a.add(Mt(n,t-32)).mul(e),d=l.add(Mt(n,t-24)).mul(e);return os(he(c.add(u),43).add(he(h,30)).add(d),c.add(he(u.add(s),18)).add(h),e)}function bw(n,t=n.length){const e=Ds.fromNumber(81,!0);if(t<=32)return t<=16?mw(n,t):gw(n,t);if(t<=64)return xw(n,t);let s=e,o=e.mul(Fs).add(113),r=Su(o.mul(Ne).add(113)).mul(Ne),i=[Ds.UZERO,Ds.UZERO],a=[Ds.UZERO,Ds.UZERO];s=s.mul(Ne).add(Mt(n,0));let l=0;const c=(t-1>>6)*64,u=c+(t-1&63)-63;do s=he(s.add(o).add(i[0]).add(Mt(n,l+8)),37).mul(Fs),o=he(o.add(i[1]).add(Mt(n,l+48)),42).mul(Fs),s=s.xor(a[1]),o=o.add(i[0]).add(Mt(n,l+40)),r=he(r.add(a[0]),33).mul(Fs),i=el(n,l,i[1].mul(Fs),s.add(a[0])),a=el(n,l+32,r.add(a[1]),o.add(Mt(n,l+16))),[r,s]=[s,r],l+=64;while(l!==c);const h=Fs.add(r.and(255).shl(1));return l=u,a[0]=a[0].add(t-1&63),i[0]=i[0].add(a[0]),a[0]=a[0].add(i[0]),s=he(s.add(o).add(i[0]).add(Mt(n,l+8)),37).mul(h),o=he(o.add(i[1]).add(Mt(n,l+48)),42).mul(h),s=s.xor(a[1].mul(9)),o=o.add(i[0].mul(9).add(Mt(n,l+40))),r=he(r.add(a[0]),33).mul(h),i=el(n,l,i[1].mul(h),s.add(a[0])),a=el(n,l+32,r.add(a[1]),o.add(Mt(n,l+16))),[r,s]=[s,r],os(os(i[0],a[0],h).add(Su(o).mul(ef)).add(r),os(i[1],a[1],h).add(s),h)}function rs(n,t){return t==="string"?is(n):Os([n],t)}function yw(n,t){return n instanceof Float32Array&&t==="float32"||n instanceof Int32Array&&t==="int32"||n instanceof Uint8Array&&t==="bool"}function Os(n,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(n)&&(n=_s(n)),U().getBool("DEBUG")&&Ky(n,t),yw(n,t))return n;if(t==null||t==="float32"||t==="complex64")return new Float32Array(n);if(t==="int32")return new Int32Array(n);if(t==="bool"){const e=new Uint8Array(n.length);for(let s=0;s<e.length;++s)Math.round(n[s])!==0&&(e[s]=1);return e}else throw new Error(`Unknown data type ${t}`)}function Oe(){return U().platform.now()}function is(n,t="utf-8"){return t=t||"utf-8",U().platform.encode(n,t)}function as(n,t="utf-8"){return t=t||"utf-8",U().platform.decode(n,t)}function en(n){return U().platform.isTypedArray!=null?U().platform.isTypedArray(n):Zp(n)}function _s(n,t=[],e=!1){if(t==null&&(t=[]),typeof n=="boolean"||typeof n=="number"||typeof n=="string"||Dc(n)||n==null||en(n)&&e)t.push(n);else if(Array.isArray(n)||en(n))for(let s=0;s<n.length;++s)_s(n[s],t,e);else{let s=-1;for(const o of Object.keys(n))/^([1-9]+[0-9]*|0)$/.test(o)&&(s=Math.max(s,Number(o)));for(let o=0;o<=s;o++)_s(n[o],t,e)}return t}class ww{constructor(t,e){this.backendTimer=t,this.logger=e,e==null&&(this.logger=new Iw)}profileKernel(t,e,s){let o;const r=()=>{o=s()};let i;const a=Oe();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(r);else{r();for(const c of o)c.dataSync();i=Promise.resolve({kernelMs:Oe()-a})}if(U().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let c=0;c<o.length;c++){const u=o[c];u.data().then(h=>{Cw(h,u.dtype,t)})}return{kernelName:t,outputs:o,inputs:e,timeMs:i.then(c=>c.kernelMs),extraInfo:i.then(c=>c.getExtraProfileInfo!=null?c.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:e,outputs:s,timeMs:o,inputs:r,extraInfo:i}=t;s.forEach(a=>{Promise.all([a.data(),o,i]).then(l=>{this.logger.logKernelProfile(e,a,l[0],l[1],r,l[2])})})}}function Cw(n,t,e){if(t!=="float32")return!1;for(let s=0;s<n.length;s++){const o=n[s];if(isNaN(o)||!isFinite(o))return console.warn(`Found ${o} in the result of '${e}'`),!0}return!1}class Iw{logKernelProfile(t,e,s,o,r,i){const a=typeof o=="number"?bo(`${o}ms`,9):o.error,l=bo(t,25),c=e.rank,u=e.size,h=bo(e.shape.toString(),14);let d="";for(const p in r){const f=r[p];if(f!=null){const m=f.shape||e.shape,g=m.length;d+=`${p}: ${g}D ${g>0?m:""} `}}console.log(`%c${l}	%c${a}	%c${c}D ${h}	%c${u}	%c${d}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}function $w(n,t,e){const s={},o={};for(let l=0;l<t.length;l++)s[t[l].id]=!0;for(let l=0;l<n.length;l++){const c=n[l],u=c.inputs;for(const h in u){const d=u[h];let p=!1;for(let f=0;f<t.length;f++)if(s[d.id]){c.outputs.forEach(m=>s[m.id]=!0),p=!0,o[c.id]=!0;break}if(p)break}}const r={};r[e.id]=!0;const i={};for(let l=n.length-1;l>=0;l--){const c=n[l],u=c.inputs;for(let h=0;h<c.outputs.length;h++)if(r[c.outputs[h].id]){for(const d in u)r[u[d].id]=!0,i[c.id]=!0;break}}const a=[];for(let l=0;l<n.length;l++){const c=n[l];if(o[c.id]&&i[c.id]){const u={};for(const d in c.inputs){const p=c.inputs[d];s[p.id]&&(u[d]=p)}const h=Object.assign({},c);h.inputs=u,h.outputs=c.outputs,a.push(h)}}return a}function kw(n,t,e,s){for(let o=t.length-1;o>=0;o--){const r=t[o],i=[];if(r.outputs.forEach(l=>{const c=n[l.id];c!=null?i.push(c):i.push(null)}),r.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${r.kernelName}.`);const a=r.gradient(i);for(const l in r.inputs){if(!(l in a))throw new Error(`Cannot backprop through input ${l}. Available gradients found: ${Object.keys(a)}.`);const c=e(()=>a[l]());if(c.dtype!=="float32")throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input ${l} must have 'float32' dtype, but has '${c.dtype}'`);const u=r.inputs[l];if(!Et(c.shape,u.shape))throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input '${l}' has shape '${c.shape}', which does not match the shape of the input '${u.shape}'`);if(n[u.id]==null)n[u.id]=c;else{const h=n[u.id];n[u.id]=s(h,c),h.dispose()}}}}const of=20,Zr=3,Nu=7;function vw(n,t,e,s){const o=lt(t),r=Sw(n,t,e,o),i=t.length,a=nl(n,t,e,o,r),l=["Tensor"];return s&&(l.push(`  dtype: ${e}`),l.push(`  rank: ${i}`),l.push(`  shape: [${t}]`),l.push("  values:")),l.push(a.map(c=>"    "+c).join(`
`)),l.join(`
`)}function Sw(n,t,e,s){const o=q(t),r=s[s.length-1],i=new Array(r).fill(0),a=t.length,l=e==="complex64"?Qr(n):n;if(a>1)for(let c=0;c<o/r;c++){const u=c*r;for(let h=0;h<r;h++)i[h]=Math.max(i[h],Jr(l[u+h],0,e).length)}return i}function Jr(n,t,e){let s;return Array.isArray(n)?s=`${parseFloat(n[0].toFixed(Nu))} + ${parseFloat(n[1].toFixed(Nu))}j`:er(n)?s=`'${n}'`:e==="bool"?s=rf(n):s=parseFloat(n.toFixed(Nu)).toString(),bo(s,t)}function rf(n){return n===0?"false":"true"}function nl(n,t,e,s,o,r=!0){const i=e==="complex64"?2:1,a=t[0],l=t.length;if(l===0){if(e==="complex64"){const m=Qr(n);return[Jr(m[0],0,e)]}return e==="bool"?[rf(n[0])]:[n[0].toString()]}if(l===1){if(a>of){const g=Zr*i;let x=Array.from(n.slice(0,g)),b=Array.from(n.slice((a-Zr)*i,a*i));return e==="complex64"&&(x=Qr(x),b=Qr(b)),["["+x.map((w,y)=>Jr(w,o[y],e)).join(", ")+", ..., "+b.map((w,y)=>Jr(w,o[a-Zr+y],e)).join(", ")+"]"]}return["["+(e==="complex64"?Qr(n):Array.from(n)).map((g,x)=>Jr(g,o[x],e)).join(", ")+"]"]}const c=t.slice(1),u=s.slice(1),h=s[0]*i,d=[];if(a>of){for(let m=0;m<Zr;m++){const g=m*h,x=g+h;d.push(...nl(n.slice(g,x),c,e,u,o,!1))}d.push("...");for(let m=a-Zr;m<a;m++){const g=m*h,x=g+h;d.push(...nl(n.slice(g,x),c,e,u,o,m===a-1))}}else for(let m=0;m<a;m++){const g=m*h,x=g+h;d.push(...nl(n.slice(g,x),c,e,u,o,m===a-1))}const p=l===2?",":"";d[0]="["+(a>0?d[0]+p:"");for(let m=1;m<d.length-1;m++)d[m]=" "+d[m]+p;let f=`,
`;for(let m=2;m<l;m++)f+=`
`;return d[d.length-1]=" "+d[d.length-1]+"]"+(r?"":f),d}function Qr(n){const t=[];for(let e=0;e<n.length;e+=2)t.push([n[e],n[e+1]]);return t}class me{constructor(t,e,s){if(this.dtype=e,this.shape=t.slice(),this.size=q(t),s!=null){const o=s.length;S(o===this.size,()=>`Length of values '${o}' does not match the size inferred by the shape '${this.size}'.`)}if(e==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||Yt(e,this.size),this.strides=lt(t)}set(t,...e){e.length===0&&(e=[0]),S(e.length===this.rank,()=>`The number of provided coordinates (${e.length}) must match the rank (${this.rank})`);const s=this.locToIndex(e);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let e=0;for(const o of t){if(o<0||o>=this.shape[e]){const r=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(r)}e++}let s=t[t.length-1];for(let o=0;o<t.length-1;++o)s+=this.strides[o]*t[o];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let e=t[t.length-1];for(let s=0;s<t.length-1;++s)e+=this.strides[s]*t[s];return e}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const e=new Array(this.shape.length);for(let s=0;s<e.length-1;++s)e[s]=Math.floor(t/this.strides[s]),t-=e[s]*this.strides[s];return e[e.length-1]=t,e}get rank(){return this.shape.length}toTensor(){return fn().makeTensor(this.values,this.shape,this.dtype)}}let fn=null,$o=null;function Nw(n){fn=n}function Tw(n){$o=n}class oe{constructor(t,e,s,o){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=e||"float32",this.size=q(t),this.strides=lt(t),this.dataId=s,this.id=o,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return $o.buffer(this.shape,this.dtype,t)}bufferSync(){return $o.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return pn(this.shape,t,this.dtype==="complex64")}arraySync(){return pn(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=fn().read(this.dataId);if(this.dtype==="string"){const e=await t;try{return e.map(s=>as(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),fn().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=fn().readSync(this.dataId);if(this.dtype==="string")try{return t.map(e=>as(e))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await fn().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),fn().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return $o.print(this,t)}clone(){return this.throwIfDisposed(),$o.clone(this)}toString(t=!1){const e=this.dataSync();return vw(e,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),$o.cast(this,t)}variable(t=!0,e,s){return this.throwIfDisposed(),fn().makeVariable(this,t,e,s)}}Object.defineProperty(oe,Symbol.hasInstance,{value:n=>!!n&&n.data!=null&&n.dataSync!=null&&n.throwIfDisposed!=null});function G(){return Oc("Tensor",()=>oe)}G();class sl extends oe{constructor(t,e,s,o){super(t.shape,t.dtype,t.dataId,o),this.trainable=e,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Et(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);fn().disposeTensor(this),this.dataId=t.dataId,fn().incRef(this,null)}dispose(){fn().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(sl,Symbol.hasInstance,{value:n=>n instanceof oe&&n.assign!=null&&n.assign instanceof Function});var af;(function(n){n.R0="R0",n.R1="R1",n.R2="R2",n.R3="R3",n.R4="R4",n.R5="R5",n.R6="R6"})(af||(af={}));var Tu;(function(n){n.float32="float32",n.int32="int32",n.bool="int32",n.complex64="complex64"})(Tu||(Tu={}));var Eu;(function(n){n.float32="float32",n.int32="int32",n.bool="bool",n.complex64="complex64"})(Eu||(Eu={}));var Ru;(function(n){n.float32="float32",n.int32="float32",n.bool="float32",n.complex64="complex64"})(Ru||(Ru={}));var Au;(function(n){n.float32="complex64",n.int32="complex64",n.bool="complex64",n.complex64="complex64"})(Au||(Au={}));const Ew={float32:Ru,int32:Tu,bool:Eu,complex64:Au};function Ue(n,t){if(n==="string"||t==="string"){if(n==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${n} with ${t}`)}return Ew[n][t]}function Du(n){return Ue(n,"int32")}function lf(n){return n!=null&&typeof n=="object"&&"texture"in n&&n.texture instanceof WebGLTexture}function cf(n){return typeof GPUBuffer<"u"&&n!=null&&typeof n=="object"&&"buffer"in n&&n.buffer instanceof GPUBuffer}function Zt(n,t){if(n.dtype===t.dtype)return[n,t];const e=Ue(n.dtype,t.dtype);return[n.cast(e),t.cast(e)]}function uf(n){const t=[];return hf(n,t,new Set),t}function hf(n,t,e){if(n==null)return;if(n instanceof oe){t.push(n);return}if(!Rw(n))return;const s=n;for(const o in s){const r=s[o];e.has(r)||(e.add(r),hf(r,t,e))}}function Rw(n){return Array.isArray(n)||typeof n=="object"}function Fu(n){return n.kernelName!=null}class df{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class ko{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new df}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const s=t[e];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:e}=this.initializeBackendsAndReturnBest();if(e)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:e}=this.initializeBackend(t);if(e)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,e,s=1){return t in this.registryFactory?(je(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:e,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:e,asyncInit:s}=this.initializeBackend(t);if(!(s?await e:e))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new ww(this.backendInstance),!0}setupRegisteredKernels(){jp(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){jp(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const e=this.registryFactory[t];if(e==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=e.factory();if(s&&!(s instanceof $c)&&typeof s.then=="function"){const o=++this.pendingBackendInitId,r=s.then(i=>o<this.pendingBackendInitId?!1:(this.registry[t]=i,this.pendingBackendInit=null,!0)).catch(i=>(o<this.pendingBackendInitId||(this.pendingBackendInit=null,je(`Initialization of backend ${t} failed`),je(i.stack||i.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return je(`Initialization of backend ${t} failed`),je(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,e)=>this.registryFactory[e].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const s=t[e],{success:o,asyncInit:r}=this.initializeBackend(s);if(r||o)return{name:s,asyncInit:r}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,e){const s=this.state.tensorInfo.get(e),o=s.backend,r=this.readSync(e),i=o.refCount(e);o.disposeData(e,!0),s.backend=t,t.move(e,r,s.shape,s.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,e){let s=null;if(e==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");e=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof e!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let o;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(o),()=>(o=e(),o instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),o))}scopedRun(t,e,s){t();try{const o=s();return e(),o}catch(o){throw e(),o}}nextTensorId(){return ko.nextTensorId++}nextVariableId(){return ko.nextVariableId++}clone(t){const e=O.runKernel(Ir,{x:t}),s={x:t},o=i=>({x:()=>{const a="float32",l={x:i},c={dtype:a};return O.runKernel(cr,l,c)}}),r=[];return this.addTapeNode(this.state.activeScope.name,s,[e],o,r,{}),e}runKernel(t,e,s){if(this.backendName==null&&this.backend,!(qp(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:e,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,e,s){const o=this.backend.numDataIds();let r=0;s.forEach(l=>{r+=l.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=o-e-r-i;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${t}'`)}runKernelFunc(t){let e,s=[];const o=this.isTapeOn(),r=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let l;const c=Fu(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Fu(t)){const{kernelName:f,inputs:m,attrs:g}=t;this.backendName==null&&this.backend;const x=qp(f,this.backendName);S(x!=null,()=>`Cannot find registered kernel '${f}' for backend '${this.backendName}'`),a=()=>{const b=this.backend.numDataIds();l=x.kernelFunc({inputs:m,attrs:g,backend:this.backend});const w=Array.isArray(l)?l:[l];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(f,b,w);const y=w.map(C=>C.rank!=null?C:this.makeTensorFromTensorInfo(C));if(o){const C=this.getTensorsForGradient(f,m,y);s=this.saveTensorsForBackwardMode(C)}return y}}else{const{forwardFunc:f}=t,m=g=>{o&&(s=g.map(x=>this.keep(this.clone(x))))};a=()=>{const g=this.backend.numDataIds();l=this.tidy(()=>f(this.backend,m));const x=Array.isArray(l)?l:[l];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,g,x),x}}const{inputs:u,attrs:h}=t,d=Fu(t)?null:t.backwardsFunc;let p;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?e=a():(p=this.profiler.profileKernel(c,u,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(p),e=p.outputs)}),o&&this.addTapeNode(c,u,e,d,s,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-r,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(u).map(f=>u[f]!=null?u[f].shape:null),outputShapes:e.map(f=>f.shape),kernelTimeMs:p.timeMs,extraInfo:p.extraInfo}),Array.isArray(l)?e:e[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,e,s){const o=Kp(t);if(o!=null){const r=o.inputsToSave||[],i=o.outputsToSave||[];let a;o.saveAllInputs?(S(Array.isArray(e),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(e).map(c=>e[c])):a=r.map(c=>e[c]);const l=s.filter((c,u)=>i[u]);return a.concat(l)}return[]}makeTensor(t,e,s,o){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",o=o||this.backend;let r=t;s==="string"&&er(t[0])&&(r=t.map(l=>is(l)));const i=o.write(r,e,s),a=new oe(e,s,i,this.nextTensorId());if(this.trackTensor(a,o),s==="string"){const l=this.state.tensorInfo.get(i),c=Xy(r);this.state.numBytes+=c-l.bytes,l.bytes=c}return a}makeTensorFromDataId(t,e,s,o){s=s||"float32";const r={dataId:t,shape:e,dtype:s};return this.makeTensorFromTensorInfo(r,o)}makeTensorFromTensorInfo(t,e){const{dataId:s,shape:o,dtype:r}=t,i=new oe(o,r,s,this.nextTensorId());return this.trackTensor(i,e),i}makeVariable(t,e=!0,s,o){s=s||this.nextVariableId().toString(),o!=null&&o!==t.dtype&&(t=t.cast(o));const r=new sl(t,e,s,this.nextTensorId());if(this.state.registeredVariables[r.name]!=null)throw new Error(`Variable with name ${r.name} was already registered`);return this.state.registeredVariables[r.name]=r,this.incRef(r,this.backend),r}trackTensor(t,e){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*Yi(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:e||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof sl||this.track(t)}incRef(t,e){this.trackTensor(t,e),this.backend.incRef(t.dataId)}removeDataId(t,e){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===e&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const e=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=e.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*Yi(t.dtype);this.state.numBytes-=s}e.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,e.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const e=this.state.registeredVariables[t];this.disposeVariable(e)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const e=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(o=>o.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-e,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const o of this.state.activeProfile.kernels)o.kernelTimeMs=await o.kernelTimeMs,o.extraInfo=await o.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,e,s,o,r,i){const a={id:this.state.nextTapeNodeId++,kernelName:t,inputs:e,outputs:s,saved:r},l=Kp(t);l!=null&&(o=l.gradFunc),o!=null&&(a.gradient=c=>(c=c.map((u,h)=>{if(u==null){const d=s[h],p=Ie(d.size,d.dtype);return this.makeTensor(p,d.shape,d.dtype)}return u}),o(c.length>1?c:c[0],r,i))),this.state.activeTape.push(a)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const e={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(e.name=t),this.state.scopeStack.push(e),this.state.activeScope=e}endScope(t){const e=uf(t),s=new Set(e.map(r=>r.id));for(let r=0;r<this.state.activeScope.track.length;r++){const i=this.state.activeScope.track[r];!i.kept&&!s.has(i.id)&&i.dispose()}const o=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],e.forEach(r=>{!r.kept&&r.scopeId===o.id&&this.track(r)})}gradients(t,e,s,o=!1){if(S(e.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const r=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));S(r instanceof oe,()=>"The result y returned by f() must be a tensor.");const i=$w(this.state.activeTape,e,r);if(!o&&i.length===0&&e.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[r.id]=s??Aw(r.shape),kw(a,i,c=>this.tidy(c),Dw);const l=e.map(c=>a[c.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(c=>{for(const u of c.saved)u.dispose()}),this.state.activeTape=null),{value:r,grads:l}})}customGrad(t){return S(Ec(t),()=>"The f passed in customGrad(f) must be a function."),(...e)=>{S(e.every(a=>a instanceof oe),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const o={};e.forEach((a,l)=>{o[l]=a});const r=(a,l)=>(s=t(...e,l),S(s.value instanceof oe,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),S(Ec(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),i=(a,l)=>{const c=s.gradFunc(a,l),u=Array.isArray(c)?c:[c];S(u.length===e.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),S(u.every(d=>d instanceof oe),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const h={};return u.forEach((d,p)=>{h[p]=()=>d}),h};return this.runKernelFunc({forwardFunc:r,backwardsFunc:i,inputs:o})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,e){return this.state.tensorInfo.get(t).backend.readToGPU(t,e)}async time(t){const e=Oe(),s=await this.backend.time(t);return s.wallMs=Oe()-e,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new df;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}ko.nextTensorId=0,ko.nextVariableId=0;function Aw(n){const t=Ac(q(n),"float32");return O.makeTensor(t,n,"float32")}function pf(){const n=kp();if(n._tfengine==null){const t=new Jy(n);n._tfengine=new ko(t)}return nw(n._tfengine.ENV),Nw(()=>n._tfengine),n._tfengine}const O=pf();function Dw(n,t){const e={a:n,b:t};return O.runKernel(Co,e)}function Fw(){return typeof navigator<"u"&&navigator!=null}function ff(n){if(n||Fw()){if(n||(n=navigator),n.product==="ReactNative")return!0;const t=n.userAgent||n.vendor||(typeof window<"u"?window.opera:"");if(!t){const e=n;return e.userAgentData&&e.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function mf(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const _e=U();_e.registerFlag("DEBUG",()=>!1,n=>{n&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")}),_e.registerFlag("IS_BROWSER",()=>mf()),_e.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u"),_e.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),_e.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),_e.registerFlag("PROD",()=>!1),_e.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>_e.getBool("DEBUG")),_e.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0),_e.registerFlag("IS_TEST",()=>!1),_e.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>_e.getBool("DEBUG")),_e.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1),_e.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1),_e.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);function ti(n,t){let e=n;if(en(n))return t==="string"?[]:[n.length];if(lf(n)){const o=n.channels||"RGBA";return[n.height,n.width*o.length]}else if(cf(n))return[n.buffer.size/(t==null?4:Yi(t))];if(!Array.isArray(n))return[];const s=[];for(;Array.isArray(e)||en(e)&&t!=="string";)s.push(e.length),e=e[0];return Array.isArray(n)&&U().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&gf(n,s,[]),s}function gf(n,t,e){if(e=e||[],!Array.isArray(n)&&!en(n)){S(t.length===0,()=>`Element arr[${e.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}S(t.length>0,()=>`Element arr[${e.join("][")}] should be a primitive, but is an array of ${n.length} elements`),S(n.length===t[0],()=>`Element arr[${e.join("][")}] should have ${t[0]} elements, but has ${n.length} elements`);const s=t.slice(1);for(let o=0;o<n.length;++o)gf(n[o],s,e.concat(o))}function xf(n,t,e,s){if(n!=="string_or_numeric"){if(n==null)throw new Error("Expected dtype cannot be null.");if(n!=="numeric"&&n!==t||n==="numeric"&&t==="string")throw new Error(`Argument '${e}' passed to '${s}' must be ${n} tensor, but got ${t} tensor`)}}function E(n,t,e,s="numeric"){if(n instanceof G())return xf(s,n.dtype,t,e),n;let o=yo(n);if(o!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(o=s),xf(s,o,t,e),n==null||!en(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string"){const l=n==null?"null":n.constructor.name;throw new Error(`Argument '${t}' passed to '${e}' must be a Tensor or TensorLike, but got '${l}'`)}const r=ti(n,o);!en(n)&&!Array.isArray(n)&&(n=[n]);const a=o!=="string"?Os(n,o):_s(n,[],!0);return O.makeTensor(a,r,o)}function bf(n,t,e,s="numeric"){if(!Array.isArray(n))throw new Error(`Argument ${t} passed to ${e} must be a \`Tensor[]\` or \`TensorLike[]\``);return n.map((r,i)=>E(r,`${t}[${i}]`,e,s))}const Ow="__op";function M(n){const t=Object.keys(n);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let e=t[0];const s=n[e];e.endsWith("_")&&(e=e.substring(0,e.length-1)),e=e+Ow;const o=(...r)=>{O.startScope(e);try{const i=s(...r);return Dc(i)&&console.error("Cannot return a Promise inside of tidy."),O.endScope(i),i}catch(i){throw O.endScope(null),i}};return Object.defineProperty(o,"name",{value:e,configurable:!0}),o}function _w(n,t){const e=E(n,"real","complex"),s=E(t,"imag","complex");vc(e.shape,s.shape,`real and imag shapes, ${e.shape} and ${s.shape}, must match in call to tf.complex().`);const o={real:e,imag:s};return O.runKernel(Wc,o)}const vo=M({complex_:_w});function ei(n,t,e,s){if(s==null)s=yo(n);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(cf(n)||lf(n)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return O.backend.createTensorFromGPUData(n,t||e,s)}if(!en(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){Un(t);const o=q(t),r=q(e);S(o===r,()=>`Based on the provided shape, [${t}], the tensor should have ${o} values but has ${r}`);for(let i=0;i<e.length;++i){const a=e[i],l=i===e.length-1?a!==q(t.slice(i)):!0;S(e[i]===t[i]||!l,()=>`Error creating a new Tensor. Inferred shape (${e}) does not match the provided shape (${t}). `)}}return!en(n)&&!Array.isArray(n)&&(n=[n]),t=t||e,n=s!=="string"?Os(n,s):_s(n,[],!0),O.makeTensor(n,t,s)}function yf(n,t,e){const s=ti(n,e);return ei(n,t,s,e)}class So{static join(t){return new So(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>en(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let e=0;for(let s=0;s<t.length;s++){const o=t[s];s!==t.length-1&&o.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const r=e+o.byteLength;this.shards.push({buffer:o,start:e,end:r}),e=r}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,e=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,e=isNaN(Number(e))?0:e,t=Math.max(0,t),e=Math.min(this.byteLength,e),e<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const o=e-t,r=new ArrayBuffer(o),i=new Uint8Array(r);let a=0;for(let l=s;l<this.shards.length;l++){const c=this.shards[l],h=t+a-c.start,d=a,f=Math.min(e,c.end)-c.start,m=new Uint8Array(c.buffer,h,f-h);if(i.set(m,d),a+=m.length,e<c.end)break}return r}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function e(o){return t<o.start?-1:t>=o.end?1:0}if(e(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=Lw(this.shards,e);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function Lw(n,t){let e=0,s=n.length;for(;e<=s;){const o=Math.floor((s-e)/2)+e,r=t(n[o]);if(r===0)return o;r<0?s=o:e=o+1}return-1}function Nn(){return O}function wf(){return O.memory()}function V(n,t){return O.tidy(n,t)}function It(n){uf(n).forEach(e=>e.dispose())}function Tn(n){return O.keep(n)}function Cf(n){return O.setBackend(n)}function Mw(){return O.ready()}function If(n,t,e=1){return O.registerBackend(n,t,e)}function Pw(){return O.backend}const $f=4;async function kf(n,t){const e=[],s=[],o=Array.isArray(n)?n.map(i=>i.name):Object.keys(n);for(let i=0;i<o.length;++i){const a=o[i],l=Array.isArray(n)?n[i].tensor:n[a];if(l.dtype!=="float32"&&l.dtype!=="int32"&&l.dtype!=="bool"&&l.dtype!=="string"&&l.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${l.dtype}`);const c={name:a,shape:l.shape,dtype:l.dtype};if(l.dtype==="string"){const u=new Promise(async h=>{const d=await l.bytes(),p=d.reduce((g,x)=>g+x.length,0)+$f*d.length,f=new Uint8Array(p);let m=0;for(let g=0;g<d.length;g++){const x=d[g],b=new Uint8Array(new Uint32Array([x.length]).buffer);f.set(b,m),m+=$f,f.set(x,m),m+=x.length}h(f)});s.push(u)}else s.push(l.data());t!=null&&(c.group=t),e.push(c)}const r=await Promise.all(s);return{data:Bw(r),specs:e}}function Bw(n){if(n===null)throw new Error(`Invalid input value: ${JSON.stringify(n)}`);let t=0;const e=[];n.forEach(r=>{if(t+=r.byteLength,e.push(r.byteLength===r.buffer.byteLength?r:new r.constructor(r)),!(r instanceof Float32Array||r instanceof Int32Array||r instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${r.constructor.name}`)});const s=new Uint8Array(t);let o=0;return e.forEach(r=>{s.set(new Uint8Array(r.buffer),o),o+=r.byteLength}),s.buffer}const Ou=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function vf(n){return Ou?Buffer.byteLength(n,"utf8"):new Blob([n]).size}function zw(n){if(Ou)return Buffer.from(n).toString("base64");const t=new Uint8Array(n);let e="";for(let s=0,o=t.length;s<o;s++)e+=String.fromCharCode(t[s]);return btoa(e)}function Vw(n){if(Ou){const s=Buffer.from(n,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(n),e=new Uint8Array(t.length);for(let s=0;s<t.length;++s)e.set([t.charCodeAt(s)],s);return e.buffer}function Ww(n){return So.join(n)}function Sf(n){if(n.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:n.modelTopology==null?0:vf(JSON.stringify(n.modelTopology)),weightSpecsBytes:n.weightSpecs==null?0:vf(JSON.stringify(n.weightSpecs)),weightDataBytes:n.weightData==null?0:new So(n.weightData).byteLength}}class Te{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Te.instance==null&&(Te.instance=new Te),Te.instance}static registerSaveRouter(t){Te.getInstance().saveRouters.push(t)}static registerLoadRouter(t){Te.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return Te.getHandlers(t,"save")}static getLoadHandlers(t,e){return Te.getHandlers(t,"load",e)}static getHandlers(t,e,s){const o=[];return(e==="load"?Te.getInstance().loadRouters:Te.getInstance().saveRouters).forEach(i=>{const a=i(t,s);a!==null&&o.push(a)}),o}}const Uw=n=>Te.getSaveHandlers(n);const _u="tensorflowjs",Lu=1,Ls="models_store",ls="model_info_store";function Nf(){if(!U().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const n=typeof window>"u"?self:window,t=n.indexedDB||n.mozIndexedDB||n.webkitIndexedDB||n.msIndexedDB||n.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Mu(n){const t=n.result;t.createObjectStore(Ls,{keyPath:"modelPath"}),t.createObjectStore(ls,{keyPath:"modelPath"})}class Ms{constructor(t){if(this.indexedDB=Nf(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,e){return new Promise((s,o)=>{const r=this.indexedDB.open(_u,Lu);r.onupgradeneeded=()=>Mu(r),r.onsuccess=()=>{const i=r.result;if(e==null){const a=i.transaction(Ls,"readonly"),c=a.objectStore(Ls).get(this.modelPath);c.onsuccess=()=>{if(c.result==null)return i.close(),o(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(c.result.modelArtifacts)},c.onerror=u=>(i.close(),o(c.error)),a.oncomplete=()=>i.close()}else{e.weightData=So.join(e.weightData);const a=Sf(e),l=i.transaction(ls,"readwrite");let c=l.objectStore(ls),u;try{u=c.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(d){return o(d)}let h;u.onsuccess=()=>{h=i.transaction(Ls,"readwrite");const d=h.objectStore(Ls);let p;try{p=d.put({modelPath:this.modelPath,modelArtifacts:e,modelArtifactsInfo:a})}catch(f){return o(f)}p.onsuccess=()=>s({modelArtifactsInfo:a}),p.onerror=f=>{c=l.objectStore(ls);const m=c.delete(this.modelPath);m.onsuccess=()=>(i.close(),o(p.error)),m.onerror=g=>(i.close(),o(p.error))}},u.onerror=d=>(i.close(),o(u.error)),l.oncomplete=()=>{h==null?i.close():h.oncomplete=()=>i.close()}}},r.onerror=i=>o(r.error)})}}Ms.URL_SCHEME="indexeddb://";const Tf=n=>U().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(Ms.URL_SCHEME)?Gw(n.slice(Ms.URL_SCHEME.length)):null;Te.registerSaveRouter(Tf),Te.registerLoadRouter(Tf);function Gw(n){return new Ms(n)}function Hw(n){return n.startsWith(Ms.URL_SCHEME)?n.slice(Ms.URL_SCHEME.length):n}class qw{constructor(){this.indexedDB=Nf()}async listModels(){return new Promise((t,e)=>{const s=this.indexedDB.open(_u,Lu);s.onupgradeneeded=()=>Mu(s),s.onsuccess=()=>{const o=s.result,r=o.transaction(ls,"readonly"),a=r.objectStore(ls).getAll();a.onsuccess=()=>{const l={};for(const c of a.result)l[c.modelPath]=c.modelArtifactsInfo;t(l)},a.onerror=l=>(o.close(),e(a.error)),r.oncomplete=()=>o.close()},s.onerror=o=>e(s.error)})}async removeModel(t){return t=Hw(t),new Promise((e,s)=>{const o=this.indexedDB.open(_u,Lu);o.onupgradeneeded=()=>Mu(o),o.onsuccess=()=>{const r=o.result,i=r.transaction(ls,"readwrite"),a=i.objectStore(ls),l=a.get(t);let c;l.onsuccess=()=>{if(l.result==null)return r.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const u=a.delete(t),h=()=>{c=r.transaction(Ls,"readwrite");const p=c.objectStore(Ls).delete(t);p.onsuccess=()=>e(l.result.modelArtifactsInfo),p.onerror=f=>s(l.error)};u.onsuccess=h,u.onerror=d=>(h(),r.close(),s(l.error))}},l.onerror=u=>(r.close(),s(l.error)),i.oncomplete=()=>{c==null?r.close():c.oncomplete=()=>r.close()}},o.onerror=r=>s(o.error)})}}const Gn="/",No="tensorflowjs_models",Ef="info",Kw="model_topology",jw="weight_specs",Xw="weight_data",Yw="model_metadata";function Rf(n){return{info:[No,n,Ef].join(Gn),topology:[No,n,Kw].join(Gn),weightSpecs:[No,n,jw].join(Gn),weightData:[No,n,Xw].join(Gn),modelMetadata:[No,n,Yw].join(Gn)}}function Af(n){for(const t of Object.values(n))window.localStorage.removeItem(t)}function Zw(n){const t=n.split(Gn);if(t.length<3)throw new Error(`Invalid key format: ${n}`);return t.slice(1,t.length-1).join(Gn)}function Jw(n){return n.startsWith(Ps.URL_SCHEME)?n.slice(Ps.URL_SCHEME.length):n}class Ps{constructor(t){if(!U().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=Rf(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const e=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),o=Sf(t),r=So.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(o)),this.LS.setItem(this.keys.topology,e),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,zw(r));const i={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(i)),{modelArtifactsInfo:o}}catch{throw Af(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${o.modelTopologyBytes}, weightSpecsBytes=${o.weightSpecsBytes}, weightDataBytes=${o.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const e={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);e.modelTopology=s;const o=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(o==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);e.weightSpecs=o;const r=this.LS.getItem(this.keys.modelMetadata);if(r!=null){const a=JSON.parse(r);e.format=a.format,e.generatedBy=a.generatedBy,e.convertedBy=a.convertedBy,a.signature!=null&&(e.signature=a.signature),a.userDefinedMetadata!=null&&(e.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(e.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(e.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(e.trainingConfig=a.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return e.weightData=Vw(i),e}}Ps.URL_SCHEME="localstorage://";const Df=n=>U().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(Ps.URL_SCHEME)?Qw(n.slice(Ps.URL_SCHEME.length)):null;Te.registerSaveRouter(Df),Te.registerLoadRouter(Df);function Qw(n){return new Ps(n)}class tC{constructor(){S(U().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),S(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},e=No+Gn,s=Gn+Ef;for(let o=0;o<this.LS.length;++o){const r=this.LS.key(o);if(r.startsWith(e)&&r.endsWith(s)){const i=Zw(r);t[i]=JSON.parse(this.LS.getItem(r))}}return t}async removeModel(t){t=Jw(t);const e=Rf(t);if(this.LS.getItem(e.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(e.info));return Af(e),s}}const Ff="://";class En{constructor(){this.managers={}}static getInstance(){return En.instance==null&&(En.instance=new En),En.instance}static registerManager(t,e){S(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(Ff)&&(t=t.slice(0,t.indexOf(Ff))),S(t.length>0,()=>"scheme must not be an empty string.");const s=En.getInstance();S(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=e}static getManager(t){const e=En.getInstance().managers[t];if(e==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return e}static getSchemes(){return Object.keys(En.getInstance().managers)}}class eC{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,e){return fetch(t,e)}now(){return performance.now()}encode(t,e){if(e!=="utf-8"&&e!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${e}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,e){return new TextDecoder(e).decode(t)}setTimeoutCustom(t,e){if(typeof window>"u"||!U().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,e);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},e),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const o=this.functionRefs[s.data.index];o(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return Zp(t)}}if(U().get("IS_BROWSER")){U().setPlatform("browser",new eC);try{En.registerManager(Ps.URL_SCHEME,new tC)}catch{}try{En.registerManager(Ms.URL_SCHEME,new qw)}catch{}}const nC={importFetch:()=>require("node-fetch")};let Pu;class sC{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,e){return U().global.fetch!=null?U().global.fetch(t,e):(Pu==null&&(Pu=nC.importFetch()),Pu(t,e))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,e){if(e!=="utf-8"&&e!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${e}`);return this.textEncoder.encode(t)}decode(t,e){return t.length===0?"":new this.util.TextDecoder(e).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}U().get("IS_NODE")&&!U().get("IS_BROWSER")&&U().setPlatform("node",new sC);function wt(n,t="float32",e){return t=t||"float32",Un(n),new me(n,t,e)}function oC(n,t){const e=E(n,"x","cast");if(!jy(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&e.dtype!=="string"||t!=="string"&&e.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:e},o={dtype:t};return O.runKernel(cr,s,o)}const nt=M({cast_:oC});function rC(n){const e={x:E(n,"x","clone","string_or_numeric")};return O.runKernel(Ir,e)}const Bs=M({clone_:rC});function iC(n,t=!1){console.log(n.toString(t))}pf(),Tw({buffer:wt,cast:nt,clone:Bs,print:iC});function aC(n,t){let e=E(n,"a","add"),s=E(t,"b","add");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(Co,o)}const J=M({add_:aC});function lC(n,t){let e=E(n,"a","floorDiv"),s=E(t,"b","floorDiv");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(wr,o)}const Of=M({floorDiv_:lC});function cC(n,t){let e=E(n,"a","div"),s=E(t,"b","div");if([e,s]=Zt(e,s),e.dtype==="int32"&&s.dtype==="int32")return Of(e,s);const o={a:e,b:s},r={};return O.runKernel(fr,o,r)}const dt=M({div_:cC});function uC(n,t){let e=E(n,"a","mul"),s=E(t,"b","mul");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(Ar,o)}const F=M({mul_:uC});function hC(n){const t=E(n,"x","abs");if(t.dtype==="complex64"){const e={x:t};return O.runKernel(oa,e)}else{const e={x:t};return O.runKernel(Zi,e)}}const Ee=M({abs_:hC});function dC(n){const e={x:E(n,"x","acos")};return O.runKernel(nr,e)}const pC=M({acos_:dC});function fC(n){const e={x:E(n,"x","acosh")};return O.runKernel(sr,e)}const mC=M({acosh_:fC});function gC(n,t=null,e=!1){const o={x:E(n,"x","all","bool")},r={axis:t,keepDims:e};return O.runKernel(Lc,o,r)}const _f=M({all_:gC});function xC(n,t=null,e=!1){const o={x:E(n,"x","any","bool")},r={axis:t,keepDims:e};return O.runKernel(Mc,o,r)}const Bu=M({any_:xC});function bC(n,t=0){const s={x:E(n,"x","argMax")},o={axis:t};return O.runKernel(Ji,s,o)}const ni=M({argMax_:bC});function yC(n,t=0){const s={x:E(n,"x","argMin")},o={axis:t};return O.runKernel(Qi,s,o)}const wC=M({argMin_:yC});function CC(n){const e={x:E(n,"x","asin")};return O.runKernel(or,e)}const IC=M({asin_:CC});function $C(n){const e={x:E(n,"x","asinh")};return O.runKernel(rr,e)}const kC=M({asinh_:$C});function vC(n){const e={x:E(n,"x","atan")};return O.runKernel(ir,e)}const SC=M({atan_:vC});function NC(n,t){let e=E(n,"a","atan2"),s=E(t,"b","atan2");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(lr,o)}const TC=M({atan2_:NC});function EC(n){const e={x:E(n,"x","atanh")};return O.runKernel(ar,e)}const RC=M({atanh_:EC});function si(n,t,e,s,o="NHWC",r){const i=n[3],a=[...t,i],l=qn(o);return ge(n,a,e,r,s,null,null,l)}function nn(n,t,e,s,o,r,i="channelsLast"){const[a,l]=oi(t);let c;if(i==="channelsLast")c=[a,l,n[3],n[3]];else if(i==="channelsFirst")c=[a,l,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return ge(n,c,e,s,o,r,!1,i)}function Hn(n,t,e,s,o,r,i="NDHWC"){const[a,l,c]=Vu(t);let u,h;if(i==="NDHWC")h="channelsLast",u=[a,l,c,n[4],n[4]];else if(i==="NCDHW")h="channelsFirst",u=[a,l,c,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return cs(n,u,e,s,o,!1,h,r)}function ge(n,t,e,s,o,r,i=!1,a="channelsLast"){let[l,c,u,h]=[-1,-1,-1,-1];if(a==="channelsLast")[l,c,u,h]=n;else if(a==="channelsFirst")[l,h,c,u]=n;else throw new Error(`Unknown dataFormat ${a}`);const[d,p,,f]=t,[m,g]=oi(e),[x,b]=oi(s),w=To(d,x),y=To(p,b),{padInfo:C,outHeight:$,outWidth:N}=FC(o,c,u,m,g,w,y,r,a),T=i?f*h:f;let k;return a==="channelsFirst"?k=[l,T,$,N]:a==="channelsLast"&&(k=[l,$,N,T]),{batchSize:l,dataFormat:a,inHeight:c,inWidth:u,inChannels:h,outHeight:$,outWidth:N,outChannels:T,padInfo:C,strideHeight:m,strideWidth:g,filterHeight:d,filterWidth:p,effectiveFilterHeight:w,effectiveFilterWidth:y,dilationHeight:x,dilationWidth:b,inShape:n,outShape:k,filterShape:t}}function cs(n,t,e,s,o,r=!1,i="channelsLast",a){let[l,c,u,h,d]=[-1,-1,-1,-1,-1];if(i==="channelsLast")[l,c,u,h,d]=n;else if(i==="channelsFirst")[l,d,c,u,h]=n;else throw new Error(`Unknown dataFormat ${i}`);const[p,f,m,,g]=t,[x,b,w]=Vu(e),[y,C,$]=Vu(s),N=To(p,y),T=To(f,C),k=To(m,$),{padInfo:v,outDepth:I,outHeight:R,outWidth:A}=OC(o,c,u,h,x,b,w,N,T,k,a),P=r?g*d:g;let _;return i==="channelsFirst"?_=[l,P,I,R,A]:i==="channelsLast"&&(_=[l,I,R,A,P]),{batchSize:l,dataFormat:i,inDepth:c,inHeight:u,inWidth:h,inChannels:d,outDepth:I,outHeight:R,outWidth:A,outChannels:P,padInfo:v,strideDepth:x,strideHeight:b,strideWidth:w,filterDepth:p,filterHeight:f,filterWidth:m,effectiveFilterDepth:N,effectiveFilterHeight:T,effectiveFilterWidth:k,dilationDepth:y,dilationHeight:C,dilationWidth:$,inShape:n,outShape:_,filterShape:t}}function AC(n,t,e,s,o){s==null&&(s=zu(n,t,e));const r=n[0],i=n[1],a=ri((r-t+2*s)/e+1,o),l=ri((i-t+2*s)/e+1,o);return[a,l]}function DC(n,t,e,s,o,r){o==null&&(o=zu(n,t[0],s[0]));const i=[0,0,0,e];for(let a=0;a<3;a++)n[a]+2*o>=t[a]&&(i[a]=ri((n[a]-t[a]+2*o)/s[a]+1,r));return i}function zu(n,t,e,s=1){const o=To(t,s);return Math.floor((n[0]*(e-1)-e+o)/2)}function oi(n){return typeof n=="number"?[n,n,n]:n.length===2?[n[0],n[1],1]:n}function Vu(n){return typeof n=="number"?[n,n,n]:n}function To(n,t){return t<=1?n:n+(n-1)*(t-1)}function FC(n,t,e,s,o,r,i,a,l){let c,u,h;if(typeof n=="number"){c={top:n,bottom:n,left:n,right:n,type:n===0?"VALID":"NUMBER"};const p=AC([t,e],r,s,n,a);u=p[0],h=p[1]}else if(n==="same"){u=Math.ceil(t/s),h=Math.ceil(e/o);const d=Math.max(0,(u-1)*s+r-t),p=Math.max(0,(h-1)*o+i-e),f=Math.floor(d/2),m=d-f,g=Math.floor(p/2),x=p-g;c={top:f,bottom:m,left:g,right:x,type:"SAME"}}else if(n==="valid")c={top:0,bottom:0,left:0,right:0,type:"VALID"},u=Math.ceil((t-r+1)/s),h=Math.ceil((e-i+1)/o);else if(typeof n=="object"){const d=l==="channelsLast"?n[1][0]:n[2][0],p=l==="channelsLast"?n[1][1]:n[2][1],f=l==="channelsLast"?n[2][0]:n[3][0],m=l==="channelsLast"?n[2][1]:n[3][1];c={top:d,bottom:p,left:f,right:m,type:d===0&&p===0&&f===0&&m===0?"VALID":"EXPLICIT"},u=ri((t-r+d+p)/s+1,a),h=ri((e-i+f+m)/o+1,a)}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:c,outHeight:u,outWidth:h}}function OC(n,t,e,s,o,r,i,a,l,c,u){let h,d,p,f;if(n==="valid"&&(n=0),typeof n=="number"){h={top:n,bottom:n,left:n,right:n,front:n,back:n,type:n===0?"VALID":"NUMBER"};const g=DC([t,e,s,1],[a,l,c],1,[o,r,i],n,u);d=g[0],p=g[1],f=g[2]}else if(n==="same"){d=Math.ceil(t/o),p=Math.ceil(e/r),f=Math.ceil(s/i);const m=(d-1)*o+a-t,g=(p-1)*r+l-e,x=(f-1)*i+c-s,b=Math.floor(m/2),w=m-b,y=Math.floor(g/2),C=g-y,$=Math.floor(x/2),N=x-$;h={top:y,bottom:C,left:$,right:N,front:b,back:w,type:"SAME"}}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:h,outDepth:d,outHeight:p,outWidth:f}}function ri(n,t){if(!t)return Math.trunc(n);switch(t){case"round":return Math.round(n);case"ceil":return Math.ceil(n);case"floor":return Math.floor(n);default:throw new Error(`Unknown roundingMode ${t}`)}}function zs(n){const[t,e,s]=oi(n);return t===1&&e===1&&s===1}function $e(n,t){return zs(n)||zs(t)}function Vs(n){return oi(n).every(t=>t>0)}function qn(n){if(n==="NHWC")return"channelsLast";if(n==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${n}`)}function Le(n,t,e){if(e!=null){if(typeof t=="string")throw Error(`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${t}.`);if(typeof t=="number")S(xo(t),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(o=>{S(xo(o),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${o}.`)})});else throw Error(`Error in ${n}: Unknown padding parameter: ${t}`)}}function _C(n,t){const s={x:E(n,"x","reshape","string_or_numeric")},o={shape:t};return O.runKernel(Pa,s,o)}const L=M({reshape_:_C});function LC(n,t,e,s,o){const r=E(n,"x","avgPool","float32"),i=1;S($e(e,i),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${e} and dilations '${i}'`);let a=r,l=!1;r.rank===3&&(l=!0,a=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),S(a.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),Le("avgPool",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o};let h=O.runKernel(ta,c,u);return h=nt(h,r.dtype),l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Wu=M({avgPool_:LC});function MC(n,t,e,s,o,r="NDHWC"){const i=E(n,"x","avgPool3d","float32");let a=i,l=!1;i.rank===4&&(l=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),S(a.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${a.rank}.`),S(r==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${r}`),S(typeof e=="number"&&e>0||Array.isArray(e)&&e[0]>0&&e[1]>0&&e[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${e}'`),Le("avgPool3d",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o,dataFormat:r};let h=O.runKernel(ea,c,u);return h=nt(h,a.dtype),l?L(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const PC=M({avgPool3d_:MC});function BC(n,t=0){S(n.length>=1,()=>"Pass at least one tensor to concat");const e=bf(n,"tensors","concat","string_or_numeric");if(e[0].dtype==="complex64"&&e.forEach(r=>{if(r.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${r.dtype}. `)}),e.length===1)return Bs(e[0]);const s=e,o={axis:t};return O.runKernel(ra,s,o)}const Me=M({concat_:BC});function zC(n,t,e=!1,s=!1){let o=E(n,"a","matMul"),r=E(t,"b","matMul");[o,r]=Zt(o,r);const i={a:o,b:r},a={transposeA:e,transposeB:s};return O.runKernel(na,i,a)}const Rt=M({matMul_:zC});function VC(n){const e={x:E(n,"x","sigmoid","float32")};return O.runKernel(Wr,e)}const Eo=M({sigmoid_:VC});function WC(n,t,e){const s=E(n,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const o={x:s},r={begin:t,size:e};return O.runKernel(Ua,o,r)}const Pt=M({slice_:WC});function UC(n){const e={x:E(n,"x","tanh","float32")};return O.runKernel(jr,e)}const ol=M({tanh_:UC});function GC(n,t,e){const s=E(n,"x","batchToSpaceND"),o=t.reduce((a,l)=>a*l);S(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),S(e.length===t.length,()=>`crops.length is ${e.length} but should be equal to blockShape.length  ${t.length}`),S(s.shape[0]%o===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${o}`);const r={x:s},i={blockShape:t,crops:e};return O.runKernel(sa,r,i)}const Uu=M({batchToSpaceND_:GC});function HC(n){let t;return n.rank===0||n.rank===1?t=L(n,[1,1,1,n.size]):n.rank===2?t=L(n,[1,1,n.shape[0],n.shape[1]]):n.rank===3?t=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]):t=n,t}function qC(n,t,e,s,o,r){r==null&&(r=.001);const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;s!=null&&(u=E(s,"offset","batchNorm")),S(a.rank===l.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),S(u==null||a.rank===u.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),S(c==null||a.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d={x:HC(i),scale:c,offset:u,mean:a,variance:l},p={varianceEpsilon:r},f=O.runKernel(fa,d,p);return L(f,i.shape)}const rl=M({batchNorm_:qC});function KC(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),S(i.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${i.rank}.`),S(a.rank===2||a.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${a.rank}.`),S(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${l.rank}.`),c!=null&&S(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${c.rank}.`),u!=null&&S(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),rl(i,a,l,u,c,r)}const jC=M({batchNorm2d_:KC});function XC(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),S(i.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${i.rank}.`),S(a.rank===3||a.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${a.rank}.`),S(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${l.rank}.`),c!=null&&S(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${c.rank}.`),u!=null&&S(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),rl(i,a,l,u,c,r)}const YC=M({batchNorm3d_:XC});function ZC(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),S(i.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${i.rank}.`),S(a.rank===4||a.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${a.rank}.`),S(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${l.rank}.`),c!=null&&S(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${c.rank}.`),u!=null&&S(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),rl(i,a,l,u,c,r)}const JC=M({batchNorm4d_:ZC});function QC(n,t,e){const s=E(n,"x","bincount"),o=E(t,"weights","bincount");S(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),S(e>=0,()=>`size must be non-negative, but got ${e}.`),S(o.size===s.size||o.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${o.shape}.`);const r={x:s,weights:o},i={size:e};return O.runKernel(zc,r,i)}const tI=M({bincount_:QC});function eI(n,t){let e=E(n,"broadcastTo","x");const s=e.shape;if(Un(t),t.length<e.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${e.rank}.`);if(t.length>e.rank){const c=e.shape.slice();for(;c.length<t.length;)c.unshift(1);e=L(e,c)}const o=e.shape,r=Array.from(t);for(let c=t.length-1;c>=0;c--)if(o[c]===t[c])r[c]=1;else if(e.shape[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(r.map((c,u)=>c>1?u:-1).filter(c=>c>=0).length===0)return Bs(e);const a={x:e},l={reps:r};return O.runKernel(Xr,a,l)}const ii=M({broadcastTo_:eI});function nI(n){const e={x:E(n,"x","ceil","float32")};return O.runKernel(ur,e)}const sI=M({ceil_:nI});function il(n,t,e){Un(n),e=e||yo(t);const s={shape:n,value:t,dtype:e};return O.runKernel(su,{},s)}function oI(n,t,e){const s=E(n,"x","clipByValue");if(S(t<=e,()=>`Error in clip: min (${t}) must be less than or equal to max (${e}).`),t===e)return il(s.shape,t,s.dtype);const o={x:s},r={clipValueMin:t,clipValueMax:e};return O.runKernel(hr,o,r)}const Xe=M({clipByValue_:oI});function rI(n){return Me(n,0)}const iI=M({concat1d_:rI});function aI(n,t){return Me(n,t)}const lI=M({concat2d_:aI});function cI(n,t){return Me(n,t)}const uI=M({concat3d_:cI});function hI(n,t){return Me(n,t)}const dI=M({concat4d_:hI});function pI(n,t,e,s,o="NHWC",r=[1,1],i){const a=E(n,"x","conv2d","float32"),l=E(t,"filter","conv2d","float32");let c=a,u=!1;a.rank===3&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),S(c.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),S(l.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${l.rank}.`),Le("conv2d",s,i);const h=o==="NHWC"?c.shape[3]:c.shape[1];S(h===l.shape[2],()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${l.shape[2]}.`),S($e(e,r),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`),S(Vs(r),()=>"Error in conv2D: Dilated rates should be larger than 0."),S(Vs(e),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:c,filter:l},p={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i},f=O.runKernel(ia,d,p);return u?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const Ws=M({conv2d_:pI});function fI(n,t,e,s,o="NWC",r=1,i){const a=E(n,"x","conv1d"),l=E(t,"filter","conv1d");let c=a,u=!1;a.rank===2&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1]])),S(c.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),S(l.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),Le("conv1d",s,i),S(c.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${l.shape[1]}.`),S($e(e,r),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${e} and dilation '${r}'`),S(Vs(r),()=>"Error in conv1D: Dilated rates should be larger than 0."),S(Vs(e),()=>"Error in conv1D: Stride should be larger than 0."),S(o==="NWC",()=>`Error in conv1d: got dataFormat of ${o} but only NWC is currently supported.`);const h=L(l,[1,l.shape[0],l.shape[1],l.shape[2]]),d=L(c,[c.shape[0],1,c.shape[1],c.shape[2]]),g=Ws(d,h,[1,e],s,"NHWC",[1,r],i);return u?L(g,[g.shape[2],g.shape[3]]):L(g,[g.shape[0],g.shape[2],g.shape[3]])}const Lf=M({conv1d_:fI});function mI(n,t,e,s,o,r="NHWC",i){S(n.length===t.rank,()=>`Length of inShape (${n.length}) and rank of dy (${t.rank}) must match`);let a=n,l=t,c=!1;t.rank===3&&(c=!0,l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]),a=[1,n[0],n[1],n[2]]),S(a.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${a.length}.`),S(l.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${l.rank}`),S(e.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${e.rank}`);const u=r==="NHWC"?a[3]:a[1],h=r==="NHWC"?l.shape[3]:l.shape[1];S(u===e.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${e.shape[2]}.`),S(h===e.shape[3],()=>`Error in conv2dDerInput: depth of output (${h}) must match output depth for filter ${e.shape[3]}.`),Le("conv2dDerInput",o,i);const d={dy:l,filter:e},p={strides:s,pad:o,dataFormat:r,dimRoundingMode:i,inputShape:a},f=O.runKernel(aa,d,p);return c?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const Gu=M({conv2DBackpropInput_:mI});function gI(n,t,e,s,o,r){const i=E(n,"x","conv2dTranspose"),a=E(t,"filter","conv2dTranspose");return Gu(e,i,a,s,o,"NHWC",r)}const Mf=M({conv2dTranspose_:gI});function xI(n,t,e,s,o="NDHWC",r=[1,1,1]){const i=E(n,"x","conv3d"),a=E(t,"filter","conv3d");let l=i,c=!1;i.rank===4&&(c=!0,l=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),S(l.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${l.rank}.`),S(a.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${a.rank}.`),S(l.shape[4]===a.shape[3],()=>`Error in conv3d: depth of input (${l.shape[4]}) must match input depth for filter ${a.shape[3]}.`),S($e(e,r),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`),S(o==="NDHWC",()=>`Error in conv3d: got dataFormat of ${o} but only NDHWC is currently supported.`),S(Vs(r),()=>"Error in conv3D: Dilated rates should be larger than 0."),S(Vs(e),()=>"Error in conv3D: Strides should be larger than 0.");const u={x:l,filter:a},h={strides:e,pad:s,dataFormat:o,dilations:r},d=O.runKernel(la,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const bI=M({conv3d_:xI});function yI(n,t,e,s,o){S(n.length===t.rank,()=>`Length of inShape (${n.length}) and rank of dy (${t.rank}) must match`);let r=n,i=t,a=!1;t.rank===4&&(a=!0,i=L(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),r=[1,n[0],n[1],n[2],n[3]]);const l=r[4],c=i.shape[4];S(r.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${r.length}.`),S(i.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${i.rank}`),S(e.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${e.rank}`),S(l===e.shape[3],()=>`Error in conv3dDerInput: depth of input (${l}) must match input depth for filter ${e.shape[3]}.`),S(c===e.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${e.shape[4]}.`);const u={dy:i,filter:e},h={pad:o,strides:s,inputShape:r},d=O.runKernel(Hc,u,h);return a?L(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Pf=M({conv3DBackpropInput_:yI});function wI(n,t,e,s,o){const r=E(n,"x","conv3dTranspose"),i=E(t,"filter","conv3dTranspose");return Pf(e,r,i,s,o)}const CI=M({conv3dTranspose_:wI});function II(n){const e={x:E(n,"x","cos","float32")};return O.runKernel(dr,e)}const Hu=M({cos_:II});function $I(n){const e={x:E(n,"x","cosh","float32")};return O.runKernel(pr,e)}const Bf=M({cosh_:$I});function kI(n,t=0,e=!1,s=!1){const r={x:E(n,"x","cumprod")},i={axis:t,exclusive:e,reverse:s};return O.runKernel(qc,r,i)}const qu=M({cumprod_:kI});function vI(n,t=0,e=!1,s=!1){const r={x:E(n,"x","cumsum")},i={axis:t,exclusive:e,reverse:s};return O.runKernel(ca,r,i)}const zf=M({cumsum_:vI});function SI(n,t,e,s=!1){const o=E(n,"x","denseBincount"),r=E(t,"weights","denseBincount");S(o.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${o.dtype}`),S(o.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${o.rank}.`),S(e>=0,()=>`size must be non-negative, but got ${e}.`),S(r.size===o.size||r.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${o.shape}, weights shape: ${r.shape}.`);const i={x:o,weights:r},a={size:e,binaryOutput:s};return O.runKernel(jc,i,a)}const Vf=M({denseBincount_:SI});function NI(n,t,e="NHWC"){const s=E(n,"x","depthToSpace","float32"),o=e==="NHWC"?s.shape[1]:s.shape[2],r=e==="NHWC"?s.shape[2]:s.shape[3],i=e==="NHWC"?s.shape[3]:s.shape[1];S(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),S(o*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${o} and ${t}  for depthToSpace with input shape
    ${s.shape}`),S(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t} for depthToSpace with input shape
        ${s.shape}`),S(i%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${i} for depthToSpace with input shape ${s.shape}`);const a={x:s},l={blockSize:t,dataFormat:e};return O.runKernel(Xc,a,l)}const TI=M({depthToSpace_:NI});function EI(n,t,e,s,o="NHWC",r=[1,1],i){const a=E(n,"x","depthwiseConv2d","float32"),l=E(t,"filter","depthwiseConv2d","float32");let c=a,u=!1;a.rank===3&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),S(c.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),S(l.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${l.rank}.`);const h=o==="NHWC"?c.shape[3]:c.shape[1];S(h===l.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${h}) must match the inChannels dimension in filter ${l.shape[2]}.`),Le("depthwiseConv2d",s,i);const d={x:c,filter:l},p={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i},f=O.runKernel(ua,d,p);return u?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const Ku=M({depthwiseConv2d_:EI});function RI(n,t,e,s,o=[1,1],r="NHWC"){const i=E(n,"x","dilation2d"),a=E(t,"filter","dilation2d");S(i.rank===3||i.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${i.rank}.`),S(a.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${a.rank}.`),S(r==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${r}`);let l=i,c=!1;i.rank===3&&(l=L(i,[1,i.shape[0],i.shape[1],i.shape[2]]),c=!0),S(l.shape[3]===a.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${l.shape[3]} vs ${a.shape[2]}`);const u={x:l,filter:a},h={strides:e,pad:s,dilations:o},d=O.runKernel(ha,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const AI=M({dilation2d_:RI});function Ro(n,t){const e=n.length,s=[];for(let o=0;o<e;o++){const r=e-1-o,i=n[r]||1;(t[t.length-1-o]||1)>1&&i===1&&s.unshift(r)}return s}function re(n,t){const e=[];for(let s=0;s<t.length;s++){const o=n[n.length-s-1],r=t.length-s-1,i=t[r];(o==null||o===1&&i>1)&&e.unshift(r)}return e}function mt(n,t){const e=Math.max(n.length,t.length),s=new Array(e);for(let o=0;o<e;o++){let r=n[n.length-o-1];r==null&&(r=1);let i=t[t.length-o-1];if(i==null&&(i=1),r===1)s[e-o-1]=i;else if(i===1)s[e-o-1]=r;else if(r!==i){const a=`Operands could not be broadcast together with shapes ${n} and ${t}.`;throw Error(a)}else s[e-o-1]=r}return s}function DI(n,t){let e=E(n,"a","equal","string_or_numeric"),s=E(t,"b","equal","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(da,o)}const Rn=M({equal_:DI});function FI(n,t,e){const s=E(t,"a","where"),o=E(e,"b","where"),r=E(n,"condition","where","bool"),i=mt(mt(r.shape,s.shape),o.shape),a=ii(r,i),l=ii(s,i),c=ii(o,i),u={condition:a,t:l,e:c};return O.runKernel(Wa,u)}const Re=M({where_:FI});function OI(n){const e={x:E(n,"x","zerosLike")};return O.runKernel(Ya,e)}const kt=M({zerosLike_:OI});function _I(n,t){let e=E(n,"a","div"),s=E(t,"b","div");[e,s]=Zt(e,s);const o=dt(e,s),r=kt(o),i=Rn(s,r);return Re(i,r,o)}const LI=M({divNoNan_:_I});function MI(n,t){const e=E(n,"t1","dot"),s=E(t,"t2","dot");S((e.rank===1||e.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${e.rank} and ${s.rank}.`);const o=e.rank===1?e.size:e.shape[1],r=s.rank===1?s.size:s.shape[0];if(S(o===r,()=>`Error in dot: inner dimensions of inputs must match, but got ${o} and ${r}.`),e.rank===1&&s.rank===1){const i=L(e,[1,-1]),a=L(s,[-1,1]),l=Rt(i,a);return L(l,[])}else if(e.rank===1&&s.rank===2){const i=L(e,[1,-1]),a=L(s,[s.shape[0],s.shape[1]]),l=Rt(i,a);return L(l,[l.size])}else if(e.rank===2&&s.rank===1){const i=L(s,[-1,1]),a=Rt(e,i);return L(a,[a.size])}else{const i=L(s,[s.shape[0],s.shape[1]]);return Rt(e,i)}}const PI=M({dot_:MI});function BI(n,...t){const e=t.map((o,r)=>E(o,`tensors${r}`,"einsum")),s={equation:n};return O.runKernel(tu,e,s)}const ai=M({einsum_:BI});function zI(n){const e={x:E(n,"x","elu","float32")};return O.runKernel(mr,e)}const al=M({elu_:zI});function VI(n){let t=E(n,"x","erf");S(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=nt(t,"float32"));const e={x:t};return O.runKernel(gr,e)}const Wf=M({erf_:VI});function ju(n,t){for(let e=0;e<n.length;++e)if(n[n.length-e-1]!==t-1-e)return!1;return!0}function Uf(n,t,e){const s=n.length+t.length,o=[];let r=0,i=0;for(let a=0;a<s;a++)e.indexOf(a)===-1?o.push(n[r++]):o.push(t[i++]);return o}function de(n,t){const e=[],s=n.length;for(let r=0;r<s;r++)t.indexOf(r)===-1&&e.push(n[r]);const o=t.map(r=>n[r]);return[e,o]}function ne(n,t){const e=t.map(s=>1);return Uf(n,e,t)}function xe(n,t,e){S(ju(t,e),()=>`${n} supports only inner-most axes for now. Got axes ${t} and rank-${e} input.`)}function qt(n,t){if(ju(n,t))return null;const e=[];for(let s=0;s<t;++s)n.indexOf(s)===-1&&e.push(s);return n.forEach(s=>e.push(s)),e}function us(n){return n.map((t,e)=>[e,t]).sort((t,e)=>t[1]-e[1]).map(t=>t[0])}function Jt(n,t){const e=[];for(let s=t-n;s<t;++s)e.push(s);return e}function WI(n,t=null,e=!1){const o={x:E(n,"x","max")},r={reductionIndices:t,keepDims:e};return O.runKernel(ka,o,r)}const mn=M({max_:WI});function UI(n,t=null,e=!1){const o={x:E(n,"x","min")},r={axis:t,keepDims:e};return O.runKernel(Ta,o,r)}const ll=M({min_:UI});function GI(n,t){let e=E(n,"base","pow"),s=E(t,"exp","pow");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(Dr,o)}const Us=M({pow_:GI});function At(n,t){if((en(n)&&t!=="string"||Array.isArray(n))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&en(n)&&!(n instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return ei(n,[],[],t)}function HI(n){const e={x:E(n,"x","sqrt","float32")};return O.runKernel(Gr,e)}const ke=M({sqrt_:HI});function qI(n){const t=E(n,"x","square"),e={};return O.runKernel("Square",{x:t},e)}const Vt=M({square_:qI});function KI(n,t=null,e=!1){let s=E(n,"x","sum");s.dtype==="bool"&&(s=nt(s,"int32"));const o={x:s},r={axis:t,keepDims:e};return O.runKernel(Ga,o,r)}const ut=M({sum_:KI});function jI(n,t="euclidean",e=null,s=!1){n=E(n,"x","norm");const o=Gf(n,t,e);let r=o.shape;if(s){const i=yt(e,n.shape);r=ne(o.shape,i)}return L(o,r)}function Gf(n,t,e=null){if(n.rank===0)return Ee(n);if(n.rank!==1&&e===null)return Gf(L(n,[-1]),t,e);if(n.rank===1||typeof e=="number"||Array.isArray(e)&&e.length===1){if(t===1)return ut(Ee(n),e);if(t===1/0)return mn(Ee(n),e);if(t===-1/0)return ll(Ee(n),e);if(t==="euclidean"||t===2)return ke(ut(Us(Ee(n),At(2,"int32")),e));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(e)&&e.length===2){if(t===1)return mn(ut(Ee(n),e[0]),e[1]-1);if(t===1/0)return mn(ut(Ee(n),e[1]),e[0]);if(t===-1/0)return ll(ut(Ee(n),e[1]),e[0]);if(t==="fro"||t==="euclidean")return ke(ut(Vt(n),e));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${e}`)}const cl=M({norm_:jI});function XI(n,t=null,e=!1){return cl(n,"euclidean",t,e)}const YI=M({euclideanNorm_:XI});function ZI(n){const e={x:E(n,"x","exp")};return O.runKernel(xr,e)}const An=M({exp_:ZI});function JI(n,t=0){const e=E(n,"x","expandDims","string_or_numeric");S(t<=e.rank,()=>"Axis must be <= rank of the tensor");const s={input:e},o={dim:t};return O.runKernel(pa,s,o)}const Pe=M({expandDims_:JI});function QI(n){const e={x:E(n,"x","expm1")};return O.runKernel(br,e)}const t$=M({expm1_:QI});function e$(n,t){const e=E(n,"x","tile","string_or_numeric");S(e.rank===t.length,()=>`Error in transpose: rank of input ${e.rank} must match length of reps ${t}.`);const s={x:e},o={reps:t};return O.runKernel(Xr,s,o)}const gn=M({tile_:e$});function n$(n,t,e,s="float32"){t==null&&(t=n);const o=wt([n,t],s),r=n<=t?n:t;for(let a=0;a<r;++a)o.set(1,a,a);const i=L(o.toTensor(),[n,t]);if(e==null)return i;if(e.length===1)return gn(Pe(i,0),[e[0],1,1]);if(e.length===2)return gn(Pe(Pe(i,0),0),[e[0],e[1],1,1]);if(e.length===3)return gn(Pe(Pe(Pe(i,0),0),0),[e[0],e[1],e[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${e.length}D.`)}const Hf=M({eye_:n$});function s$(n){const e={x:E(n,"x","floor","float32")};return O.runKernel(yr,e)}const ul=M({floor_:s$});function o$(n,t,e=0,s=0){const o=E(n,"x","gather"),r=E(t,"indices","gather","int32"),i={x:o,indices:r},a={axis:e,batchDims:s};return O.runKernel(ma,i,a)}const Xu=M({gather_:o$});function r$(n,t){let e=E(n,"a","greater","string_or_numeric"),s=E(t,"b","greater","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(ga,o)}const Ye=M({greater_:r$});function i$(n,t){let e=E(n,"a","greaterEqual","string_or_numeric"),s=E(t,"b","greaterEqual","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(Cr,o)}const Gs=M({greaterEqual_:i$});function a$(n){const e={input:E(n,"input","imag")};return O.runKernel(iu,e)}const Yu=M({imag_:a$});function l$(n){const e={x:E(n,"x","isFinite")};return O.runKernel($r,e)}const c$=M({isFinite_:l$});function u$(n){const e={x:E(n,"x","isInf")};return O.runKernel(kr,e)}const h$=M({isInf_:u$});function d$(n){const e={x:E(n,"x","isNaN")};return O.runKernel(vr,e)}const p$=M({isNaN_:d$});function f$(n,t=.2){const s={x:E(n,"x","leakyRelu")},o={alpha:t};return O.runKernel(xa,s,o)}const Zu=M({leakyRelu_:f$});function m$(n,t){let e=E(n,"a","less","string_or_numeric"),s=E(t,"b","less","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(ba,o)}const hl=M({less_:m$});function g$(n,t){let e=E(n,"a","lessEqual","string_or_numeric"),s=E(t,"b","lessEqual","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(ya,o)}const Ao=M({lessEqual_:g$});function x$(n,t=5,e=1,s=1,o=.5){const r=E(n,"x","localResponseNormalization");S(r.rank===4||r.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${r.rank}.`),S(xo(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let i=r,a=!1;r.rank===3&&(a=!0,i=L(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const l={x:i},c={depthRadius:t,bias:e,alpha:s,beta:o},u=O.runKernel($a,l,c);return a?L(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const b$=M({localResponseNormalization_:x$});function y$(n){const e={x:E(n,"x","log","float32")};return O.runKernel(Sr,e)}const Dn=M({log_:y$});function w$(n){const e={x:E(n,"x","log1p")};return O.runKernel(Nr,e)}const qf=M({log1p_:w$});function C$(n,t){S(Ec(n),()=>"The f passed in variableGrads(f) must be a function"),S(t==null||Array.isArray(t)&&t.every(c=>c instanceof sl),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const e=t!=null;if(!e){t=[];for(const c in O.registeredVariables)t.push(O.registeredVariables[c])}const s=e?t.filter(c=>!c.trainable):null,o=t.length;t=t.filter(c=>c.trainable),S(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${o} variables is trainable.`);const r=!0,{value:i,grads:a}=O.gradients(n,t,null,r);S(a.some(c=>c!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),S(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const l={};return t.forEach((c,u)=>{a[u]!=null&&(l[c.name]=a[u])}),s?.forEach(c=>l[c.name]=null),{value:i,grads:l}}function Do(n){return O.customGrad(n)}function I$(n){const e={x:E(n,"x","neg")};return O.runKernel(Ra,e)}const Qt=M({neg_:I$});function $$(n){const e={x:E(n,"x","softplus")};return O.runKernel(Ur,e)}const li=M({softplus_:$$});function k$(n){const t=E(n,"x","logSigmoid");return Do(s=>({value:Qt(li(Qt(s))),gradFunc:i=>F(i,Eo(Qt(s)))}))(t)}const v$=M({logSigmoid_:k$});function S$(n,t){let e=E(n,"a","sub"),s=E(t,"b","sub");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(qr,o)}const pt=M({sub_:S$});function N$(n,t=-1){const e=E(n,"logits","logSoftmax");if(t===-1&&(t=e.rank-1),t!==e.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${e.rank} and axis was ${t}`);return Do((o,r)=>{const a=mn(o,t,!0),l=pt(o,a),c=pt(nt(l,"float32"),Dn(ut(An(l),t,!0)));return r([c]),{value:c,gradFunc:(h,d)=>{const[p]=d,f=!0,m=An(p);return pt(h,F(ut(h,t,f),m))}}})(e)}const Kf=M({logSoftmax_:N$});function T$(n,t=null,e=!1){const s=E(n,"x","logSumExp"),o=yt(t,s.shape),r=mn(s,o,!0),i=pt(s,r),a=An(i),l=ut(a,o),c=Dn(l),u=J(L(r,c.shape),c);if(e){const h=ne(u.shape,o);return L(u,h)}return u}const jf=M({logSumExp_:T$});function E$(n,t){const e=E(n,"a","logicalAnd","bool"),s=E(t,"b","logicalAnd","bool");mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(wa,o)}const Kn=M({logicalAnd_:E$});function R$(n){const e={x:E(n,"x","logicalNot","bool")};return O.runKernel(Ca,e)}const Ju=M({logicalNot_:R$});function A$(n,t){const e=E(n,"a","logicalOr","bool"),s=E(t,"b","logicalOr","bool");mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(Ia,o)}const Xf=M({logicalOr_:A$});function D$(n,t){const e=E(n,"a","logicalXor","bool"),s=E(t,"b","logicalXor","bool");return mt(e.shape,s.shape),Kn(Xf(n,t),Ju(Kn(n,t)))}const F$=M({logicalXor_:D$});function O$(n,t,e,s,o){const r=E(n,"x","maxPool"),i=1;let a=r,l=!1;r.rank===3&&(l=!0,a=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),S(a.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),S($e(e,i),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${e} and dilations '${i}'`),Le("maxPool",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o},h=O.runKernel(va,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Qu=M({maxPool_:O$});function _$(n,t=[1,1,1],e,s,o,r="NDHWC"){const i=E(n,"x","maxPool3d");let a=i,l=!1;i.rank===4&&(l=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),S(a.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${a.rank}.`),S(r==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${r}`),Le("maxPool3d",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o,dataFormat:r},h=O.runKernel(Sa,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const L$=M({maxPool3d_:_$});function M$(n,t){let e=E(n,"a","maximum"),s=E(t,"b","maximum");[e,s]=Zt(e,s),e.dtype==="bool"&&(e=nt(e,"int32"),s=nt(s,"int32")),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(Tr,o)}const hs=M({maximum_:M$});function P$(n,t=null,e=!1){const o={x:E(n,"x","mean")},r={axis:t,keepDims:e};return O.runKernel(Na,o,r)}const se=M({mean_:P$});function pe(n,t="float32"){if(Un(n),t==="complex64"){const s=pe(n,"float32"),o=pe(n,"float32");return vo(s,o)}const e=Ie(q(n),t);return O.makeTensor(e,n,t)}function ds(n,t="float32"){if(Un(n),t==="complex64"){const s=ds(n,"float32"),o=pe(n,"float32");return vo(s,o)}const e=Ac(q(n),t);return O.makeTensor(e,n,t)}function B$(n,t){let e=E(n,"a","minimum"),s=E(t,"b","minimum");[e,s]=Zt(e,s),e.dtype==="bool"&&(e=nt(e,"int32"),s=nt(s,"int32")),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(Er,o)}const ci=M({minimum_:B$});function z$(n,t,e){S(e==="reflect"||e==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${e}.`);const s=E(n,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");S(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const o=e==="reflect"?1:0;for(let a=0;a<s.rank;a++)S(t[a].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),S(t[a][0]>=0&&t[a][0]<=s.shape[a]-o&&t[a][1]>=0&&t[a][1]<=s.shape[a]-o,()=>`Padding in dimension ${a} cannot be greater than or equal to ${s.shape[a]-o} or less than 0 for input of shape ${s.shape}`);const r={paddings:t,mode:e},i={x:s};return O.runKernel(Ea,i,r)}const V$=M({mirrorPad_:z$});function W$(n,t){let e=E(n,"a","mod"),s=E(t,"b","mod");[e,s]=Zt(e,s);const o={a:e,b:s};return O.runKernel(Rr,o)}const U$=M({mod_:W$});function G$(n,t=null,e=!1){n=E(n,"x","moments");const s=yt(t,n.shape),o=se(n,s,e);let r=o.shape;e||(r=ne(o.shape,s));const i=Vt(pt(nt(n,"float32"),L(o,r))),a=se(i,s,e);return{mean:o,variance:a}}const th=M({moments_:G$});function H$(n,t){let e=E(n,"a","notEqual","string_or_numeric"),s=E(t,"b","notEqual","string_or_numeric");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s};return O.runKernel(Aa,o)}const dl=M({notEqual_:H$});function q$(n,t,e=1,s=0,o="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const i={indices:E(n,"indices","oneHot","int32")},a={dtype:o,depth:t,onValue:e,offValue:s};return O.runKernel(Fa,i,a)}const Yf=M({oneHot_:q$});function K$(n){const e={x:E(n,"x","onesLike")};return O.runKernel(Da,e)}const sn=M({onesLike_:K$});function j$(n,t,e=0){const s=E(n,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const o={paddings:t,constantValue:e},r={x:s};return O.runKernel(_a,r,o)}const eh=M({pad_:j$});function X$(n,t,e){const s=E(n,"x","spaceToBatchND");S(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),S(e.length===t.length,()=>`paddings.shape[0] ${e.length} must be equal to [blockShape] ${t.length}`),S(s.shape.reduce((i,a,l)=>l>0&&l<=t.length?i&&(a+e[l-1][0]+e[l-1][1])%t[l-1]===0:i,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${e.toString()} must be divisible by blockShapes ${t.toString()}`);const o={x:s},r={blockShape:t,paddings:e};return O.runKernel(Ha,o,r)}const nh=M({spaceToBatchND_:X$});function Y$(n,t,e,s,o,r,i){o==null&&(o=[1,1]),r==null&&(r=1),s===0&&(s="valid");const a=E(n,"x","maxPool");let l=a,c=!1;a.rank===3&&(c=!0,l=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),S($e(r,o),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${r} and dilations '${o}'`);const u=nn(l.shape,t,r,o,s),h=[u.dilationHeight,u.dilationWidth];let d;s==="same"?d=J$([u.filterHeight,u.filterWidth],h):d=[[0,0],[0,0]];const p=h[0]===1&&h[1]===1,[f,m]=Z$([u.inHeight,u.inWidth],h,d),g=p?s:"valid",x=p?l:nh(l,h,f),w=(e==="avg"?()=>Wu(x,t,r,g,i):()=>Qu(x,t,r,g,i))(),y=p?w:Uu(w,h,m);return c?L(y,[y.shape[1],y.shape[2],y.shape[3]]):y}function Z$(n,t,e){const s=e.map(u=>u[0]),o=e.map(u=>u[1]),r=n.concat(s,o),i=t.map((u,h)=>(u-r[h]%u)%u),a=o.map((u,h)=>u+i[h]),l=t.map((u,h)=>[s[h],a[h]]),c=t.map((u,h)=>[0,i[h]]);return[l,c]}function J$(n,t){const s=n.map((i,a)=>i+(i-1)*(t[a]-1)).map(i=>i-1),o=s.map(i=>Math.floor(i/2)),r=s.map((i,a)=>i-o[a]);return s.map((i,a)=>[o[a],r[a]])}const Q$=M({pool_:Y$});function tk(n,t){const e=E(n,"x","prelu"),s=E(t,"alpha","prelu"),o={x:e,alpha:s};return O.runKernel(La,o)}const sh=M({prelu_:tk});function ek(n,t=null,e=!1){let s=E(n,"x","prod");s.dtype==="bool"&&(s=nt(s,"int32"));const o={x:s},r={axis:t,keepDims:e};return O.runKernel(Ma,o,r)}const nk=M({prod_:ek});var pl={exports:{}},sk=pl.exports,Zf;function ok(){return Zf||(Zf=1,(function(n){(function(t,e,s){function o(l){var c=this,u=a();c.next=function(){var h=2091639*c.s0+c.c*23283064365386963e-26;return c.s0=c.s1,c.s1=c.s2,c.s2=h-(c.c=h|0)},c.c=1,c.s0=u(" "),c.s1=u(" "),c.s2=u(" "),c.s0-=u(l),c.s0<0&&(c.s0+=1),c.s1-=u(l),c.s1<0&&(c.s1+=1),c.s2-=u(l),c.s2<0&&(c.s2+=1),u=null}function r(l,c){return c.c=l.c,c.s0=l.s0,c.s1=l.s1,c.s2=l.s2,c}function i(l,c){var u=new o(l),h=c&&c.state,d=u.next;return d.int32=function(){return u.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,h&&(typeof h=="object"&&r(h,u),d.state=function(){return r(u,{})}),d}function a(){var l=4022871197,c=function(u){u=String(u);for(var h=0;h<u.length;h++){l+=u.charCodeAt(h);var d=.02519603282416938*l;l=d>>>0,d-=l,d*=l,l=d>>>0,d-=l,l+=d*4294967296}return(l>>>0)*23283064365386963e-26};return c}e&&e.exports?e.exports=i:this.alea=i})(sk,n)})(pl)),pl.exports}var fl={exports:{}},rk=fl.exports,Jf;function ik(){return Jf||(Jf=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.x=0,l.y=0,l.z=0,l.w=0,l.next=function(){var h=l.x^l.x<<11;return l.x=l.y,l.y=l.z,l.z=l.w,l.w^=l.w>>>19^h^h>>>8},a===(a|0)?l.x=a:c+=a;for(var u=0;u<c.length+64;u++)l.x^=c.charCodeAt(u)|0,l.next()}function r(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xor128=i})(rk,n)})(fl)),fl.exports}var ml={exports:{}},ak=ml.exports,Qf;function lk(){return Qf||(Qf=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.next=function(){var h=l.x^l.x>>>2;return l.x=l.y,l.y=l.z,l.z=l.w,l.w=l.v,(l.d=l.d+362437|0)+(l.v=l.v^l.v<<4^(h^h<<1))|0},l.x=0,l.y=0,l.z=0,l.w=0,l.v=0,a===(a|0)?l.x=a:c+=a;for(var u=0;u<c.length+64;u++)l.x^=c.charCodeAt(u)|0,u==c.length&&(l.d=l.x<<10^l.x>>>4),l.next()}function r(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l.v=a.v,l.d=a.d,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xorwow=i})(ak,n)})(ml)),ml.exports}var gl={exports:{}},ck=gl.exports,tm;function uk(){return tm||(tm=1,(function(n){(function(t,e,s){function o(a){var l=this;l.next=function(){var u=l.x,h=l.i,d,p;return d=u[h],d^=d>>>7,p=d^d<<24,d=u[h+1&7],p^=d^d>>>10,d=u[h+3&7],p^=d^d>>>3,d=u[h+4&7],p^=d^d<<7,d=u[h+7&7],d=d^d<<13,p^=d^d<<9,u[h]=p,l.i=h+1&7,p};function c(u,h){var d,p=[];if(h===(h|0))p[0]=h;else for(h=""+h,d=0;d<h.length;++d)p[d&7]=p[d&7]<<15^h.charCodeAt(d)+p[d+1&7]<<13;for(;p.length<8;)p.push(0);for(d=0;d<8&&p[d]===0;++d);for(d==8?p[7]=-1:p[d],u.x=p,u.i=0,d=256;d>0;--d)u.next()}c(l,a)}function r(a,l){return l.x=a.x.slice(),l.i=a.i,l}function i(a,l){a==null&&(a=+new Date);var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(u.x&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xorshift7=i})(ck,n)})(gl)),gl.exports}var xl={exports:{}},hk=xl.exports,em;function dk(){return em||(em=1,(function(n){(function(t,e,s){function o(a){var l=this;l.next=function(){var u=l.w,h=l.X,d=l.i,p,f;return l.w=u=u+1640531527|0,f=h[d+34&127],p=h[d=d+1&127],f^=f<<13,p^=p<<17,f^=f>>>15,p^=p>>>12,f=h[d]=f^p,l.i=d,f+(u^u>>>16)|0};function c(u,h){var d,p,f,m,g,x=[],b=128;for(h===(h|0)?(p=h,h=null):(h=h+"\0",p=0,b=Math.max(b,h.length)),f=0,m=-32;m<b;++m)h&&(p^=h.charCodeAt((m+32)%h.length)),m===0&&(g=p),p^=p<<10,p^=p>>>15,p^=p<<4,p^=p>>>13,m>=0&&(g=g+1640531527|0,d=x[m&127]^=p+g,f=d==0?f+1:0);for(f>=128&&(x[(h&&h.length||0)&127]=-1),f=127,m=512;m>0;--m)p=x[f+34&127],d=x[f=f+1&127],p^=p<<13,d^=d<<17,p^=p>>>15,d^=d>>>12,x[f]=p^d;u.w=g,u.X=x,u.i=f}c(l,a)}function r(a,l){return l.i=a.i,l.w=a.w,l.X=a.X.slice(),l}function i(a,l){a==null&&(a=+new Date);var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(u.X&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xor4096=i})(hk,n)})(xl)),xl.exports}var bl={exports:{}},pk=bl.exports,nm;function fk(){return nm||(nm=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.next=function(){var h=l.b,d=l.c,p=l.d,f=l.a;return h=h<<25^h>>>7^d,d=d-p|0,p=p<<24^p>>>8^f,f=f-h|0,l.b=h=h<<20^h>>>12^d,l.c=d=d-p|0,l.d=p<<16^d>>>16^f,l.a=f-h|0},l.a=0,l.b=0,l.c=-1640531527,l.d=1367130551,a===Math.floor(a)?(l.a=a/4294967296|0,l.b=a|0):c+=a;for(var u=0;u<c.length+20;u++)l.b^=c.charCodeAt(u)|0,l.next()}function r(a,l){return l.a=a.a,l.b=a.b,l.c=a.c,l.d=a.d,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.tychei=i})(pk,n)})(bl)),bl.exports}var yl={exports:{}},mk={},gk=Object.freeze({__proto__:null,default:mk}),xk=hw(gk),bk=yl.exports,sm;function yk(){return sm||(sm=1,(function(n){(function(t,e,s){var o=256,r=6,i=52,a="random",l=s.pow(o,r),c=s.pow(2,i),u=c*2,h=o-1,d;function p(y,C,$){var N=[];C=C==!0?{entropy:!0}:C||{};var T=x(g(C.entropy?[y,w(e)]:y??b(),3),N),k=new f(N),v=function(){for(var I=k.g(r),R=l,A=0;I<c;)I=(I+A)*o,R*=o,A=k.g(1);for(;I>=u;)I/=2,R/=2,A>>>=1;return(I+A)/R};return v.int32=function(){return k.g(4)|0},v.quick=function(){return k.g(4)/4294967296},v.double=v,x(w(k.S),e),(C.pass||$||function(I,R,A,P){return P&&(P.S&&m(P,k),I.state=function(){return m(k,{})}),A?(s[a]=I,R):I})(v,T,"global"in C?C.global:this==s,C.state)}function f(y){var C,$=y.length,N=this,T=0,k=N.i=N.j=0,v=N.S=[];for($||(y=[$++]);T<o;)v[T]=T++;for(T=0;T<o;T++)v[T]=v[k=h&k+y[T%$]+(C=v[T])],v[k]=C;(N.g=function(I){for(var R,A=0,P=N.i,_=N.j,B=N.S;I--;)R=B[P=h&P+1],A=A*o+B[h&(B[P]=B[_=h&_+R])+(B[_]=R)];return N.i=P,N.j=_,A})(o)}function m(y,C){return C.i=y.i,C.j=y.j,C.S=y.S.slice(),C}function g(y,C){var $=[],N=typeof y,T;if(C&&N=="object")for(T in y)try{$.push(g(y[T],C-1))}catch{}return $.length?$:N=="string"?y:y+"\0"}function x(y,C){for(var $=y+"",N,T=0;T<$.length;)C[h&T]=h&(N^=C[h&T]*19)+$.charCodeAt(T++);return w(C)}function b(){try{var y;return d&&(y=d.randomBytes)?y=y(o):(y=new Uint8Array(o),(t.crypto||t.msCrypto).getRandomValues(y)),w(y)}catch{var C=t.navigator,$=C&&C.plugins;return[+new Date,t,$,t.screen,w(e)]}}function w(y){return String.fromCharCode.apply(0,y)}if(x(s.random(),e),n.exports){n.exports=p;try{d=xk}catch{}}else s["seed"+a]=p})(typeof self<"u"?self:bk,[],Math)})(yl)),yl.exports}var oh,om;function wk(){if(om)return oh;om=1;var n=ok(),t=ik(),e=lk(),s=uk(),o=dk(),r=fk(),i=yk();return i.alea=n,i.xor128=t,i.xorwow=e,i.xorshift7=s,i.xor4096=o,i.tychei=r,oh=i,oh}var rh=wk();class rm{constructor(t,e,s,o,r){this.mean=t,this.stdDev=e,this.dtype=s,this.nextVal=NaN,this.truncated=o,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const i=r||Math.random();this.random=rh.alea(i.toString())}nextValue(){if(!isNaN(this.nextVal)){const o=this.nextVal;return this.nextVal=NaN,o}let t,e,s=!1;for(;!s;){let o,r,i;do o=2*this.random()-1,r=2*this.random()-1,i=o*o+r*r;while(i>=1||i===0);const a=Math.sqrt(-2*Math.log(i)/i);t=this.mean+this.stdDev*o*a,e=this.mean+this.stdDev*r*a,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(e))&&(this.nextVal=this.convertValue(e)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class Ck{constructor(t=0,e=1,s,o){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=e-t,this.dtype=s,o==null&&(o=Math.random()),typeof o=="number"&&(o=o.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${e} <= 1 and dtype is not float`);this.random=rh.alea(o)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}function Ik(n,t=0,e=1,s,o){if(Un(n),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const r=new rm(t,e,s,!1,o),i=wt(n,s);for(let a=0;a<i.values.length;a++)i.values[a]=r.nextValue();return i.toTensor()}const $k=M({randomNormal_:Ik});function kk(n,t=0,e=1,s="float32",o){Un(n);const r=wt(n,s),i=new Ck(t,e,null,o);for(let a=0;a<r.values.length;a++)r.values[a]=i.nextValue();return r.toTensor()}const ui=M({randomUniform_:kk});function hi(n,t,e=1,s="float32"){if(e===0)throw new Error("Cannot have a step of zero");const o={start:n,stop:t,step:e,dtype:s};return O.runKernel(pu,{},o)}function vk(n){const e={input:E(n,"input","real")};return O.runKernel(fu,e)}const wl=M({real_:vk});function Sk(n){const e={x:E(n,"x","reciprocal")};return O.runKernel(Fr,e)}const Nk=M({reciprocal_:Sk});function Tk(n){const e={x:E(n,"x","relu")};return O.runKernel(Or,e)}const Hs=M({relu_:Tk});function Ek(n){const e={x:E(n,"x","relu6")};return O.runKernel(_r,e)}const im=M({relu6_:Ek});function Rk(n,t){const s={x:E(n,"x","reverse")},o={dims:t};return O.runKernel(Va,s,o)}const qs=M({reverse_:Rk});function Ak(n){const e={x:E(n,"x","round")};return O.runKernel(Lr,e)}const am=M({round_:Ak});function Dk(n){const e={x:E(n,"x","rsqrt","float32")};return O.runKernel(Mr,e)}const lm=M({rsqrt_:Dk});function Fk(n){const e={x:E(n,"x","selu")};return O.runKernel(Pr,e)}const cm=M({selu_:Fk});function Ok(n,t,e,s,o,r=[1,1],i="NHWC"){const a=E(n,"x","separableConv2d"),l=E(t,"depthwiseFilter","separableConv2d"),c=E(e,"pointwiseFilter","separableConv2d");let u=a,h=!1;if(a.rank===3&&(h=!0,u=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),i==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");S(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),S(l.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),S(c.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),S(c.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),S(c.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const d=l.shape[2],p=l.shape[3];S(c.shape[2]===d*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*p}, but got ${c.shape[2]}.`);const f=Ku(u,l,s,o,i,r),g=Ws(f,c,1,"valid",i);return h?L(g,[g.shape[1],g.shape[2],g.shape[3]]):g}const um=M({separableConv2d_:Ok});function _k(n){const e={x:E(n,"x","sign")};return O.runKernel(Vr,e)}const Lk=M({sign_:_k});function Mk(n){const e={x:E(n,"x","sin","float32")};return O.runKernel(Br,e)}const hm=M({sin_:Mk});function Pk(n){const e={x:E(n,"x","sinh")};return O.runKernel(zr,e)}const dm=M({sinh_:Pk});function Bk(n,t,e){const s=E(n,"x","slice1d");return S(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),Pt(s,[t],[e])}const ih=M({slice1d_:Bk});function zk(n,t,e){const s=E(n,"x","slice2d");return S(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),Pt(s,t,e)}const pm=M({slice2d_:zk});function Vk(n,t,e){const s=E(n,"x","slice3d");return S(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),Pt(s,t,e)}const ah=M({slice3d_:Vk});function Wk(n,t,e){const s=E(n,"x","slice4d");return S(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),Pt(s,t,e)}const Cl=M({slice4d_:Wk});function Uk(n,t=-1){const e=E(n,"logits","softmax","float32");if(t===-1&&(t=e.rank-1),t!==e.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${e.rank} and dim was ${t}`);const s={logits:e},o={dim:t};return O.runKernel(Ka,s,o)}const lh=M({softmax_:Uk});function Gk(n){S(n.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${n.dtype}.`);const t={input:n};return O.runKernel(nu,t)}const fm=M({fft_:Gk});function Hk(n){S(n.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${n.dtype}.`);const t={input:n};return O.runKernel(ru,t)}const ch=M({ifft_:Hk});function qk(n){const t=n.shape[n.shape.length-1],e=n.size/t;let s;if(t<=2){const o=L(n,[e,t]);s=ch(o)}else{const o=[e,2*(t-1)],r=L(wl(n),[e,t]),i=L(Yu(n),[e,t]),a=qs(Pt(r,[0,1],[e,t-2]),1),l=F(qs(Pt(i,[0,1],[e,t-2]),1),At(-1)),c=Me([r,a],1),u=Me([i,l],1),h=L(vo(c,u),[o[0],o[1]]);s=ch(h)}if(s=wl(s),n.rank===3&&n.shape[0]!==0){const o=s,r=n.shape[0];s=L(s,[r,s.shape[0]/r,s.shape[1]]),o.dispose()}return s}const Kk=M({irfft_:qk});function jk(n,t,e=0){const o={x:E(n,"x","split")},r={numOrSizeSplits:t,axis:e};return O.runKernel(qa,o,r)}const Ze=M({split_:jk});function Xk(n,t){S(n.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${n.dtype}`);let e=n.shape[n.shape.length-1];const s=n.size/e;let o;if(t!=null&&t<e){const f=n.shape.map(g=>0),m=n.shape.map(g=>g);m[n.shape.length-1]=t,o=Pt(n,f,m),e=t}else if(t!=null&&t>e){const f=n.shape.map(m=>m);f[n.shape.length-1]=t-e,o=Me([n,pe(f)],n.shape.length-1),e=t}else o=n;const r=kt(o),i=L(vo(o,r),[s,e]),a=fm(i),l=Math.floor(e/2)+1,c=wl(a),u=Yu(a),h=Ze(c,[l,e-l],c.shape.length-1),d=Ze(u,[l,e-l],u.shape.length-1),p=o.shape.slice();return p[o.shape.length-1]=l,L(vo(h[0],d[0]),p)}const Yk=M({rfft_:Xk});function Zk(n,t){let e=E(n,"a","squaredDifference"),s=E(t,"b","squaredDifference");[e,s]=Zt(e,s),mt(e.shape,s.shape);const o={a:e,b:s},r={};return O.runKernel(Hr,o,r)}const Jk=M({squaredDifference_:Zk});function Qk(n,t){const e=E(n,"x","squeeze","string_or_numeric");return L(e,ss(e.shape,t).newShape)}const di=M({squeeze_:Qk});function tv(n,t=0){const e=bf(n,"tensors","stack","string_or_numeric");S(e.length>=1,()=>"Pass at least one tensor to tf.stack"),e.length>0&&S(t<=e[0].rank,()=>"Axis must be <= rank of the tensor");const s=e,o={axis:t};return O.runKernel(Oa,s,o)}const jn=M({stack_:tv});function ev(n,t=0){const s={x:E(n,"x","step")},o={alpha:t};return O.runKernel(Yr,s,o)}const pi=M({step_:ev});function nv(n,t,e,s,o=0,r=0,i=0,a=0,l=0){const u={x:E(n,"x","stridedSlice","string_or_numeric")},h={begin:t,end:e,strides:s,beginMask:o,endMask:r,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:l};return O.runKernel(yu,u,h)}const sv=M({stridedSlice_:nv});function ov(n){const e={x:E(n,"x","tan","float32")};return O.runKernel(Kr,e)}const rv=M({tan_:ov});function Ge(n,t){Sc(n);const e=ti(n,t);if(e.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return ei(n,null,e,t)}function Fo(n,t,e){if(Sc(n),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=ti(n,e);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return ei(n,t,s,e)}function Il(n,t,e){if(Sc(n),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=ti(n,e);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return ei(n,t,s,e)}function mm(n,t,e){const s=t.rank>1?t.shape[t.rank-1]:1,o=t.rank>1?t.rank-1:1,r=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${e.shape}, indices.shape: ${t.shape}, shape: ${n}, sliceDim: ${s}, and batchDim: ${o}.`;if(e.rank<o)throw new Error(r+` update.rank < ${o}. `);if(n.length<s+(e.rank-o))throw new Error(r+` Output shape length < ${s+(e.rank-o)}`);if(e.rank!==o+n.length-s)throw new Error(r+` update.rank != ${o+n.length-s}`);for(let i=0;i<o;++i)if(e.shape[i]!==t.shape[i])throw new Error(r+` updates.shape[${i}] (${e.shape[i]}) != indices.shape[${i}] (${t.shape[i]}).`);for(let i=0;i<e.rank-o;++i)if(e.shape[i+o]!==n[i+s])throw new Error(r+` updates.shape[${i+o}] (${e.shape[i+o]}) != shape[${i+o}] (${n[i+o]})`)}function iv(n,t,e){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(n.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${n.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(e.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${e}`);if(e.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(n.size===0)throw new Error(`Updates specified for empty output. updates shape: ${n.shape}`)}mm(e,t,n)}function Ks(n,t,e){const s=t.shape.length,o=s>1?t.shape[s-1]:1,r=e.length;let i=1;for(let h=o;h<r;++h)i*=e[h];const a=o<1?1:o,l=q(t.shape)/a,c=[...lt(e.slice(0,o)),1],u=q(e);return{sliceRank:o,numUpdates:l,sliceSize:i,strides:c,outputSize:u}}function av(n,t=1,e=!0){const s=E(n,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const o=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>o)throw new Error(`'k' passed to topk() must be <= the last dimension (${o}) but got ${t}`);const r={x:s},i={k:t,sorted:e},[a,l]=O.runKernel(wu,r,i);return{values:a,indices:l}}const lv=M({topk_:av});function cv(n,t=0,e=1,s,o){if(Un(n),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const r=new rm(t,e,s,!0,o),i=wt(n,s);for(let a=0;a<i.values.length;a++)i.values[a]=r.nextValue();return i.toTensor()}const gm=M({truncatedNormal_:cv});function uv(n,t=0){const e=E(n,"x","unique","string_or_numeric");S(e.rank>0,()=>"The input tensor must be at least 1D");const s={x:e},o={axis:t},[r,i]=O.runKernel(Iu,s,o);return{values:r,indices:i}}const hv=M({unique_:uv});function dv(n,t,e){const s=E(n,"x","unsortedSegmentSum"),o=E(t,"segmentIds","unsortedSegmentSum","int32");S(xo(e),()=>"numSegments must be of dtype int");const r={x:s,segmentIds:o},i={numSegments:e};return O.runKernel(Xa,r,i)}const xm=M({unsortedSegmentSum_:dv});function pv(n,t=0){const e=E(n,"x","unstack","string_or_numeric");S(t>=-e.shape.length&&t<e.shape.length,()=>`Axis = ${t} is not in [-${e.shape.length}, ${e.shape.length})`);const s={value:e},o={axis:t};return O.runKernel(ja,s,o)}const js=M({unstack_:pv});function fv(n,t=!0,e,s){return O.makeVariable(n,t,e,s)}function bm(n,t){const e=[];for(let r=0;r<t.length;r++)t[r]&&e.push(r);const s=wt(n,"int32"),o=wt([e.length,n.length],"int32");for(let r=0;r<e.length;r++){const i=s.indexToLoc(e[r]),a=r*n.length;o.values.set(i,a)}return o.toTensor()}function mv(n,t,e){const s=E(n,"x","transpose");if(t==null&&(t=s.shape.map((i,a)=>a).reverse()),S(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(i=>{S(i>=0&&i<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const o={x:s},r={perm:t};return s.dtype==="complex64"?V(()=>{let i=wl(s),a=Yu(s);return i=O.runKernel(Io,{x:i},r),a=O.runKernel(Io,{x:a},r),e&&(a=Qt(a)),vo(i,a)}):O.runKernel(Io,o,r)}const vt=M({transpose_:mv});function gv(n,t){if(t==null)return n.shape.slice();if(Et(n.shape,t))return t;if(n.shape.length===t.length){const e=[];for(let s=0;s<n.shape.length;s++)t[s]==null&&n.shape[s]!=null?e.push(n.shape[s]):e.push(t[s]);return e}return t}function xv(n,t,e,s){const o=E(n,"x","dropout");if(S(o.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${o.dtype} tensor instead.`),S(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return n instanceof oe?o.clone():o;const r=gv(o,e),i=1-t,a=dt(ul(J(ui(r,0,1,"float32",s),i)),i);return F(o,a)}const bv=M({dropout_:xv});function yv(n,t,e,s,o,r="NHWC",i){let a=n;n.rank===3&&(a=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]));let l=t;l.rank===3&&(l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]])),S(a.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${a.shape}.`),S(l.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${l.shape}.`),S(e.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${e}.`);const c=r==="NHWC"?a.shape[3]:a.shape[1],u=r==="NHWC"?l.shape[3]:l.shape[1];S(c===e[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${e[2]}.`),S(u===e[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${e[3]}).`),Le("conv2dDerFilter",o,i);const h={x:a,dy:l},d={strides:s,pad:o,dataFormat:r,dimRoundingMode:i,filterShape:e};return O.runKernel(Uc,h,d)}const uh=M({conv2DBackpropFilter_:yv});function hh(n,t,e){if(e==null||e==="linear")return n;if(e==="relu")return F(n,pi(t));throw new Error(`Cannot compute gradient for fused activation ${e}.`)}function dh(n,t){let e=t;const s=re(n.shape,t.shape);return s.length>0&&(e=ut(e,s)),L(e,n.shape)}function ph(n,t,e,s){if(t==="linear")return n;if(t==="relu")return Hs(n);if(t==="elu")return al(n);if(t==="relu6")return im(n);if(t==="prelu")return sh(n,e);if(t==="leakyrelu")return Zu(n,s);if(t==="sigmoid")return Eo(n);throw new Error(`Unknown fused activation ${t}.`)}const fh=(n,t)=>!(n>0)||t==="linear";function wv({x:n,filter:t,strides:e,pad:s,dataFormat:o="NHWC",dilations:r=[1,1],dimRoundingMode:i,bias:a,activation:l="linear",preluActivationWeights:c,leakyreluAlpha:u}){if(l=l||"linear",fh(O.state.gradientDepth,l)===!1){S(o==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${o} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let $=Ws(n,t,e,s,o,r,i);return a!=null&&($=J($,a)),ph($,l,c,u)}const h=E(n,"x","conv2d","float32"),d=E(t,"filter","conv2d","float32");let p=h,f=!1;h.rank===3&&(f=!0,p=L(h,[1,h.shape[0],h.shape[1],h.shape[2]])),S(p.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),S(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),Le("fused conv2d",s,i);const m=o==="NHWC"?p.shape[3]:p.shape[1];S(d.shape[2]===m,()=>`Error in conv2d: depth of input (${m}) must match input depth for filter ${d.shape[2]}.`),S($e(e,r),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`);const g=ge(p.shape,d.shape,e,r,s,i);let x;a!=null&&(x=E(a,"bias","fused conv2d"),[x]=Zt(x,h),o==="NHWC"?mt(g.outShape,x.shape):(S(x.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${x.shape.length}.`),S(x.shape.length===0||x.shape[0]===g.outChannels||x.shape[0]===1,()=>`Error in fused conv2d: bias shape (${x.shape}) is not compatible with the number of output channels (${g.outChannels})`)));let b;if(c!=null){const $=c.shape;if(S($.length<=1||$.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${$.length}.`),$.length===1)S($[0]===1||$[0]===g.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${$}) is not compatible with the number of output channels (${g.outChannels}).`);else if($.length===3)try{mt($,g.outShape)}catch{const T=`Error in fused conv2d: PReLU activation weights (${$}) is not compatible with the output shape of the conv2d (${g.outShape}).`;throw Error(T)}b=E(c,"prelu weights","fused conv2d")}const w=($,N)=>{S(o==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${o} but only NHWC is currently supported.`);const[T,k,v,I]=N,R=hh($,v,l);S(zs(r),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`);const A=Gu(k.shape,R,T,e,s),P=uh(k,R,T.shape,e,s),_=[A,P];if(I!=null){const B=dh(I,R);_.push(B)}return _},y={x:p,filter:d,bias:x,preluActivationWeights:b},C={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i,activation:l,leakyreluAlpha:u};return a==null?Do((N,T,k)=>{let v=O.runKernel(Ja,y,C);return k([T,N,v]),f&&(v=L(v,[v.shape[1],v.shape[2],v.shape[3]])),{value:v,gradFunc:w}})(p,d):Do((N,T,k,v)=>{let I=O.runKernel(Ja,y,C);return v([T,N,I,k]),f&&(I=L(I,[I.shape[1],I.shape[2],I.shape[3]])),{value:I,gradFunc:w}})(p,d,x)}const Cv=M({fusedConv2d_:wv});function Iv(n,t,e,s,o,r=[1,1],i){let a=n;n.rank===3&&(a=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]));let l=t;l.rank===3&&(l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={x:a,dy:l},u={strides:s,pad:o,dimRoundingMode:i,dilations:r,filterShape:e};return O.runKernel(Yc,c,u)}const $v=M({depthwiseConv2dNativeBackpropFilter_:Iv});function kv(n,t,e,s,o,r=[1,1],i){let a=t,l=!1;t.rank===3&&(l=!0,a=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={dy:a,filter:e},u={strides:s,pad:o,dimRoundingMode:i,dilations:r,inputShape:n},h=O.runKernel(Zc,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const vv=M({depthwiseConv2dNativeBackpropInput_:kv});function Sv({a:n,b:t,transposeA:e=!1,transposeB:s=!1,bias:o,activation:r="linear",preluActivationWeights:i,leakyreluAlpha:a=.2}){if(fh(O.state.gradientDepth,r)===!1){let I=Rt(n,t,e,s);return o!=null&&(I=J(I,o)),ph(I,r,i,a)}let l=E(n,"a","fused matMul"),c=E(t,"b","fused matMul");[l,c]=Zt(l,c);const u=e?l.shape[l.rank-2]:l.shape[l.rank-1],h=s?c.shape[c.rank-1]:c.shape[c.rank-2],d=e?l.shape[l.rank-1]:l.shape[l.rank-2],p=s?c.shape[c.rank-2]:c.shape[c.rank-1],f=l.shape.slice(0,-2),m=c.shape.slice(0,-2),g=q(f),x=q(m);S(u===h,()=>`Error in fused matMul: inner shapes (${u}) and (${h}) of Tensors with shapes ${l.shape} and ${c.shape} and transposeA=${e} and transposeB=${s} must match.`);const w=mt(l.shape.slice(0,-2),c.shape.slice(0,-2)).concat([d,p]),y=e?L(l,[g,u,d]):L(l,[g,d,u]),C=s?L(c,[x,p,h]):L(c,[x,h,p]);let $;o!=null&&($=E(o,"bias","fused matMul"),[$]=Zt($,l),mt(w,$.shape));let N;i!=null&&(N=E(i,"prelu weights","fused matMul"));const T=(I,R)=>{const[A,P,_,B]=R,z=hh(L(I,_.shape),_,r);let W,H;if(!e&&!s?(W=Rt(z,P,!1,!0),H=Rt(A,z,!0,!1)):!e&&s?(W=Rt(z,P,!1,!1),H=Rt(z,A,!0,!1)):e&&!s?(W=Rt(P,z,!1,!0),H=Rt(A,z,!1,!1)):(W=Rt(P,z,!0,!0),H=Rt(z,A,!0,!0)),o!=null){const K=dh(B,z);return[W,H,K]}else return[W,H]},k={a:y,b:C,bias:$,preluActivationWeights:N},v={transposeA:e,transposeB:s,activation:r,leakyreluAlpha:a};return o==null?Do((R,A,P)=>{const _=O.runKernel(Za,k,v);return P([R,A,_]),{value:L(_,w),gradFunc:T}})(y,C):Do((R,A,P,_)=>{const B=O.runKernel(Za,k,v);return _([R,A,B,P]),{value:L(B,w),gradFunc:T}})(y,C,$)}const ym=M({fusedMatMul_:Sv});function Nv(n,t,e,s,o="bilinear",r=0){const i=E(n,"image","cropAndResize"),a=E(t,"boxes","cropAndResize","float32"),l=E(e,"boxInd","cropAndResize","int32"),c=a.shape[0];S(i.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${i.rank}.`),S(a.rank===2&&a.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${a.shape}.`),S(l.rank===1&&l.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${a.shape}.`),S(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),S(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),S(o==="bilinear"||o==="nearest",()=>`method must be bilinear or nearest, but was ${o}`);const u={image:i,boxes:a,boxInd:l},h={method:o,extrapolationValue:r,cropSize:s};return O.runKernel(Kc,u,h)}const Tv=M({cropAndResize_:Nv});function Ev(n){const t=E(n,"image","flipLeftRight","float32");S(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const e={image:t};return O.runKernel(ou,e,{})}const Rv=M({flipLeftRight_:Ev});function Av(n){const t=E(n,"image","grayscaleToRGB"),e=t.rank-1,s=t.shape[e];S(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),S(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const o=new Array(t.rank);return o.fill(1,0,e),o[e]=3,gn(t,o)}const Dv=M({grayscaleToRGB_:Av});function Fv(n){const t=E(n,"image","RGBToGrayscale"),e=t.rank-1,s=t.shape[e];S(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),S(s===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${s}.`);const o=t.dtype,r=nt(t,"float32"),i=Ge([.2989,.587,.114]);let a;switch(t.rank){case 2:a=ai("ij,j->i",r,i);break;case 3:a=ai("ijk,k->ij",r,i);break;case 4:a=ai("ijkl,l->ijk",r,i);break;case 5:a=ai("ijklm,m->ijkl",r,i);break;case 6:a=ai("ijklmn,n->ijklm",r,i);break;default:throw new Error("Not a valid tensor rank.")}return a=Pe(a,-1),nt(a,o)}const Ov=M({rgbToGrayscale_:Fv});function _v(n,t,e=0,s=.5){const o=E(n,"image","rotateWithOffset","float32");S(o.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${o.rank}.`);const r={image:o},i={radians:t,fillValue:e,center:s};return O.runKernel($u,r,i)}const Lv=M({rotateWithOffset_:_v});function Oo(n,t,e,s,o,r){s==null&&(s=.5),o==null&&(o=Number.NEGATIVE_INFINITY),r==null&&(r=0);const i=n.shape[0];return e=Math.min(e,i),S(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),S(n.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${n.rank}'`),S(n.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${n.shape[1]}`),S(t.rank===1,()=>"scores must be a 1D tensor"),S(t.shape[0]===i,()=>`scores has incompatible shape with boxes. Expected ${i}, but was ${t.shape[0]}`),S(0<=r&&r<=1,()=>`softNmsSigma must be in [0, 1], but was '${r}'`),{maxOutputSize:e,iouThreshold:s,scoreThreshold:o,softNmsSigma:r}}function Mv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY){const r=E(n,"boxes","nonMaxSuppression","float32"),i=E(t,"scores","nonMaxSuppression","float32"),a=Oo(r,i,e,s,o);e=a.maxOutputSize,s=a.iouThreshold,o=a.scoreThreshold;const l={maxOutputSize:e,iouThreshold:s,scoreThreshold:o};return O.runKernel(uu,{boxes:r,scores:i},l)}const Pv=M({nonMaxSuppression_:Mv});function Bv(n,t,e){const s=zv(n,t,e),o=s<0?-(s+1):s;n.splice(o,0,t)}function zv(n,t,e){return Wv(n,t,e||Vv)}function Vv(n,t){return n>t?1:n<t?-1:0}function Wv(n,t,e){let s=0,o=n.length,r=0,i=!1;for(;s<o;){r=s+(o-s>>>1);const a=e(t,n[r]);a>0?s=r+1:(o=r,i=!a)}return i?s:-s-1}function mh(n,t,e,s,o){return bh(n,t,e,s,o,0)}function gh(n,t,e,s,o,r){return bh(n,t,e,s,o,0,!1,r,!0)}function xh(n,t,e,s,o,r){return bh(n,t,e,s,o,r,!0)}function bh(n,t,e,s,o,r,i=!1,a=!1,l=!1){const c=[];for(let g=0;g<t.length;g++)t[g]>o&&c.push({score:t[g],boxIndex:g,suppressBeginIndex:0});c.sort(wm);const u=r>0?-.5/r:0,h=[],d=[];for(;h.length<e&&c.length>0;){const g=c.pop(),{score:x,boxIndex:b,suppressBeginIndex:w}=g;if(x<o)break;let y=!1;for(let C=h.length-1;C>=w;--C){const $=Uv(n,b,h[C]);if($>=s){y=!0;break}if(g.score=g.score*Gv(s,u,$),g.score<=o)break}g.suppressBeginIndex=h.length,y||(g.score===x?(h.push(b),d.push(g.score)):g.score>o&&Bv(c,g,wm))}const p=h.length,f=e-p;a&&f>0&&(h.push(...new Array(f).fill(0)),d.push(...new Array(f).fill(0)));const m={selectedIndices:h};return i&&(m.selectedScores=d),l&&(m.validOutputs=p),m}function Uv(n,t,e){const s=n.subarray(t*4,t*4+4),o=n.subarray(e*4,e*4+4),r=Math.min(s[0],s[2]),i=Math.min(s[1],s[3]),a=Math.max(s[0],s[2]),l=Math.max(s[1],s[3]),c=Math.min(o[0],o[2]),u=Math.min(o[1],o[3]),h=Math.max(o[0],o[2]),d=Math.max(o[1],o[3]),p=(a-r)*(l-i),f=(h-c)*(d-u);if(p<=0||f<=0)return 0;const m=Math.max(r,c),g=Math.max(i,u),x=Math.min(a,h),b=Math.min(l,d),w=Math.max(x-m,0)*Math.max(b-g,0);return w/(p+f-w)}function Gv(n,t,e){const s=Math.exp(t*e*e);return e<=n?s:0}function wm(n,t){return n.score-t.score||n.score===t.score&&t.boxIndex-n.boxIndex}async function Hv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY){const r=E(n,"boxes","nonMaxSuppressionAsync"),i=E(t,"scores","nonMaxSuppressionAsync"),a=Oo(r,i,e,s,o);e=a.maxOutputSize,s=a.iouThreshold,o=a.scoreThreshold;const l=await Promise.all([r.data(),i.data()]),c=l[0],u=l[1],{selectedIndices:h}=mh(c,u,e,s,o);return r!==n&&r.dispose(),i!==t&&i.dispose(),Ge(h,"int32")}const qv=Hv;function Kv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=0){const i=E(n,"boxes","nonMaxSuppression"),a=E(t,"scores","nonMaxSuppression"),l=Oo(i,a,e,s,o,r);e=l.maxOutputSize,s=l.iouThreshold,o=l.scoreThreshold,r=l.softNmsSigma;const c={boxes:i,scores:a},u={maxOutputSize:e,iouThreshold:s,scoreThreshold:o,softNmsSigma:r},h=O.runKernel(du,c,u);return{selectedIndices:h[0],selectedScores:h[1]}}const jv=M({nonMaxSuppressionWithScore_:Kv});async function Xv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=0){const i=E(n,"boxes","nonMaxSuppressionAsync"),a=E(t,"scores","nonMaxSuppressionAsync"),l=Oo(i,a,e,s,o,r);e=l.maxOutputSize,s=l.iouThreshold,o=l.scoreThreshold,r=l.softNmsSigma;const c=await Promise.all([i.data(),a.data()]),u=c[0],h=c[1],{selectedIndices:d,selectedScores:p}=xh(u,h,e,s,o,r);return i!==n&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:Ge(d,"int32"),selectedScores:Ge(p)}}const Yv=Xv;function Zv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=!1){const i=E(n,"boxes","nonMaxSuppression"),a=E(t,"scores","nonMaxSuppression"),l=Oo(i,a,e,s,o,null),c=l.maxOutputSize,u=l.iouThreshold,h=l.scoreThreshold,d={boxes:i,scores:a},p={maxOutputSize:c,iouThreshold:u,scoreThreshold:h,padToMaxOutputSize:r},f=O.runKernel(hu,d,p);return{selectedIndices:f[0],validOutputs:f[1]}}const Jv=M({nonMaxSuppressionPadded_:Zv});async function Qv(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=!1){const i=E(n,"boxes","nonMaxSuppressionAsync"),a=E(t,"scores","nonMaxSuppressionAsync"),l=Oo(i,a,e,s,o,null),c=l.maxOutputSize,u=l.iouThreshold,h=l.scoreThreshold,[d,p]=await Promise.all([i.data(),a.data()]),{selectedIndices:f,validOutputs:m}=gh(d,p,c,u,h,r);return i!==n&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:Ge(f,"int32"),validOutputs:At(m,"int32")}}const tS=Qv;function eS(n,t,e=!1,s=!1){const o=E(n,"images","resizeBilinear");S(o.rank===3||o.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${o.rank}.`),S(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),S(s===!1||e===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let r=o,i=!1;o.rank===3&&(i=!0,r=L(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const a={images:r},l={alignCorners:e,halfPixelCenters:s,size:t},c=O.runKernel(za,a,l);return i?L(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Cm=M({resizeBilinear_:eS});function nS(n,t,e=!1,s=!1){const o=E(n,"images","resizeNearestNeighbor");S(o.rank===3||o.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${o.rank}.`),S(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),S(o.dtype==="float32"||o.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),S(s===!1||e===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let r=o,i=!1;o.rank===3&&(i=!0,r=L(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const a={images:r},l={alignCorners:e,halfPixelCenters:s,size:t},c=O.runKernel(Ba,a,l);return i?L(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Im=M({resizeNearestNeighbor_:nS});function sS(n,t="binary",e=!1,s=.5){const o=E(n,"image","threshold"),r=.2989,i=.587,a=.114,l=o.shape[0]*o.shape[1];let c=F(Ge([s]),255),u,h,d,p;if(S(o.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${o.rank}.`),S(o.shape[2]===3||o.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${o.shape[2]}.`),S(o.dtype==="int32"||o.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${o.dtype}.`),S(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),o.shape[2]===3){[u,h,d]=Ze(o,[1,1,1],-1);const g=F(u,r),x=F(h,i),b=F(d,a);p=J(J(g,x),b)}else p=n;if(t==="otsu"){const g=tI(nt(am(p),"int32"),yf([]),256);c=oS(g,l)}const f=e?Ao(p,c):Ye(p,c);return nt(F(f,255),"int32")}function oS(n,t){let e=Ge([-1]),s=Ge([0]),o=Ge([0]),r,i,a,l,c,u;for(let h=0;h<n.size-1;h++){r=Pt(n,0,h+1),i=Pt(n,h+1),c=dt(ut(r),t),u=dt(ut(i),t);const d=ut(F(r,hi(0,r.size)));a=dt(d,ut(r));const p=il(i.shape,r.size),f=J(hi(0,i.size),p),m=F(i,f);l=dt(ut(m),ut(i));const g=pt(a,l),x=pt(a,l),b=F(c,u);o=F(F(b,g),x);const w=Ye(o,s);s=Re(w,o,s),e=Re(w,Ge([h]),e)}return e}const rS=M({threshold_:sS});function iS(n,t,e="nearest",s="constant",o=0,r){const i=E(n,"image","transform","float32"),a=E(t,"transforms","transform","float32");S(i.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${i.rank}.`),S(a.rank===2&&(a.shape[0]===i.shape[0]||a.shape[0]===1)&&a.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),S(r==null||r.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${r}.`);const l={image:i,transforms:a},c={interpolation:e,fillMode:s,fillValue:o,outputShape:r};return O.runKernel(Cu,l,c)}const aS=M({transform_:iS});function lS(n,t,e){const s=E(n,"a","bandPart");S(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const o=s.shape,[r,i]=s.shape.slice(-2);let a,l;typeof t=="number"?(S(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),S(t<=r,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${r}).`),a=E(t<0?r:t,"numLower","bandPart")):(S(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),a=Re(hl(t,0),r,ci(t,r))),typeof e=="number"?(S(e%1===0,()=>`bandPart(): numUpper must be an integer, got ${e}.`),S(e<=i,()=>`bandPart(): numUpper (${e}) must not be greater than the number of columns (${i}).`),l=E(e<0?i:e,"numUpper","bandPart")):(S(e.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),l=Re(hl(e,0),i,ci(e,i)));const c=L(hi(0,r,1,"int32"),[-1,1]),u=hi(0,i,1,"int32"),h=pt(c,u),d=Kn(Ao(h,a),Gs(h,Qt(l))),p=pe([r,i],s.dtype);return L(jn(js(L(s,[-1,r,i])).map(f=>Re(d,f,p))),o)}const cS=M({bandPart_:lS});function uS(n){let t;if(Array.isArray(n)){t=!1,S(n!=null&&n.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const o=n[0].shape[0];for(let r=1;r<n.length;++r)S(n[r].shape[0]===o,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${n[r].shape[0]} vs. ${o})`)}else t=!0,n=Ze(n,n.shape[0],0).map(o=>di(o,[0]));S(n.length<=n[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${n.length}) exceeds number of dimensions (${n[0].shape[0]}).`);const e=[],s=n;for(let o=0;o<n.length;++o)e.push(O.tidy(()=>{let r=s[o];if(o>0)for(let i=0;i<o;++i){const a=F(ut(F(e[i],r)),e[i]);r=pt(r,a)}return dt(r,cl(r,"euclidean"))}));return t?jn(e,0):e}const hS=M({gramSchmidt_:uS});function dS(n,t=!1){if(S(n.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${n.rank}`),n.rank===2)return $m(n,t);{const e=n.shape.slice(0,n.shape.length-2).reduce((l,c)=>l*c),s=js(L(n,[e,n.shape[n.shape.length-2],n.shape[n.shape.length-1]]),0),o=[],r=[];s.forEach(l=>{const[c,u]=$m(l,t);o.push(c),r.push(u)});const i=L(jn(o,0),n.shape),a=L(jn(r,0),n.shape);return[i,a]}}function $m(n,t=!1){return O.tidy(()=>{S(n.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${n.shape.length}D Tensor.`);const e=n.shape[0],s=n.shape[1];let o=Hf(e),r=Bs(n);const i=Fo([[1]],[1,1]);let a=Bs(i);const l=e>=s?s:e;for(let c=0;c<l;++c){const u=r,h=a,d=o;[a,r,o]=O.tidy(()=>{const p=Pt(r,[c,c],[e-c,1]),f=cl(p),m=Pt(r,[c,c],[1,1]),g=Re(Ye(m,0),Fo([[-1]]),Fo([[1]])),x=pt(m,F(g,f)),b=dt(p,x);b.shape[0]===1?a=Bs(i):a=Me([i,Pt(b,[1,0],[b.shape[0]-1,b.shape[1]])],0);const w=Qt(dt(Rt(g,x),f)),y=Pt(r,[c,0],[e-c,s]),C=F(w,a),$=vt(a);if(c===0)r=pt(y,Rt(C,Rt($,y)));else{const k=pt(y,Rt(C,Rt($,y)));r=Me([Pt(r,[0,0],[c,s]),k],0)}const N=vt(C),T=Pt(o,[0,c],[e,o.shape[1]-c]);if(c===0)o=pt(T,Rt(Rt(T,a),N));else{const k=pt(T,Rt(Rt(T,a),N));o=Me([Pt(o,[0,0],[e,c]),k],1)}return[a,r,o]}),It([u,h,d])}return!t&&e>s&&(o=Pt(o,[0,0],[e,s]),r=Pt(r,[0,0],[s,s])),[o,r]})}const pS=M({qr_:dS});const Xn={flipLeftRight:Rv,grayscaleToRGB:Dv,resizeNearestNeighbor:Im,resizeBilinear:Cm,rgbToGrayscale:Ov,rotateWithOffset:Lv,cropAndResize:Tv,nonMaxSuppression:Pv,nonMaxSuppressionAsync:qv,nonMaxSuppressionWithScore:jv,nonMaxSuppressionWithScoreAsync:Yv,nonMaxSuppressionPadded:Jv,nonMaxSuppressionPaddedAsync:tS,threshold:rS,transform:aS},fS={bandPart:cS,gramSchmidt:hS,qr:pS};const mS=new Map,gS=new Map;class _o{getClassName(){return this.constructor.className}static fromConfig(t,e){return new t(e)}}class on{constructor(){this.classNameMap={}}static getMap(){return on.instance==null&&(on.instance=new on),on.instance}static register(t){on.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function Z(n,t,e){S(n.className!=null,()=>"Class being registered does not have the static className property defined."),S(typeof n.className=="string",()=>"className is required to be a string, but got type "+typeof n.className),S(n.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof e>"u"&&(e=n.className);const s=e,o=t+">"+s;return on.register(n),mS.set(o,n),gS.set(n,o),n}class ps extends _o{minimize(t,e=!1,s){const{value:o,grads:r}=this.computeGradients(t,s);if(s!=null){const i=s.map(a=>({name:a.name,tensor:r[a.name]}));this.applyGradients(i)}else this.applyGradients(r);return It(r),e?o:(o.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,e){return C$(t,e)}dispose(){this.iterations_!=null&&It(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:At(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(ps,Symbol.hasInstance,{value:n=>n.minimize!=null&&n.computeGradients!=null&&n.applyGradients!=null});class km extends ps{static get className(){return"Adadelta"}constructor(t,e,s=null){super(),this.learningRate=t,this.rho=e,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=O.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=O.registeredVariables[s],i=!1;this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accum_grad`,variable:V(()=>kt(r).variable(i))}),this.accumulatedUpdates[o]==null&&(this.accumulatedUpdates[o]={originalName:`${s}/accum_var`,variable:V(()=>kt(r).variable(i))});const a=Array.isArray(t)?t[o].tensor:t[s];if(a==null)return;const l=this.accumulatedGrads[o].variable,c=this.accumulatedUpdates[o].variable;V(()=>{const u=J(F(l,this.rho),F(Vt(a),1-this.rho)),h=F(dt(ke(J(c,this.epsilon)),ke(J(l,this.epsilon))),a),d=J(F(c,this.rho),F(Vt(h),1-this.rho));l.assign(u),c.assign(d);const p=J(F(h,-this.learningRate),r);r.assign(p)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(It(this.accumulatedGrads.map(t=>t.variable)),It(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedUpdates=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,e){return new t(e.learningRate,e.rho,e.epsilon)}}class vm extends ps{static get className(){return"Adagrad"}constructor(t,e=.1){super(),this.learningRate=t,this.initialAccumulatorValue=e,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=O.registeredVariables[s];this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accumulator`,variable:V(()=>il(r.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(t)?t[o].tensor:t[s];if(i==null)return;const a=this.accumulatedGrads[o].variable;V(()=>{const l=J(a,Vt(i));a.assign(l);const c=J(F(dt(i,ke(J(l,O.backend.epsilon()))),-this.learningRate),r);r.assign(c)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&It(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(e)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,e){return new t(e.learningRate,e.initialAccumulatorValue)}}class Sm extends ps{static get className(){return"Adam"}constructor(t,e,s,o=null){super(),this.learningRate=t,this.beta1=e,this.beta2=s,this.epsilon=o,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],V(()=>{this.accBeta1=At(e).variable(),this.accBeta2=At(s).variable()}),o==null&&(this.epsilon=O.backend.epsilon())}applyGradients(t){const e=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);V(()=>{const s=pt(1,this.accBeta1),o=pt(1,this.accBeta2);e.forEach((r,i)=>{const a=O.registeredVariables[r],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${r}/m`,variable:V(()=>kt(a).variable(l))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${r}/v`,variable:V(()=>kt(a).variable(l))});const c=Array.isArray(t)?t[i].tensor:t[r];if(c==null)return;const u=this.accumulatedFirstMoment[i].variable,h=this.accumulatedSecondMoment[i].variable,d=J(F(u,this.beta1),F(c,1-this.beta1)),p=J(F(h,this.beta2),F(Vt(c),1-this.beta2)),f=dt(d,s),m=dt(p,o);u.assign(d),h.assign(p);const g=J(F(dt(f,J(ke(m),this.epsilon)),-this.learningRate),a);a.assign(g)}),this.accBeta1.assign(F(this.accBeta1,this.beta1)),this.accBeta2.assign(F(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&It(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&It(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t),V(()=>{this.accBeta1.assign(Us(this.beta1,this.iterations_+1)),this.accBeta2.assign(Us(this.beta2,this.iterations_+1))});const e=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,e){return new t(e.learningRate,e.beta1,e.beta2,e.epsilon)}}class Nm extends ps{static get className(){return"Adamax"}constructor(t,e,s,o=null,r=0){super(),this.learningRate=t,this.beta1=e,this.beta2=s,this.epsilon=o,this.decay=r,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],V(()=>{this.iteration=At(0).variable(),this.accBeta1=At(e).variable()}),o==null&&(this.epsilon=O.backend.epsilon())}applyGradients(t){const e=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);V(()=>{const s=pt(1,this.accBeta1),o=dt(-this.learningRate,J(F(this.iteration,this.decay),1));e.forEach((r,i)=>{const a=O.registeredVariables[r],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${r}/m`,variable:kt(a).variable(l)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${r}/v`,variable:kt(a).variable(l)});const c=Array.isArray(t)?t[i].tensor:t[r];if(c==null)return;const u=this.accumulatedFirstMoment[i].variable,h=this.accumulatedWeightedInfNorm[i].variable,d=J(F(u,this.beta1),F(c,1-this.beta1)),p=F(h,this.beta2),f=Ee(c),m=hs(p,f);u.assign(d),h.assign(m);const g=J(F(dt(o,s),dt(d,J(m,this.epsilon))),a);a.assign(g)}),this.iteration.assign(J(this.iteration,1)),this.accBeta1.assign(F(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&It(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&It(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,e){return new t(e.learningRate,e.beta1,e.beta2,e.epsilon,e.decay)}}class yh extends ps{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=Array.isArray(t)?t[o].tensor:t[s];if(r==null)return;const i=O.registeredVariables[s];V(()=>{const a=J(F(this.c,r),i);i.assign(a)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=Tn(At(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,e){return new t(e.learningRate)}}class Tm extends yh{static get className(){return"Momentum"}constructor(t,e,s=!1){super(t),this.learningRate=t,this.momentum=e,this.useNesterov=s,this.accumulations=[],this.m=At(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=O.registeredVariables[s];this.accumulations[o]==null&&(this.accumulations[o]={originalName:`${s}/momentum`,variable:V(()=>kt(r).variable(!1))});const i=this.accumulations[o].variable,a=Array.isArray(t)?t[o].tensor:t[s];a!=null&&V(()=>{let l;const c=J(F(this.m,i),a);this.useNesterov?l=J(F(this.c,J(a,F(c,this.m))),r):l=J(F(this.c,c),r),i.assign(c),r.assign(l)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&It(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(e)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,e){return new t(e.learningRate,e.momentum,e.useNesterov)}}class Em extends ps{static get className(){return"RMSProp"}constructor(t,e=.9,s=0,o=null,r=!1){if(super(),this.learningRate=t,this.decay=e,this.momentum=s,this.epsilon=o,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=r,o==null&&(this.epsilon=O.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=O.registeredVariables[s],i=!1;this.accumulatedMeanSquares[o]==null&&(this.accumulatedMeanSquares[o]={originalName:`${s}/rms`,variable:V(()=>kt(r).variable(i))}),this.accumulatedMoments[o]==null&&(this.accumulatedMoments[o]={originalName:`${s}/momentum`,variable:V(()=>kt(r).variable(i))}),this.accumulatedMeanGrads[o]==null&&this.centered&&(this.accumulatedMeanGrads[o]={originalName:`${s}/mg`,variable:V(()=>kt(r).variable(i))});const a=Array.isArray(t)?t[o].tensor:t[s];if(a==null)return;const l=this.accumulatedMeanSquares[o].variable,c=this.accumulatedMoments[o].variable;V(()=>{const u=J(F(l,this.decay),F(Vt(a),1-this.decay));if(this.centered){const h=this.accumulatedMeanGrads[o].variable,d=J(F(h,this.decay),F(a,1-this.decay)),p=dt(F(a,this.learningRate),ke(pt(u,J(Vt(d),this.epsilon)))),f=J(F(c,this.momentum),p);l.assign(u),h.assign(d),c.assign(f);const m=pt(r,f);r.assign(m)}else{const h=J(F(l,this.decay),F(Vt(a),1-this.decay)),d=J(F(c,this.momentum),dt(F(a,this.learningRate),ke(J(h,this.epsilon))));l.assign(h),c.assign(d);const p=pt(r,d);r.assign(p)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&It(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&It(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&It(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedMoments=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(e*2,e*3).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,e){return new t(e.learningRate,e.decay,e.momentum,e.epsilon,e.centered)}}const xS=[km,vm,Sm,Nm,Tm,Em,yh];function bS(){for(const n of xS)Z(n)}class yS{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}function Rm(n){return new yS(n)}function wh(n,t){const e=n.shape.length,s=t.shape.length;if(e<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${e}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[s-1]>e)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[s-1]} vs. ${e}`);if(q(n.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${n.shape}.`);const o=t.shape,r=o[o.length-1];let i=1;for(let h=0;h<o.length-1;++h)i*=o[h];const a=n.shape,l=o.slice();l.pop();let c=1;for(let h=r;h<e;++h)c*=a[h],l.push(a[h]);const u=[...lt(n.shape).map(h=>h/c),1].slice(0,r);return[l,i,c,u]}const Ch=-2,wS=-1;function Ih(n,t,e){const s=n.shape.length;S(s===t.length,()=>`Error in slice${s}D: Length of begin ${t} must match the rank of the array (${s}).`),S(s===e.length,()=>`Error in slice${s}D: Length of size ${e} must match the rank of the array (${s}).`);for(let o=0;o<s;++o)S(t[o]+e[o]<=n.shape[o],()=>`Error in slice${s}D: begin[${o}] + size[${o}] (${t[o]+e[o]}) would overflow input.shape[${o}] (${n.shape[o]})`)}function CS(n){const t=[];let e=0;for(;n>0;)n&1&&t.push(e),n/=2,e++;return t}function $h(n,t,e){const s=[];for(let o=0;o<n.length;o++)s[o]=Math.ceil((t[o]-n[o])/e[o]);return s}function Am(n,t,e,s){const o=[...n];for(let r=o.length;r<s.length;r++)o.push(1);for(let r=0;r<e;r++)r===0?o[t]=1:(o.splice(t,0,1),o.pop());return o}function Dm(n,t,e){return e<=n?e:e-(t-1)}function Fm(n,t){const e=[];for(let s=0;s<n;s++)e.push(t+s);return e}function IS(n,t,e,s,o,r,i,a,l){const c=n.length;let u=new Array(c),h=new Array(c),d=new Array(c);if(t.length&&e>0){const p=t[0],f=e+1;u=Om(i,p,f,s,n),h=_m(a,p,f,o,n),d=Am(r,p,f,n)}else for(let p=0;p<c;p++)u[p]=Mm(i,s,r,n,p,l),h[p]=Pm(a,o,r,n,p,l),d[p]=Lm(r,p,l);return{begin:u,end:h,strides:d}}function Om(n,t,e,s,o){const r=[...o],i=Fm(e,t);for(let a=0;a<r.length;a++)if(i.indexOf(a)>-1)r[a]=0;else{const l=Dm(t,e,a);let c=s[l];n&1<<l&&(c=0),r[a]=c}return r}function _m(n,t,e,s,o){const r=[...o],i=Fm(e,t);for(let a=0;a<r.length;a++)if(i.indexOf(a)>-1)r[a]=Number.MAX_SAFE_INTEGER;else{const l=Dm(t,e,a);let c=s[l];n&1<<l&&(c=Number.MAX_SAFE_INTEGER),r[a]=c}for(let a=0;a<r.length;a++){const l=o[a];r[a]<0&&(r[a]+=l),r[a]=As(0,r[a],o[a])}return r}function Lm(n,t,e){let s=n[t];return(e&1<<t||s==null)&&(s=1),s}function Mm(n,t,e,s,o,r){let i=t[o];const a=e[o]||1;(n&1<<o||r&1<<o||i==null)&&(a>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const l=s[o];return i<0&&(i+=l),i=As(0,i,l-1),i}function Pm(n,t,e,s,o,r){let i=t[o];const a=e[o]||1;(n&1<<o||r&1<<o||i==null)&&(a>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const l=s[o];return i<0&&(i+=l),a>0?i=As(0,i,l):i=As(-1,i,l-1),i}function kh(n,t,e){let s=e.length;for(let o=0;o<e.length;o++)if(e[o]>1){s=o;break}for(let o=s+1;o<e.length;o++)if(t[o]>0||e[o]!==n[o])return!1;return!0}function vh(n,t){let e=n.length>0?n[n.length-1]:1;for(let s=0;s<n.length-1;s++)e+=n[s]*t[s];return e}function $l(n,t,e){let s;const o=n.shape.length;typeof t=="number"?s=[t,...new Array(o-1).fill(0)]:t.length<o?s=t.concat(new Array(o-t.length).fill(0)):s=t.slice(),s.forEach(i=>{S(i!==-1,()=>"slice() does not support negative begin indexing.")});let r;return e==null?r=new Array(o).fill(-1):typeof e=="number"?r=[e,...new Array(o-1).fill(-1)]:e.length<o?r=e.concat(new Array(o-e.length).fill(-1)):r=e,r=r.map((i,a)=>i>=0?i:(S(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),n.shape[a]-s[a])),[s,r]}function Sh(n,t,e,s,o,r,i,a,l){let c;if(s==null?(c=new Array(t.length),c.fill(1)):c=s,i!=null&&(i&i-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let u=!1;const h={dims:c.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:e.slice(),strides:c.slice(),beginMask:o,endMask:r,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:l};for(let w=0;w<h.dims;w++)u&&(1<<w&a)!==0&&h.numAddAxisAfterEllipsis++,1<<w&i&&(u=!0);u||(h.ellipsisMask|=1<<h.dims,h.dims++);const d={dims:n.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};$S(h,d);let p=!0,f=!0,m=!0;const g=[],x=[];for(let w=0;w<n.length;++w){if(d.strides[w]===0)throw Error(`strides[${w}] must be non-zero`);const y=!!(d.shrinkAxisMask&1<<w),C=n[w];if(C===-1){g.push(y?1:-1);continue}const $=[d.beginMask&1<<w,d.endMask&1<<w],N=[d.strides[w]>0?0:-1,d.strides[w]>0?C:C-1];if(y&&d.strides[w]<=0)throw Error("only stride 1 allowed on non-range indexing.");m=m&&d.strides[w]===1;const T=!!(d.beginMask&1<<w&&d.endMask&1<<w);if(d.beginValid&&d.endValid){if(y){const R=d.begin[w]<0?C+d.begin[w]:d.begin[w];if(d.begin[w]=R,d.end[w]=d.begin[w]+1,R<0||R>=C)throw Error(`slice index ${d.begin[w]} of dimension ${w} out of bounds.`)}else d.begin[w]=Bm(d.begin[w],0,d.strides[w],C,$,N),d.end[w]=Bm(d.end[w],1,d.strides[w],C,$,N);const I=d.strides[w]===1&&d.begin[w]===0&&d.end[w]===C;p=p&&I,f=f&&(w===0&&d.strides[w]===1||I)}else p=p&&d.strides[w]===1&&T,f=f&&(w===0&&d.strides[w]===1||T);let k,v=!1;if(d.beginValid&&d.endValid?(k=d.end[w]-d.begin[w],v=!0):y?(k=1,v=!0):T&&C>=0&&(d.strides[w]<0?k=-C:k=C,v=!0),v){let I;k===0||k<0!=d.strides[w]<0?I=0:I=Math.trunc(k/d.strides[w])+(k%d.strides[w]!==0?1:0),g.push(I)}else g.push(-1)}for(let w=0;w<d.finalShapeGatherIndices.length;++w){const y=d.finalShapeGatherIndices[w];y>=0?x.push(g[y]):y===Ch&&x.push(1)}return{finalShapeSparse:x.filter((w,y)=>d.finalShapeGatherIndices[y]!==Ch),finalShape:x,isIdentity:p,sliceDim0:f,isSimpleSlice:m,begin:d.begin,end:d.end,strides:d.strides}}function $S(n,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let e=0;t.beginValid=n.begin!=null,t.endValid=n.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let s=0;s<n.dims;s++)if(1<<s&n.ellipsisMask){const o=Math.min(t.dims-(n.dims-s)+1+n.numAddAxisAfterEllipsis,t.dims);for(;e<o;e++)t.begin[e]=0,t.end[e]=0,t.strides[e]=1,t.beginMask|=1<<e,t.endMask|=1<<e,t.finalShapeGatherIndices.push(e),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[e]=s}else if(1<<s&n.newAxisMask)t.finalShapeGatherIndices.push(Ch),t.finalShapeGatherIndicesSparse.push(-1);else{if(e===t.begin.length)throw Error(`Index out of range using input dim ${e}; input has only ${t.dims} dims, ${t.begin.length}.`);n.begin!=null&&(t.begin[e]=n.begin[s]),n.end!=null&&(t.end[e]=n.end[s]),t.strides[e]=n.strides[s],n.beginMask&1<<s&&(t.beginMask|=1<<e),n.endMask&1<<s&&(t.endMask|=1<<e),n.shrinkAxisMask&1<<s?(t.finalShapeGatherIndices.push(wS),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<e):(t.finalShapeGatherIndices.push(e),t.finalShapeGatherIndicesSparse.push(s)),t.inputShapeGatherIndicesSparse[e]=s,e++}}function Bm(n,t,e,s,o,r){if(o[t])return e>0?r[t]:r[t+1&1];{const i=n<0?s+n:n;return i<r[0]?r[0]:i>r[1]?r[1]:i}}var kS=Object.freeze({__proto__:null,assertParamsValid:Ih,computeFlatOffset:vh,computeOutShape:$h,getNormalizedAxes:IS,isSliceContinous:kh,maskToAxes:CS,parseSliceParams:$l,sliceInfo:Sh,startForAxis:Mm,startIndicesWithElidedDims:Om,stopForAxis:Pm,stopIndicesWithElidedDims:_m,stridesForAxis:Lm,stridesWithElidedDims:Am});class vS{static sgd(t){return new yh(t)}static momentum(t,e,s=!1){return new Tm(t,e,s)}static rmsprop(t,e=.9,s=0,o=null,r=!1){return new Em(t,e,s,o,r)}static adam(t=.001,e=.9,s=.999,o=null){return new Sm(t,e,s,o)}static adadelta(t=.001,e=.95,s=null){return new km(t,e,s)}static adamax(t=.002,e=.9,s=.999,o=null,r=0){return new Nm(t,e,s,o,r)}static adagrad(t,e=.1){return new vm(t,e)}}const Xs=vS;const SS=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:n=>n();function zm(){return new Promise(n=>SS(()=>n()))}function Nh(n,t){const e=n[0].length;n.forEach((o,r)=>{S(o.length===e,()=>`Error in concat${e}D: rank of tensors[${r}] must be the same as the rank of the rest (${e})`)}),S(t>=0&&t<e,()=>`Error in concat${e}D: axis must be between 0 and ${e-1}.`);const s=n[0];n.forEach((o,r)=>{for(let i=0;i<e;i++)S(i===t||o[i]===s[i],()=>`Error in concat${e}D: Shape of tensors[${r}] (${o}) does not match the shape of the rest (${s}) along the non-concatenated axis ${r}.`)})}function Fn(n,t){const e=n[0].slice();for(let s=1;s<n.length;s++)e[t]+=n[s][t];return e}var xn;(function(n){n[n.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",n[n.VALUE_ROWIDS=1]="VALUE_ROWIDS",n[n.ROW_LENGTHS=2]="ROW_LENGTHS",n[n.ROW_SPLITS=3]="ROW_SPLITS",n[n.ROW_LIMITS=4]="ROW_LIMITS",n[n.ROW_STARTS=5]="ROW_STARTS"})(xn||(xn={}));function Vm(n,t,e){let s=new Array;if(e==null&&t==null)return s;if(t==null)for(;s.length<n+e.length;)s.push(-1);else s=t.slice();if(e==null)return s;if(n+e.length!==s.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${n+e.length}, but shape.rank = ${s.length}`);for(let o=1;o<e.length;++o){const r=e[o],i=s[s.length-e.length+o],a=s[i];if(r>=0)if(a>=0){if(a!==r)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${o+n}] = ${r} but shape[${o+n}] = ${a}`)}else s[i]=r}return s}function Wm(n){const t={FIRST_DIM_SIZE:xn.FIRST_DIM_SIZE,VALUE_ROWIDS:xn.VALUE_ROWIDS,ROW_LENGTHS:xn.ROW_LENGTHS,ROW_SPLITS:xn.ROW_SPLITS,ROW_LIMITS:xn.ROW_LIMITS,ROW_STARTS:xn.ROW_STARTS},e=[];for(const s of n)if(s in t)e.push(t[s]);else break;return e}function Um(n){return n.length===0?0:n[0]===xn.FIRST_DIM_SIZE?n.length-1:n.length}function Gm(n,t){if(n==null||t==null)return;const e=n.length,s=t.length;if(e>=s)throw new Error(`defaultValue.shape=${n} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${e} must be less than ragged tensor input flatValues.rank = ${s})`);for(let o=0;o<Math.min(e,s-1);++o){const r=n[o],i=t[o+1];if(r>=0&&i>=0&&r!==1&&r!==i)throw new Error(`defaultValue.shape=${n}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${o-n.length}] = ${r} but ragged tensor input.flatValues.shape[${o-n.length}] = ${i}`)}}const Th=30;function kl(n){return n<=Th?n:Rc(n,Math.floor(Math.sqrt(n)))}function Eh(n,t,e){const s=e*(typeof n=="number"?n:n[0]),o=t*(typeof n=="number"?n:n[1]);return[s,o]}function fi(n,t,e,s=!0){let o=[];if(s)o=o.concat(t.slice(0)),o.push(n[0]/e),o=o.concat(n.slice(1));else{o=o.concat(n[0]);const r=t.length;for(let i=0;i<r;++i)o=o.concat([n[i+1]/t[i],t[i]]);o=o.concat(n.slice(r+1))}return o}function mi(n,t,e=!0){const s=[];if(e){s.push(t);for(let o=t+1;o<n;++o)o<=2*t?(s.push(o),s.push(o-(t+1))):s.push(o)}else{const o=[],r=[];for(let i=1;i<n;++i)i>=t*2+1||i%2===1?r.push(i):o.push(i);s.push(...o),s.push(0),s.push(...r)}return s}function gi(n,t,e,s=!0){const o=[];s?o.push(n[0]/e):o.push(n[0]*e);for(let r=1;r<n.length;++r)r<=t.length?s?o.push(t[r-1]*n[r]):o.push(n[r]/t[r-1]):o.push(n[r]);return o}function Rh(n,t){const e=[0];for(let s=0;s<t;++s)e.push(n[s][0]);return e}function Ah(n,t,e){const s=n.slice(0,1);for(let o=0;o<e;++o)s.push(n[o+1]-t[o][0]-t[o][1]);return s}const vl=1.7580993408473768,Sl=1.0507009873554805;const Dh=.3275911,Fh=.254829592,Oh=-.284496736,_h=1.421413741,Lh=-1.453152027,Mh=1.061405429;function Yn(n,t){if(n.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${n.length}, imag: ${t.length}.`);const e=new Float32Array(n.length*2);for(let s=0;s<e.length;s+=2)e[s]=n[s/2],e[s+1]=t[s/2];return e}function Hm(n){const t=new Float32Array(n.length/2),e=new Float32Array(n.length/2);for(let s=0;s<n.length;s+=2)t[s/2]=n[s],e[s/2]=n[s+1];return{real:t,imag:e}}function qm(n){const t=Math.ceil(n.length/4),e=new Float32Array(t),s=new Float32Array(t);for(let o=0;o<n.length;o+=4)e[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:e,imag:s}}function Km(n){const t=Math.floor(n.length/4),e=new Float32Array(t),s=new Float32Array(t);for(let o=2;o<n.length;o+=4)e[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:e,imag:s}}function Ph(n,t){const e=n[t*2],s=n[t*2+1];return{real:e,imag:s}}function jm(n,t,e,s){n[s*2]=t,n[s*2+1]=e}function Xm(n,t){const e=new Float32Array(n/2),s=new Float32Array(n/2);for(let o=0;o<Math.ceil(n/2);o++){const r=(t?2:-2)*Math.PI*(o/n);e[o]=Math.cos(r),s[o]=Math.sin(r)}return{real:e,imag:s}}function Ym(n,t,e){const s=(e?2:-2)*Math.PI*(n/t),o=Math.cos(s),r=Math.sin(s);return{real:o,imag:r}}const Bh="->",NS=/->/g,Zm=",",Jm="...";function zh(n,t){n=n.replace(/\s/g,"");const e=(n.length-n.replace(NS,"").length)/Bh.length;if(e<1)throw new Error("Equations without an arrow are not supported.");if(e>1)throw new Error(`Equation must contain exactly one arrow ("${Bh}").`);const[s,o]=n.split(Bh);S(s.indexOf(Jm)===-1,()=>`The ellipsis notation ("${Jm}") is not supported yet.`);const r=s.split(Zm),i=r.length;if(t!==i)throw new Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const a=[];for(let d=0;d<o.length;++d){const p=o[d];if(!r.some(f=>f.indexOf(p)!==-1))throw new Error(`Output subscripts contain the label ${p} not present in the input subscripts.`);a.indexOf(p)===-1&&a.push(p)}for(let d=0;d<s.length;++d){const p=s[d];a.indexOf(p)===-1&&p!==Zm&&a.push(p)}const l=new Array(r.length);for(let d=0;d<i;++d){if(new Set(r[d].split("")).size!==r[d].length)throw new Error(`Found duplicate axes in input component ${r[d]}. Support for duplicate axes in input is not implemented yet.`);l[d]=[];for(let p=0;p<r[d].length;++p)l[d].push(a.indexOf(r[d][p]))}const c=a.length,u=o.length,h=[];for(let d=u;d<c;++d)h.push(d);return{allDims:a,summedDims:h,idDims:l}}function Vh(n,t){let e=new Array(n);e.fill(-1);for(let o=0;o<t.length;++o)e[t[o]]=o;const s=[];for(let o=0;o<n;++o)e[o]===-1&&s.push(o);return e=e.filter(o=>o!==-1),{permutationIndices:e,expandDims:s}}function Wh(n,t,e){const s=new Array(n);for(let o=0;o<e.length;++o){const r=e[o].shape;for(let i=0;i<t[o].length;++i)s[t[o][i]]===void 0?s[t[o][i]]=r[i]:S(s[t[o][i]]===r[i],()=>`Expected dimension ${s[t[o][i]]} at axis ${i} of input shaped ${JSON.stringify(r)}, but got dimension ${r[i]}`)}}function Uh(n,t){const e=n,s=[];let o=0;n.length===0&&e.push(-1),o=n.length+1;for(let i=0;i<o;++i)s.push([]);const r=[];for(let i=0;i<e.length;++i){const a=e[i],l=TS(t,a);for(const c of l)r.indexOf(c)===-1&&(s[i].push(c),r.push(c))}return{path:e,steps:s}}function Gh(n){return n.every((t,e)=>t===e)}function TS(n,t){const e=[];for(let s=0;s<n.length;++s)(n[s].length===0||n[s].indexOf(t)!==-1||t===-1)&&e.push(s);return e}function Hh(n,t,e=0){let s=[];if(typeof t=="number")S(n.shape[e]%t===0,()=>"Number of splits must evenly divide the axis."),s=new Array(t).fill(n.shape[e]/t);else{const o=t.reduce((i,a)=>(a===-1&&(i+=1),i),0);S(o<=1,()=>"There should be only one negative value in split array.");const r=t.indexOf(-1);if(r!==-1){const i=t.reduce((a,l)=>l>0?a+l:a);t[r]=n.shape[e]-i}S(n.shape[e]===t.reduce((i,a)=>i+a),()=>"The sum of sizes must match the size of the axis dimension."),s=t}return s}function Qm(n){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${n}`}function tg(n,t){return`indices(${n}, 0) is invalid: ${t} < 0`}function eg(n,t,e){return`indices(${n}, 0) is invalid: ${t} >= ${e}`}function ng(n,t){return`only one output dimension may be -1, not both ${n} and ${t}`}function sg(n,t){return`size ${n} must be non-negative, not ${t}`}function og(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function rg(n,t){const e=q(n),s=q(t);return`Input to reshape is a SparseTensor with ${e}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${n} outputShape= ${t}`}function ig(n,t){const e=q(n),s=q(t);return`Input to reshape is a tensor with ${e} dense values, but the requested shape has ${s}. inputShape=${n} outputShape=${t}`}function qh(){return"segment ids must be >= 0"}function ag(){return"segment ids are not increasing"}function lg(n,t){return`Segment id ${n} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function cg(n,t,e){return`Bad: indices[${n}] == ${t} out of range [0, ${e})`}function ug(n,t){let e=!1,s;for(n<=Th?(s=n,e=!0):s=Rc(n,Math.floor(Math.sqrt(n)));!e;)s>t||s===n?e=!0:s=Rc(n,s+1);return s}function hg(n,t,e){const s=[],o=n.length;for(let r=0;r<o;r++)r!==t?s.push(n[r]):s.push(e);return s}function Kh(n,t,e,s){const o=t.shape.length,r=n.shape.length;if(s!==0&&(s<-o||s>o))throw new Error(`Expect batchDims in the range of [-${o}, ${o}], but got ${s}`);if(s<0&&(s+=o),s>r)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${r}).`);if(e<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${e}).`);for(let h=0;h<s;++h)if(n.shape[h]!==t.shape[h])throw new Error(`x.shape[${h}]: ${n.shape[h]} should be equal to indices.shape[${h}]: ${t.shape[h]}.`);const i=n.shape[e],a=[];let l=1,c=1,u=1;for(let h=0;h<s;++h)a.push(n.shape[h]),l*=n.shape[h];for(let h=s;h<e;h++)a.push(n.shape[h]),c*=n.shape[h];for(let h=s;h<o;h++)a.push(t.shape[h]);for(let h=e+1;h<r;h++)a.push(n.shape[h]),u*=n.shape[h];return{batchSize:l,sliceSize:u,outerSize:c,dimSize:i,outputShape:a}}var ES=Object.freeze({__proto__:null,collectGatherOpShapeInfo:Kh,computeOutShape:hg,segOpComputeOptimalWindowSize:ug});function Zn(n){try{return n.map(t=>as(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function dg(n){return n.map(t=>is(t))}var RS=Object.freeze({__proto__:null,ERF_A1:Fh,ERF_A2:Oh,ERF_A3:_h,ERF_A4:Lh,ERF_A5:Mh,ERF_P:Dh,PARALLELIZE_THRESHOLD:Th,get RowPartitionType(){return xn},SELU_SCALE:Sl,SELU_SCALEALPHA:vl,applyActivation:ph,assertAndGetBroadcastShape:mt,assertAxesAreInnerMostDims:xe,assertParamsConsistent:Nh,assignToTypedArray:jm,axesAreInnerMostDims:ju,calculateShapes:Ks,checkEinsumDimSizes:Wh,checkPadOnDimRoundingMode:Le,combineLocations:Uf,combineRaggedTensorToTensorShapes:Vm,complexWithEvenIndex:qm,complexWithOddIndex:Km,computeConv2DInfo:ge,computeConv3DInfo:cs,computeDefaultPad:zu,computeDilation2DInfo:si,computeOptimalWindowSize:kl,computeOutAndReduceShapes:de,computeOutShape:Fn,computePool2DInfo:nn,computePool3DInfo:Hn,convertConv2DDataFormat:qn,decodeEinsumEquation:zh,eitherStridesOrDilationsAreOne:$e,expandShapeToKeepDim:ne,exponent:Ym,exponents:Xm,fromStringArrayToUint8:dg,fromUint8ToStringArray:Zn,getAxesPermutation:qt,getBroadcastDims:Ro,getComplexWithIndex:Ph,getEinsumComputePath:Uh,getEinsumPermutation:Vh,getFusedBiasGradient:dh,getFusedDyActivation:hh,getImageCenter:Eh,getInnerMostAxes:Jt,getPermuted:mi,getRaggedRank:Um,getReductionAxes:re,getReshaped:fi,getReshapedPermuted:gi,getRowPartitionTypesHelper:Wm,getSliceBeginCoords:Rh,getSliceSize:Ah,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Qm,getSparseFillEmptyRowsNegativeIndexErrorMessage:tg,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:eg,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:og,getSparseReshapeInputOutputMismatchErrorMessage:ig,getSparseReshapeInputOutputMultipleErrorMessage:rg,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:ng,getSparseReshapeNegativeOutputDimErrorMessage:sg,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:cg,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:qh,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:ag,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:lg,getUndoAxesPermutation:us,isIdentityPermutation:Gh,log:lw,mergeRealAndImagArrays:Yn,prepareAndValidate:wh,prepareSplitSize:Hh,segment_util:ES,shouldFuse:fh,slice_util:kS,splitRealAndImagArrays:Hm,stridesOrDilationsArePositive:Vs,tupleValuesAreOne:zs,upcastType:Ue,validateDefaultValueShape:Gm,validateInput:iv,validateUpdateShape:mm,warn:je});bS();const pg={kernelName:Zi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,pi(nt(e,"float32"),-1))}}};const AS={kernelName:nr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Vt(nt(e,"float32")),o=ke(pt(At(1),s));return Qt(dt(n,o))}}}};const DS={kernelName:sr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=ke(pt(Vt(nt(e,"float32")),1));return dt(n,s)}}}};const FS={kernelName:Co,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{let a=n;const l=re(e.shape,o);return l.length>0&&(a=ut(a,l)),L(a,e.shape)},b:()=>{let a=n;const l=re(s.shape,o);return l.length>0&&(a=ut(a,l)),L(a,s.shape)}}}};const OS={kernelName:_c,saveAllInputs:!0,gradFunc:(n,t)=>{const e={};return t.forEach((s,o)=>{e[o]=()=>n.clone()}),e}};const _S={kernelName:Ji,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>kt(e)}}};const LS={kernelName:Qi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>kt(e)}}};const MS={kernelName:or,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,ke(pt(At(1),Vt(nt(e,"float32")))))}}};const PS={kernelName:rr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=ke(J(At(1),Vt(nt(e,"float32"))));return dt(n,s)}}}};const BS={kernelName:lr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{const a=J(Vt(e),Vt(s));let l=F(n,dt(s,a));const c=re(e.shape,o);return c.length>0&&(l=ut(l,c)),L(l,e.shape)},b:()=>{const a=J(Vt(e),Vt(s));let l=Qt(F(n,dt(e,a)));const c=re(s.shape,o);return c.length>0&&(l=ut(l,c)),L(l,s.shape)}}}};const zS={kernelName:ir,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,J(Vt(nt(e,"float32")),1))}}};const VS={kernelName:ar,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,pt(At(1),Vt(nt(e,"float32"))))}}};function WS(n,t,e,s,o,r){const i=E(n,"dy","avgPool3dGrad"),a=E(t,"input","avgPool3dGrad");let l=i,c=a,u=!1;a.rank===4&&(u=!0,l=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]]),c=L(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),S(l.rank===5,()=>`Error in avgPool3dGrad: dy must be rank 5 but got rank ${l.rank}.`),S(c.rank===5,()=>`Error in avgPool3dGrad: input must be rank 5 but got rank ${c.rank}.`),Le("avgPool3dGrad",o,r);const h={dy:l,input:c},d={filterSize:e,strides:s,pad:o,dimRoundingMode:r},p=O.runKernel(Bc,h,d);return u?L(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const US=M({avgPool3dGrad_:WS});const GS={kernelName:ea,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{filterSize:o,strides:r,pad:i,dimRoundingMode:a}=e;return{x:()=>US(n,s,o,r,i,a)}}};function HS(n,t,e,s,o){const r=E(n,"dy","avgPoolGrad"),i=E(t,"input","avgPoolGrad");S(i.rank===r.rank,()=>`Rank of input (${i.rank}) does not match rank of dy (${r.rank})`);let a=i,l=r,c=!1;i.rank===3&&(c=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2]]),l=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),S(l.rank===4,()=>`Error in avgPoolGrad: dy must be rank 4 but got rank ${l.rank}.`),S(a.rank===4,()=>`Error in avgPoolGrad: input must be rank 4 but got rank ${a.rank}.`);const u={dy:l,input:a},h={filterSize:e,strides:s,pad:o},d=O.runKernel(Pc,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const qS=M({avgPoolGrad_:HS});const KS={kernelName:ta,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{filterSize:o,strides:r,pad:i}=e;return{x:()=>qS(n,s,o,r,i)}}};const jS={kernelName:na,inputsToSave:["a","b"],gradFunc:(n,t,e)=>{const[s,o]=t,{transposeA:r,transposeB:i}=e;return!r&&!i?{a:()=>Rt(n,o,!1,!0),b:()=>Rt(s,n,!0,!1)}:!r&&i?{a:()=>Rt(n,o,!1,!1),b:()=>Rt(n,s,!0,!1)}:r&&!i?{a:()=>Rt(o,n,!1,!0),b:()=>Rt(s,n,!1,!1)}:{a:()=>Rt(o,n,!0,!0),b:()=>Rt(n,s,!0,!0)}}};const XS={kernelName:sa,gradFunc:(n,t,e)=>{const{blockShape:s,crops:o}=e;return{x:()=>nh(n,s,o)}}};const YS={kernelName:ow,gradFunc:(n,t,e)=>{const s=e,o=s.inputShape,r=s.shape,i=Array.from(r);for(let l=o.length-1;l>=0;l--)if(o[l]===r[l])i[l]=1;else if(o[l]!==1)throw new Error(`broadcastTo(): [${o}] cannot be broadcast to [${r}].`);const a=[];for(let l=0;l<i.length;l++)i[l]>1&&a.push(l);return{x:()=>ut(n,a,!0)}}};const ZS={kernelName:cr,gradFunc:n=>({x:()=>n.clone()})};const JS={kernelName:ur,gradFunc:n=>({x:()=>kt(n)})};const QS={kernelName:hr,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{clipValueMin:o,clipValueMax:r}=e;return{x:()=>Re(Kn(Gs(s,o),Ao(s,r)),n,kt(n))}}};const t2={kernelName:oa,inputsToSave:["x"],gradFunc:pg.gradFunc};const e2={kernelName:ra,saveAllInputs:!0,gradFunc:(n,t,e)=>{const s=t.map(l=>l.shape),{axis:o}=e,r=yt(o,t[0].shape)[0],i=s.map(l=>l[r]);return Ze(n,i,r).map(l=>()=>l)}};const n2={kernelName:ia,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,{dilations:r,strides:i,pad:a,dataFormat:l}=e;return S(zs(r),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`),{x:()=>Gu(s.shape,n,o,i,a,l),filter:()=>uh(s,n,o.shape,i,a,l)}}};const s2={kernelName:aa,inputsToSave:["dy","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,{strides:r,pad:i,dataFormat:a,dimRoundingMode:l}=e;return{dy:()=>Ws(n,o,r,i,a,1,l),filter:()=>uh(n,s,o.shape,r,i,a,l)}}};function o2(n,t,e,s,o){let r=n;n.rank===4&&(r=L(n,[1,n.shape[0],n.shape[1],n.shape[2],n.shape[3]]));let i=t;i.rank===4&&(i=L(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]])),S(r.rank===5,()=>`Error in conv3dDerFilter: input must be rank 5, but got shape ${r.shape}.`),S(i.rank===5,()=>`Error in conv3dDerFilter: dy must be rank 5, but got shape ${i.shape}.`),S(e.length===5,()=>`Error in conv3dDerFilter: filterShape must be length 5, but got ${e}.`),S(r.shape[4]===e[3],()=>`Error in conv3dDerFilter: depth of input ${r.shape[4]}) must match input depth in filter (${e[3]}.`),S(i.shape[4]===e[4],()=>`Error in conv3dDerFilter: depth of dy (${i.shape[4]}) must match output depth for filter (${e[4]}).`);const a={x:r,dy:i},l={strides:s,pad:o,filterShape:e};return O.runKernel(Gc,a,l)}const r2=M({conv3DBackpropFilter_:o2});const i2={kernelName:la,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const{dilations:s,strides:o,pad:r}=e;S(zs(s),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${s}'`);const[i,a]=t;return{x:()=>Pf(i.shape,n,a,o,r),filter:()=>r2(i,n,a.shape,o,r)}}};const a2={kernelName:dr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(Qt(hm(nt(e,"float32"))),n)}}};const l2={kernelName:pr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(dm(nt(e,"float32")),n)}}};const c2={kernelName:ca,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o,exclusive:r,reverse:i}=e;return{x:()=>{const a=qt([o],s.rank);let l=zf(n,o,r,!i);return a!=null&&(l=vt(l,a)),l}}}};const u2={kernelName:ua,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const{dilations:s,strides:o,pad:r,dimRoundingMode:i}=e,a=s??[1,1];S(zs(a),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[l,c]=t;return S(l.rank===4,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${l.rank}.`),S(c.rank===4,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${c.rank}.`),S(l.shape[3]===c.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${l.shape[3]}) must match the inChannels dimension in filter ${c.shape[2]}.`),S($e(o,a),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${o} and dilations '${a}'.`),Le("depthwiseConv2d",r,i),{x:()=>vv(l.shape,n,c,o,r,a,i),filter:()=>$v(l,n,c.shape,o,r,a,i)}}};const h2={kernelName:ha,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,r={x:s,filter:o,dy:n},i={x:s,filter:o,dy:n};return{x:()=>O.runKernel(Jc,r,e),filter:()=>O.runKernel(Qc,i,e)}}};const d2={kernelName:mr,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t,s={dy:n,y:e};return{x:()=>O.runKernel(eu,s)}}};const p2={kernelName:gr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t,s=F(An(Qt(Vt(e))),2/Math.sqrt(Math.PI));return{x:()=>F(n,s)}}};const f2={kernelName:xr,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,e)}}};const m2={kernelName:pa,inputsToSave:["input"],gradFunc:(n,t)=>{const[e]=t;return{input:()=>L(n,e.shape)}}};const g2={kernelName:br,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,An(e))}}};const x2={kernelName:yr,gradFunc:n=>({x:()=>kt(n)})};const b2={kernelName:wr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{const a=dt(n,nt(s,"float32")),l=re(e.shape,o);return l.length>0?L(ut(a,l),e.shape):a},b:()=>{let a=F(n,nt(e,"float32"));const l=re(s.shape,o);l.length>0&&(a=L(ut(a,l),s.shape));const c=Vt(s);return Qt(dt(a,nt(c,"float32")))}}}};const y2={kernelName:fa,inputsToSave:["x","mean","variance","scale"],gradFunc:(n,t,e)=>{const{varianceEpsilon:s}=e,[o,r,i,a]=t,l=a??At(1),c=re(r.shape,o.shape),u=[];if(r.rank===1){for(let y=0;y<o.shape.length-1;++y)u.push(o.shape[y]);u.push(1)}const h=pt(o,r),d=F(n,l),p=lm(J(i,At(s))),f=F(F(F(p,p),p),At(-.5));return{x:()=>r.rank===1?L(F(F(n,gn(L(p,[1,1,1,r.shape[0]]),u)),l),o.shape):L(F(F(n,p),l),o.shape),mean:()=>{let y=F(F(p,At(-1)),d);return r.rank===1&&(y=ut(y,c)),L(y,r.shape)},variance:()=>{let y=F(F(f,h),d);return r.rank===1&&(y=ut(y,c)),L(y,r.shape)},scale:()=>{const y=F(h,p);let C=F(n,y);return r.rank===1&&(C=ut(C,c)),L(C,r.shape)},offset:()=>{let y=n;return r.rank===1&&(y=ut(y,c)),L(y,r.shape)}}}};const w2={kernelName:ma,inputsToSave:["x","indices"],gradFunc:(n,t,e)=>{const[s,o]=t,{axis:r,batchDims:i}=e,a=yt(r,s.shape)[0],l=(c,u,h)=>()=>{const d=c.shape,p=u.size,f=d.slice(0,a),m=f.length,g=d.slice(r,d.length).slice(1),x=g.length,b=fg(0,m),w=fg(m+1,m+1+x),y=mg([f,[p],g]),C=L(h,y),$=L(u,[p]),N=mg([[m],b,w]),T=vt(C,N);let k=xm(T,$,c.shape[a]);const v=us(N);return k=vt(k,v),k};if(i===1){const c=s.shape[0],u=s.split(c,0);return{x:()=>jn(u.map((p,f)=>l(p,o.slice(f,1),n.slice(f,1))())).reshape(s.shape),indices:()=>o}}else return{x:l(s,o,n),indices:()=>o}}};function fg(n,t){const e=[];for(let s=n;s<t;++s)e.push(s);return e}function mg(n){const t=[];for(let e=0;e<n.length;++e)for(let s=0;s<n[e].length;++s)t.push(n[e][s]);return t}const C2={kernelName:Cr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>kt(e),b:()=>kt(s)}}};const I2={kernelName:Ir,gradFunc:n=>({x:()=>nt(n,"float32")})};const $2={kernelName:$r,gradFunc:n=>({x:()=>kt(n)})};const k2={kernelName:kr,gradFunc:n=>({x:()=>kt(n)})};const v2={kernelName:vr,gradFunc:n=>({x:()=>kt(n)})};const S2={kernelName:xa,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{alpha:o}=e,r=Ye(s,0);return{x:()=>Re(r,n,F(n,o))}}};const N2={kernelName:Nr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,J(e,1))}}};const T2={kernelName:Sr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,nt(e,"float32"))}}};const E2={kernelName:iw,inputsToSave:[],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e;return{logits:()=>{const i=An(s);return pt(n,F(ut(n,o,!0),i))}}}};function R2(n,t,e,s=5,o=1,r=1,i=.5){const a={x:n,y:t,dy:e},l={depthRadius:s,bias:o,alpha:r,beta:i};return O.runKernel(au,a,l)}const A2=M({localResponseNormalizationBackprop_:R2});const D2={kernelName:$a,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{depthRadius:r,bias:i,alpha:a,beta:l}=e;return{x:()=>A2(s,o,n,r,i,a,l)}}};function gg(n,t,e,s){return t.rank<e.rank&&(t=L(t,ne(t.shape,s))),n.rank<e.rank&&(n=L(n,ne(n.shape,s))),{x:()=>F(n,nt(Rn(e,t),n.dtype))}}const xg={kernelName:ka,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const s=e,{reductionIndices:o}=s,r=t[0],i=t[1],a=yt(o,r.shape),l=gg(n,i,r,a);return{x:()=>l.x()}}};const F2={kernelName:Tr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>F(n,nt(Gs(e,s),"float32")),b:()=>F(n,nt(hl(e,s),"float32"))}}};function O2(n,t,e,s,o,r,i){const a=E(n,"dy","maxPool3dGrad"),l=E(t,"input","maxPool3dGrad"),c=E(e,"output","maxPool3dGrad");let u=a,h=l,d=c,p=!1;l.rank===4&&(p=!0,u=L(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]]),h=L(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]]),d=L(c,[1,c.shape[0],c.shape[1],c.shape[2],c.shape[3]])),S(u.rank===5,()=>`Error in maxPool3dGrad: dy must be rank 5 but got rank ${u.rank}.`),S(h.rank===5,()=>`Error in maxPool3dGrad: input must be rank 5 but got rank ${h.rank}.`),S(d.rank===5,()=>`Error in maxPool3dGrad: output must be rank 5 but got rank ${d.rank}.`),Le("maxPool3dGrad",r,i);const f={dy:u,input:h,output:d},m={filterSize:s,strides:o,pad:r,dimRoundingMode:i},g=O.runKernel(cu,f,m);return p?L(g,[g.shape[1],g.shape[2],g.shape[3],g.shape[4]]):g}const _2=M({maxPool3dGrad_:O2});const L2={kernelName:Sa,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=e;return{x:()=>_2(n,s,o,r,i,a,l)}}};function M2(n,t,e,s,o,r,i){const a=E(n,"dy","maxPoolGrad"),l=E(t,"input","maxPoolGrad"),c=E(e,"output","maxPoolGrad");S(l.rank===a.rank,()=>`Rank of input (${l.rank}) does not match rank of dy (${a.rank})`),S(a.rank===4,()=>`Error in maxPoolGrad: dy must be rank 4 but got rank ${a.rank}.`),S(l.rank===4,()=>`Error in maxPoolGrad: input must be rank 4 but got rank ${l.rank}.`),Le("maxPoolGrad",r,i);const u={dy:a,input:l,output:c},h={filterSize:s,strides:o,pad:r,dimRoundingMode:i};return O.runKernel(lu,u,h)}const P2=M({maxPoolGrad_:M2});const B2={kernelName:va,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{filterSize:r,strides:i,pad:a}=e;return{x:()=>P2(n,s,o,r,i,a)}}};const z2={kernelName:Na,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e,r=yt(o,s.shape),a=de(s.shape,r)[1],l=q(a);return{x:()=>{const u=s.shape.slice();r.forEach(p=>{u[p]=1});const h=L(n,u);return dt(F(h,ds(s.shape,"float32")),l)}}}};const V2={kernelName:Ta,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const s=e,{axis:o}=s,[r,i]=t,a=yt(o,r.shape),l=gg(n,i,r,a);return{x:()=>l.x()}}};const W2={kernelName:Er,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>F(n,nt(Ao(e,s),"float32")),b:()=>F(n,nt(Ye(e,s),"float32"))}}};const U2={kernelName:Ea,inputsToSave:["x"],gradFunc:(n,t,e)=>{const s=t[0],{paddings:o}=e,r=o.map(i=>i[0]);return{x:()=>Pt(n,r,s.shape)}}};const G2={kernelName:Rr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{const a=re(e.shape,o);return a.length>0?L(ut(n,a),e.shape):n},b:()=>{const a=F(n,Qt(ul(dt(e,s)))),l=re(s.shape,o);return l.length>0?L(ut(a,l),s.shape):a}}}};const H2={kernelName:Ar,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{const a=F(n,nt(s,"float32")),l=re(e.shape,o);return l.length>0?L(ut(a,l),e.shape):a},b:()=>{const a=F(n,nt(e,"float32")),l=re(s.shape,o);return l.length>0?L(ut(a,l),s.shape):a}}}};const q2={kernelName:Ra,gradFunc:n=>({x:()=>Qt(n)})};const K2={kernelName:Fa,inputsToSave:["indices"],gradFunc:(n,t)=>{const e=t[0];return{indices:()=>pe(e.shape,"float32")}}};const j2={kernelName:Da,gradFunc:n=>({x:()=>kt(n)})};const X2={kernelName:Oa,saveAllInputs:!0,gradFunc:(n,t,e)=>{const{axis:s}=e;return js(n,s).map(r=>()=>r)}};const bg={kernelName:_a,inputsToSave:["x"],gradFunc:(n,t,e)=>{const s=t[0],{paddings:o}=e,r=o.map(i=>i[0]);return{x:()=>Pt(n,r,s.shape)}}};const Y2={kernelName:Dr,inputsToSave:["a","b"],outputsToSave:[!0],gradFunc:(n,t)=>{const[e,s,o]=t,r=e,i=s,a=mt(r.shape,i.shape);return{a:()=>{const u=nt(i,"float32");let h=F(n,F(u,Us(r,pt(u,At(1)))));const d=re(r.shape,a);return d.length>0&&(h=ut(h,d)),L(h,r.shape)},b:()=>{const u=Ye(r,0),h=Re(u,Dn(r),kt(r));let d=F(n,F(o,h));const p=re(i.shape,a);return p.length>0&&(d=ut(d,p)),L(d,i.shape)}}}};const Z2={kernelName:La,inputsToSave:["x","alpha"],gradFunc:(n,t)=>{const[e,s]=t,o=Ye(e,0);return{x:()=>Re(o,n,F(n,s)),alpha:()=>{let r=Re(o,kt(n),F(n,e));const i=re(s.shape,n.shape);return i.length>0&&(r=ut(r,i)),L(r,s.shape)}}}};function J2(n,t,e){const s=n.shape.slice();s[e]=1;const o=L(t,s),r=qu(n,e,!0,!1),i=qu(n,e,!0,!0),a=F(r,i);return F(o,a)}function Q2(n,t,e){const s=n.shape.length,o=s-e.length,r=qt(e,s);let i=n;r!=null&&(i=vt(n,r));const a=i.shape.slice(),c=a.splice(s-e.length,e.length).reduce((d,p)=>d*p,1);a.push(c);const u=i.reshape(a);let h=J2(u,t,o);if(h=h.reshape(i.shape),r!=null){const d=us(r);h=vt(h,d)}return h}const tN={kernelName:Ma,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e;let r=[];return o==null?r=s.shape.map((i,a)=>a):typeof o=="number"?r=[o]:r=o,{x:()=>Q2(s,n,r)}}};const eN={kernelName:fr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{const a=dt(n,nt(s,"float32")),l=re(e.shape,o);return l.length>0?L(ut(a,l),e.shape):a},b:()=>{let a=F(n,nt(e,"float32"));const l=re(s.shape,o);l.length>0&&(a=L(ut(a,l),s.shape));const c=Vt(s);return Qt(dt(a,nt(c,"float32")))}}}};const nN={kernelName:Fr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,Qt(Vt(e)))}}};const sN={kernelName:_r,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t,s=F(Ao(e,6),pi(e));return{x:()=>F(n,nt(s,"float32"))}}};const oN={kernelName:Or,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,nt(pi(e),"float32"))}}};const rN={kernelName:Pa,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>L(n,e.shape)}}};const iN={kernelName:za,inputsToSave:["images"],gradFunc:(n,t,e)=>{const[s]=t,o={dy:n,images:s};return{images:()=>O.runKernel(gu,o,e)}}};const aN={kernelName:Ba,inputsToSave:["images"],gradFunc:(n,t,e)=>{const[s]=t,o={dy:n,images:s};return{images:()=>O.runKernel(mu,o,e)}}};const lN={kernelName:Va,gradFunc:(n,t,e)=>{const{dims:s}=e,o=yt(s,n.shape);return{x:()=>qs(n,o)}}};const cN={kernelName:Lr,gradFunc:n=>({x:()=>kt(n)})};const uN={kernelName:Mr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>Qt(dt(n,F(Us(e,1.5),2)))}}};const hN={kernelName:Wa,inputsToSave:["condition"],gradFunc:(n,t)=>{const[e]=t;return{condition:()=>nt(kt(e),"float32"),t:()=>F(n,nt(e,n.dtype)),e:()=>F(n,nt(Ju(e),n.dtype))}}};const dN={kernelName:Pr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Ye(e,At(0)),o=At(vl),r=At(Sl),i=F(n,r),a=F(F(n,o),An(nt(e,"float32")));return Re(s,i,a)}}}};const pN={kernelName:Wr,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,F(e,pt(At(1),e)))}}};const fN={kernelName:Vr,gradFunc:n=>({x:()=>kt(n)})};const mN={kernelName:Br,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(Hu(nt(e,"float32")),n)}}};const gN={kernelName:zr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(Bf(nt(e,"float32")),n)}}};const xN={kernelName:Ua,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{begin:o,size:r}=e,i=s.shape,[a,l]=$l(s,o,r),c=[];for(let u=0;u<n.rank;u++)c.push([a[u],i[u]-a[u]-l[u]]);return{x:()=>eh(n,c)}}};const bN={kernelName:Ka,outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s]=t,{dim:o}=e,r=!0,i=F(n,s);return{logits:()=>pt(i,F(ut(i,[o],r),s))}}};const yN={kernelName:Ur,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,Eo(e))}}};const yg={kernelName:Ha,gradFunc:(n,t,e)=>{const{blockShape:s,paddings:o}=e;return{x:()=>Uu(n,s,o)}}};const wg={kernelName:qa,gradFunc:(n,t,e)=>{const{axis:s}=e;return{x:()=>Me(n,s)}}};const wN={kernelName:Gr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,F(ke(nt(e,"float32")),2))}}};const CN={kernelName:xu,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(n,F(nt(e,"float32"),2))}}};const IN={kernelName:Hr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=At(2);return{a:()=>F(n,F(o,pt(e,s))),b:()=>F(n,F(o,pt(s,e)))}}};const $N={kernelName:Yr,gradFunc:n=>({x:()=>kt(n)})};const kN={kernelName:qr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(e.shape,s.shape);return{a:()=>{let a=n;const l=re(e.shape,o);return l.length>0&&(a=ut(a,l)),L(a,e.shape)},b:()=>{let a=n;const l=re(s.shape,o);return l.length>0&&(a=ut(a,l)),L(Qt(a),s.shape)}}}};const vN={kernelName:Ga,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,o=s.shape.slice(),{axis:r}=e;yt(r,s.shape).forEach(c=>{o[c]=1});const a=L(n,o),l=F(a,ds(s.shape,"float32"));return{x:()=>l}}};const SN={kernelName:Kr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>dt(n,Vt(Hu(e)))}}};const NN={kernelName:jr,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>F(pt(At(1),Vt(e)),n)}}};const TN={kernelName:Xr,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{reps:o}=e;return{x:()=>{let i=kt(s);if(s.rank===1)for(let a=0;a<o[0];++a)i=J(i,Pt(n,[a*s.shape[0]],[s.shape[0]]));else if(s.rank===2)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)i=J(i,Pt(n,[a*s.shape[0],l*s.shape[1]],[s.shape[0],s.shape[1]]));else if(s.rank===3)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)for(let c=0;c<o[2];++c)i=J(i,Pt(n,[a*s.shape[0],l*s.shape[1],c*s.shape[2]],[s.shape[0],s.shape[1],s.shape[2]]));else if(s.rank===4)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)for(let c=0;c<o[2];++c)for(let u=0;u<o[3];++u)i=J(i,Pt(n,[a*s.shape[0],l*s.shape[1],c*s.shape[2],u*s.shape[3]],[s.shape[0],s.shape[1],s.shape[2],s.shape[3]]));else throw new Error(`Gradient for tile operation is not implemented for rank-${s.rank} tensors yet.`);return i}}}};const EN={kernelName:Io,gradFunc:(n,t,e)=>{const s=e,{perm:o}=s,r=us(o);return{x:()=>vt(n,r)}}};const RN={kernelName:ja,gradFunc:(n,t,e)=>{const s=e,{axis:o}=s;return{value:()=>jn(n,o)}}};const AN={kernelName:Xa,inputsToSave:["segmentIds"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>DN(n,e)}}};function DN(n,t){const e=hs(t,kt(t)),s=Xu(n,e);let o=Gs(t,At(0,"int32"));const r=s.rank-o.rank;for(let a=0;a<r;++a)o=Pe(o,a+1);o=Kn(o,ds(s.shape,"bool"));const i=kt(s);return Re(o,s,i)}const FN={kernelName:Ya,gradFunc:n=>({x:()=>kt(n)})};const ON=[pg,AS,DS,FS,OS,_S,LS,MS,PS,BS,zS,VS,GS,KS,jS,XS,YS,ZS,JS,QS,t2,e2,s2,n2,i2,a2,l2,c2,u2,h2,eN,d2,p2,f2,m2,g2,b2,x2,y2,w2,C2,I2,$2,k2,v2,S2,N2,T2,E2,D2,xg,xg,F2,L2,B2,z2,V2,W2,U2,G2,H2,q2,K2,j2,X2,bg,bg,Y2,Z2,tN,nN,sN,oN,rN,iN,aN,lN,cN,uN,hN,dN,pN,fN,mN,gN,xN,bN,yN,yg,yg,wg,wg,wN,IN,CN,$N,kN,vN,SN,NN,TN,EN,RN,AN,FN];for(const n of ON)cw(n);G().prototype.abs=function(){return this.throwIfDisposed(),Ee(this)};G().prototype.acos=function(){return this.throwIfDisposed(),pC(this)};G().prototype.acosh=function(){return this.throwIfDisposed(),mC(this)};G().prototype.add=function(n){return this.throwIfDisposed(),J(this,n)};G().prototype.all=function(n,t){return this.throwIfDisposed(),_f(this,n,t)};G().prototype.any=function(n,t){return this.throwIfDisposed(),Bu(this,n,t)};G().prototype.argMax=function(n){return this.throwIfDisposed(),ni(this,n)};G().prototype.argMin=function(n){return this.throwIfDisposed(),wC(this,n)};G().prototype.asScalar=function(){return this.throwIfDisposed(),S(this.size===1,()=>"The array must have only 1 element."),L(this,[])};G().prototype.asType=function(n){return this.throwIfDisposed(),nt(this,n)};G().prototype.as1D=function(){return this.throwIfDisposed(),L(this,[this.size])};G().prototype.as2D=function(n,t){return this.throwIfDisposed(),L(this,[n,t])};G().prototype.as3D=function(n,t,e){return this.throwIfDisposed(),L(this,[n,t,e])};G().prototype.as4D=function(n,t,e,s){return this.throwIfDisposed(),L(this,[n,t,e,s])};G().prototype.as5D=function(n,t,e,s,o){return this.throwIfDisposed(),L(this,[n,t,e,s,o])};G().prototype.asin=function(){return this.throwIfDisposed(),IC(this)};G().prototype.asinh=function(){return this.throwIfDisposed(),kC(this)};G().prototype.atan=function(){return this.throwIfDisposed(),SC(this)};G().prototype.atan2=function(n){return this.throwIfDisposed(),TC(this,n)};G().prototype.atanh=function(){return this.throwIfDisposed(),RC(this)},G().prototype.avgPool=function(n,t,e,s){return this.throwIfDisposed(),Wu(this,n,t,e,s)};G().prototype.batchToSpaceND=function(n,t){return this.throwIfDisposed(),Uu(this,n,t)};G().prototype.batchNorm=function(n,t,e,s,o){return this.throwIfDisposed(),rl(this,n,t,e,s,o)};G().prototype.broadcastTo=function(n){return this.throwIfDisposed(),ii(this,n)};G().prototype.cast=function(n){return this.throwIfDisposed(),nt(this,n)};G().prototype.ceil=function(){return this.throwIfDisposed(),sI(this)};G().prototype.clipByValue=function(n,t){return this.throwIfDisposed(),Xe(this,n,t)};G().prototype.concat=function(n,t){return this.throwIfDisposed(),n instanceof oe&&(n=[n]),Me([this,...n],t)};G().prototype.conv1d=function(n,t,e,s,o,r){return this.throwIfDisposed(),Lf(this,n,t,e,s,o,r)};G().prototype.conv2dTranspose=function(n,t,e,s,o){return this.throwIfDisposed(),Mf(this,n,t,e,s,o)};G().prototype.conv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),Ws(this,n,t,e,s,o,r)};G().prototype.cos=function(){return this.throwIfDisposed(),Hu(this)};G().prototype.cosh=function(){return this.throwIfDisposed(),Bf(this)};G().prototype.cumprod=function(n,t,e){return this.throwIfDisposed(),qu(this,n,t,e)};G().prototype.cumsum=function(n,t,e){return this.throwIfDisposed(),zf(this,n,t,e)};G().prototype.depthToSpace=function(n,t){return this.throwIfDisposed(),TI(this,n,t)};G().prototype.depthwiseConv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),Ku(this,n,t,e,s,o,r)};G().prototype.dilation2d=function(n,t,e,s,o){return this.throwIfDisposed(),AI(this,n,t,e,s,o)};G().prototype.divNoNan=function(n){return this.throwIfDisposed(),LI(this,n)};G().prototype.div=function(n){return this.throwIfDisposed(),dt(this,n)};G().prototype.dot=function(n){return this.throwIfDisposed(),PI(this,n)};G().prototype.elu=function(){return this.throwIfDisposed(),al(this)};G().prototype.equal=function(n){return this.throwIfDisposed(),Rn(this,n)};G().prototype.erf=function(){return this.throwIfDisposed(),Wf(this)};G().prototype.euclideanNorm=function(n,t){return this.throwIfDisposed(),YI(this,n,t)};G().prototype.exp=function(){return this.throwIfDisposed(),An(this)};G().prototype.expandDims=function(n){return this.throwIfDisposed(),Pe(this,n)};G().prototype.expm1=function(){return this.throwIfDisposed(),t$(this)};G().prototype.fft=function(){return this.throwIfDisposed(),fm(this)};G().prototype.flatten=function(){return this.throwIfDisposed(),L(this,[this.size])};G().prototype.floor=function(){return this.throwIfDisposed(),ul(this)};G().prototype.floorDiv=function(n){return this.throwIfDisposed(),Of(this,n)};G().prototype.gather=function(n,t,e){return this.throwIfDisposed(),Xu(this,n,t,e)};G().prototype.greaterEqual=function(n){return this.throwIfDisposed(),Gs(this,n)};G().prototype.greater=function(n){return this.throwIfDisposed(),Ye(this,n)};G().prototype.ifft=function(){return this.throwIfDisposed(),ch(this)};G().prototype.irfft=function(){return this.throwIfDisposed(),Kk(this)};G().prototype.isFinite=function(){return this.throwIfDisposed(),c$(this)};G().prototype.isInf=function(){return this.throwIfDisposed(),h$(this)};G().prototype.isNaN=function(){return this.throwIfDisposed(),p$(this)};G().prototype.leakyRelu=function(n){return this.throwIfDisposed(),Zu(this,n)};G().prototype.lessEqual=function(n){return this.throwIfDisposed(),Ao(this,n)};G().prototype.less=function(n){return this.throwIfDisposed(),hl(this,n)};G().prototype.localResponseNormalization=function(n,t,e,s){return this.throwIfDisposed(),b$(this,n,t,e,s)};G().prototype.logSigmoid=function(){return this.throwIfDisposed(),v$(this)};G().prototype.logSoftmax=function(n){return this.throwIfDisposed(),Kf(this,n)};G().prototype.logSumExp=function(n,t){return this.throwIfDisposed(),jf(this,n,t)};G().prototype.log=function(){return this.throwIfDisposed(),Dn(this)};G().prototype.log1p=function(){return this.throwIfDisposed(),qf(this)};G().prototype.logicalAnd=function(n){return this.throwIfDisposed(),Kn(this,n)};G().prototype.logicalNot=function(){return this.throwIfDisposed(),Ju(this)};G().prototype.logicalOr=function(n){return this.throwIfDisposed(),Xf(this,n)};G().prototype.logicalXor=function(n){return this.throwIfDisposed(),F$(this,n)};G().prototype.matMul=function(n,t,e){return this.throwIfDisposed(),Rt(this,n,t,e)},G().prototype.maxPool=function(n,t,e,s){return this.throwIfDisposed(),Qu(this,n,t,e,s)};G().prototype.max=function(n,t){return this.throwIfDisposed(),mn(this,n,t)};G().prototype.maximum=function(n){return this.throwIfDisposed(),hs(this,n)};G().prototype.mean=function(n,t){return this.throwIfDisposed(),se(this,n,t)};G().prototype.min=function(n,t){return this.throwIfDisposed(),ll(this,n,t)};G().prototype.minimum=function(n){return this.throwIfDisposed(),ci(this,n)};G().prototype.mirrorPad=function(n,t){return this.throwIfDisposed(),V$(this,n,t)};G().prototype.mod=function(n){return this.throwIfDisposed(),U$(this,n)};G().prototype.mul=function(n){return this.throwIfDisposed(),F(this,n)};G().prototype.neg=function(){return this.throwIfDisposed(),Qt(this)};G().prototype.norm=function(n,t,e){return this.throwIfDisposed(),cl(this,n,t,e)};G().prototype.notEqual=function(n){return this.throwIfDisposed(),dl(this,n)};G().prototype.oneHot=function(n,t=1,e=0){return this.throwIfDisposed(),Yf(this,n,t,e)};G().prototype.onesLike=function(){return this.throwIfDisposed(),sn(this)};G().prototype.pad=function(n,t){return this.throwIfDisposed(),eh(this,n,t)},G().prototype.pool=function(n,t,e,s,o,r){return this.throwIfDisposed(),Q$(this,n,t,e,s,o,r)};G().prototype.pow=function(n){return this.throwIfDisposed(),Us(this,n)};G().prototype.prelu=function(n){return this.throwIfDisposed(),sh(this,n)};G().prototype.prod=function(n,t){return this.throwIfDisposed(),nk(this,n,t)};G().prototype.reciprocal=function(){return this.throwIfDisposed(),Nk(this)};G().prototype.relu=function(){return this.throwIfDisposed(),Hs(this)};G().prototype.relu6=function(){return this.throwIfDisposed(),im(this)};G().prototype.reshapeAs=function(n){return this.throwIfDisposed(),L(this,n.shape)};G().prototype.reshape=function(n){return this.throwIfDisposed(),L(this,n)};G().prototype.resizeBilinear=function(n,t,e){return this.throwIfDisposed(),Cm(this,n,t,e)};G().prototype.resizeNearestNeighbor=function(n,t,e){return this.throwIfDisposed(),Im(this,n,t,e)};G().prototype.reverse=function(n){return this.throwIfDisposed(),qs(this,n)};G().prototype.rfft=function(){return this.throwIfDisposed(),Yk(this)};G().prototype.round=function(){return this.throwIfDisposed(),am(this)};G().prototype.rsqrt=function(){return this.throwIfDisposed(),lm(this)};G().prototype.selu=function(){return this.throwIfDisposed(),cm(this)};G().prototype.separableConv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),um(this,n,t,e,s,o,r)};G().prototype.sigmoid=function(){return this.throwIfDisposed(),Eo(this)};G().prototype.sign=function(){return this.throwIfDisposed(),Lk(this)};G().prototype.sin=function(){return this.throwIfDisposed(),hm(this)};G().prototype.sinh=function(){return this.throwIfDisposed(),dm(this)};G().prototype.slice=function(n,t){return this.throwIfDisposed(),Pt(this,n,t)};G().prototype.softmax=function(n){return this.throwIfDisposed(),lh(this,n)};G().prototype.softplus=function(){return this.throwIfDisposed(),li(this)};G().prototype.spaceToBatchND=function(n,t){return this.throwIfDisposed(),nh(this,n,t)};G().prototype.split=function(n,t){return this.throwIfDisposed(),Ze(this,n,t)};G().prototype.sqrt=function(){return this.throwIfDisposed(),ke(this)};G().prototype.square=function(){return this.throwIfDisposed(),Vt(this)};G().prototype.squaredDifference=function(n){return this.throwIfDisposed(),Jk(this,n)};G().prototype.squeeze=function(n){return this.throwIfDisposed(),di(this,n)};G().prototype.stack=function(n,t){this.throwIfDisposed();const e=n instanceof oe?[this,n]:[this,...n];return jn(e,t)};G().prototype.step=function(n){return this.throwIfDisposed(),pi(this,n)};G().prototype.stridedSlice=function(n,t,e,s,o,r,i,a){return this.throwIfDisposed(),sv(this,n,t,e,s,o,r,i,a)};G().prototype.sub=function(n){return this.throwIfDisposed(),pt(this,n)};G().prototype.sum=function(n,t){return this.throwIfDisposed(),ut(this,n,t)};G().prototype.tan=function(){return this.throwIfDisposed(),rv(this)};G().prototype.tanh=function(){return this.throwIfDisposed(),ol(this)};G().prototype.tile=function(n){return this.throwIfDisposed(),gn(this,n)};G().prototype.toBool=function(){return this.throwIfDisposed(),nt(this,"bool")};G().prototype.toFloat=function(){return this.throwIfDisposed(),nt(this,"float32")};G().prototype.toInt=function(){return this.throwIfDisposed(),nt(this,"int32")};G().prototype.topk=function(n,t){return this.throwIfDisposed(),lv(this,n,t)};G().prototype.transpose=function(n){return this.throwIfDisposed(),vt(this,n)};G().prototype.unique=function(n){return this.throwIfDisposed(),hv(this,n)};G().prototype.unsortedSegmentSum=function(n,t){return this.throwIfDisposed(),xm(this,n,t)};G().prototype.unstack=function(n){return this.throwIfDisposed(),js(this,n)};G().prototype.where=function(n,t){return this.throwIfDisposed(),Re(n,this,t)};G().prototype.zerosLike=function(){return this.throwIfDisposed(),kt(this)};class On extends Error{constructor(t){super(t),Object.setPrototypeOf(this,On.prototype)}}class rn extends Error{constructor(t){super(t),Object.setPrototypeOf(this,rn.prototype)}}class D extends Error{constructor(t){super(t),Object.setPrototypeOf(this,D.prototype)}}class bt extends Error{constructor(t){super(t),Object.setPrototypeOf(this,bt.prototype)}}class jh extends Error{constructor(t){super(t),Object.setPrototypeOf(this,jh.prototype)}}class Cg{constructor(t){this.maxEntries=t||100,this.cache=new Map}get(t){let e;return this.cache.has(t)&&(e=this.cache.get(t),this.cache.delete(t),this.cache.set(t,e)),e}put(t,e){if(this.cache.has(t))this.cache.delete(t);else if(this.cache.size>=this.maxEntries){const s=this.cache.keys().next().value;this.cache.delete(s)}this.cache.set(t,e)}getMaxEntries(){return this.maxEntries}setMaxEntries(t){if(t<0)throw new Error(`The maxEntries of LRU caches must be at least 0, but got ${t}.`);if(this.maxEntries>t)for(let e=0;e<this.maxEntries-t;e++){const s=this.cache.keys().next().value;this.cache.delete(s)}this.maxEntries=t}}function Ys(n,t){if(Array.isArray(n)){let e=[];for(let s=0;s<t;s++)e=e.concat(n);return e}else{const e=new Array(t);return e.fill(n),e}}function _n(n,t){if(!n)throw new jh(t)}function Ig(n,t){let e=0;for(const s of n)s===t&&e++;return e}function Be(n){return n.length===1?n[0]:n}function Dt(n){return Array.isArray(n)?n:[n]}function Jn(n){const e=n.replace(/(.)([A-Z][a-z0-9]+)/g,"$1_$2").replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase();return e[0]!=="_"?e:"private"+e}function Zs(n){return n.length<=1||n.indexOf("_")===-1?n:n.replace(/[_]+(\w|$)/g,(t,e)=>e.toUpperCase())}let an={};function Xh(n){if(n==null)return null;const t={};return t.className=n.getClassName(),t.config=n.getConfig(),t}function Yh(n){if(!(n==null||typeof n!="object"))if(Array.isArray(n))n.forEach(t=>Yh(t));else{const t=Object.keys(n);for(const e of t){const s=n[e];s!=null&&typeof s=="object"&&(!Array.isArray(s)&&s.type==="ndarray"&&typeof s.value=="number"?n[e]=s.value:Yh(s))}}}function xi(n,t={},e={},s="object",o=!1){if(typeof n=="string"){const r=n;let i;if(r in e)i=e[r];else if(r in an)i=an[r];else if(i=t[r],i==null)throw new D(`Unknown ${s}: ${n}. This may be due to one of the following reasons:
1. The ${s} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${s} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return i}else{const r=n;if(r.className==null||r.config==null)throw new D(`${s}: Improper config format: ${JSON.stringify(r)}.
'className' and 'config' must set.`);const i=r.className;let a,l;if(i in e?[a,l]=e[i]:i in an?[a,l]=an.className:i in t&&([a,l]=t[i]),a==null)throw new D(`Unknown ${s}: ${i}. This may be due to one of the following reasons:
1. The ${s} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${s} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(l!=null){const c={};for(const p of Object.keys(an))c[p]=an[p];for(const p of Object.keys(e))c[p]=e[p];const u=r.config;u.customObjects=c;const h=Object.assign({},an);for(const p of Object.keys(e))an[p]=e[p];Yh(r.config);const d=l(a,r.config,e,o);return an=Object.assign({},h),d}else{const c=Object.assign({},an);for(const h of Object.keys(e))an[h]=e[h];const u=new a(r.config);return an=Object.assign({},c),u}}}function _N(n,t){return n<t?-1:n>t?1:0}function Nl(n,t){return-1*_N(n,t)}function fs(n){if(n==null)return n;const t=[];for(const e of n)t.indexOf(e)===-1&&t.push(e);return t}function LN(n){if(n==null)throw new D(`Invalid value in obj: ${JSON.stringify(n)}`);for(const t in n)if(n.hasOwnProperty(t))return!1;return!0}function Js(n,t,e){if(e!=null&&n.indexOf(e)<0)throw new D(`${e} is not a valid ${t}.  Valid values are ${n} or null/undefined.`)}function Zh(n,t,e=0,s=1/0){return _n(e>=0),_n(s>=e),Array.isArray(n)&&n.length>=e&&n.length<=s&&n.every(o=>typeof o===t)}function fe(n,t){Array.isArray(n)?(S(n.length>0,()=>`${t} is unexpectedly an empty array.`),n.forEach((e,s)=>fe(e,`element ${s+1} of ${t}`))):S(Number.isInteger(n)&&n>0,()=>`Expected ${t} to be a positive integer, but got ${$g(n)}.`)}function $g(n){return n===null?"null":Array.isArray(n)?"["+n.map(t=>$g(t)).join(",")+"]":typeof n=="string"?`"${n}"`:`${n}`}function MN(n,t,e){let s=e!=null?e():Oe(),o;return(...i)=>{const a=e!=null?e():Oe();return a-s<t||(s=a,o=n(...i)),o}}function kg(n){return n==="relu"?"relu":n==="linear"?"linear":n==="elu"?"elu":null}let PN=0;function vg(){return PN++}const Tl={};function El(n=""){return n in Tl||(Tl[n]=0),Tl[n]+=1,n+Tl[n].toString()}const BN=["channelsFirst","channelsLast"],zN=["nearest","bilinear"],VN=["valid","same","causal"],WN=["max","avg"],UN=["sum","mul","concat","ave"];const Lo=new Map;function te(n){Js(BN,"DataFormat",n)}function GN(n){Js(zN,"InterpolationFormat",n)}function Je(n){Js(VN,"PaddingMode",n)}function Sg(n){Js(WN,"PoolMode",n)}const bi=[],Ng="/";function Qs(n,t){bi.push(n);try{const e=t();return bi.pop(),e}catch(e){throw bi.pop(),e}}function HN(){return bi.length===0?"":bi.join(Ng)+Ng}function Tg(n){if(!Rg(n))throw new Error("Not a valid tensor name: '"+n+"'");return HN()+n}function Eg(n){if(!Rg(n))throw new Error("Not a valid tensor name: '"+n+"'");Lo.has(n)||Lo.set(n,0);const t=Lo.get(n);if(Lo.set(n,Lo.get(n)+1),t>0){const e=`${n}_${t}`;return Lo.set(e,1),e}else return n}const qN=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function Rg(n){return!!n.match(qN)}function KN(n){return n===parseInt(n.toString(),10)}function ms(n,t,e){t==null&&(t=0),e==null&&(e=n.length);let s=1;for(let o=t;o<e;++o)s*=n[o];return s}function Mo(n){if(n.length===0)return Number.NaN;let t=Number.POSITIVE_INFINITY;for(let e=0;e<n.length;e++){const s=n[e];s<t&&(t=s)}return t}function gs(n){if(n.length===0)return Number.NaN;let t=Number.NEGATIVE_INFINITY;for(let e=0;e<n.length;e++){const s=n[e];s>t&&(t=s)}return t}function bn(n,t){if(t<n)throw new D(`end (${t}) < begin (${n}) is forbidden.`);const e=[];for(let s=n;s<t;++s)e.push(s);return e}let Jh;function ie(){return Jh==null&&(Jh=Pw().epsilon()),Jh}function yn(){return"channelsLast"}function Ln(n,t){return nt(n,t)}function yi(n,t=-1){const e=n.shape.slice();return t<0&&(t=e.length+t+1),e.splice(t,0,1),L(n,e)}function jN(n,t){return V(()=>{if(n.shape.length!==2)throw new D(`repeat() expects a rank-2 tensor, but received a rank-${n.shape.length} tensor.`);const e=yi(n,1);return ed(e,[1,t,1])})}function XN(n){const t=[ms(n.shape)];return L(n,t)}function YN(n){if(n.rank<=1)throw new D(`batchFlatten requires a minimum rank of 2. Got rank: ${n.rank}.`);const t=[n.shape[0],ms(n.shape,1)];return L(n,t)}function to(n,t,e){return V(()=>{switch(n.rank){case 1:return ih(n,t,e);case 2:return pm(n,[t,0],[e,n.shape[1]]);case 3:return ah(n,[t,0,0],[e,n.shape[1],n.shape[2]]);case 4:return Cl(n,[t,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3]]);case 5:return Pt(n,[t,0,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3],n.shape[4]]);case 6:return Pt(n,[t,0,0,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3],n.shape[4],n.shape[5]]);default:throw new D(`sliceAlongFirstAxis() received an unsupported tensor rank: ${n.rank}`)}})}function Qh(n,t,e){return V(()=>{switch(n.rank){case 1:return ih(n,t,e);case 2:return pm(n,[0,t],[n.shape[0],e]);case 3:return ah(n,[0,0,t],[n.shape[0],n.shape[1],e]);case 4:return Cl(n,[0,0,0,t],[n.shape[0],n.shape[1],n.shape[2],e]);default:throw new D(`sliceAlongLastAxis() received an unsupported tensor rank: ${n.rank}`)}})}function Rl(n,t,e,s){return V(()=>{switch(n.rank){case 1:return ih(n,t,e);case 2:switch(s){case 1:return to(n,t,e);case 2:return Qh(n,t,e);default:throw new D(`The axis is not within the rank of the tensor ${s}`)}case 3:switch(s){case 1:return to(n,t,e);case 2:return ah(n,[0,t,0],[n.shape[0],e,n.shape[2]]);case 3:return Qh(n,t,e);default:throw new D(`The axis is not within the rank of the tensor ${s}`)}case 4:switch(s){case 1:return to(n,t,e);case 2:return Cl(n,[0,t,0,0],[n.shape[0],e,n.shape[2],n.shape[3]]);case 3:return Cl(n,[0,0,t,0],[n.shape[0],n.shape[1],e,n.shape[3]]);case 4:return Qh(n,t,e);default:throw new D(`The axis is not within the rank of the tensor ${s}`)}default:throw new D(`sliceAlongLastAxis() received an unsupported tensor rank: ${n.rank}`)}})}function td(n,t=-1){let e;return t<0&&(e=n[0].rank,e!==0?t=e:t=0),t===n[0].rank&&(t=-1),Me(n,t)}function Ag(n,t){switch(n.rank){case 1:return iI([n,t]);case 2:return lI([n,t],0);case 3:return uI([n,t],0);case 4:return dI([n,t],0);default:throw new D(`concatAlongFirstAxis() received an unsupported tensor rank: ${n.rank}`)}}function ed(n,t){if(Array.isArray(t)||(t=[t]),n.rank!==t.length)throw new D(`The length of input n (${t.length}) does not match the number of dimensions in input x (${n.rank})`);return gn(n,t)}function Al(n,t=0,e=1,s,o){return $k(n,t,e,s,o)}function Mn(n,t,e,s){if(n.rank<2||t.rank<2)throw new bt(`dot requires both inputs to be rank >= 2 but got x shape = ${n.shape} and y shape = ${t.shape}`);if(t.rank>=3){const o=n.shape.slice(-1)[0],r=t.shape.slice(-2)[0];if(o!==r)throw new bt(`If rank y >= 3, then the second last dim of y must equal the last dim of x but got x shape = ${n.shape} and  y shape = ${t.shape}`)}if(n.rank===2&&t.rank===2)return ym({a:n,b:t,transposeA:!1,transposeB:!1,bias:s?nd(n.rank,s,yn()):null,activation:e});{const o=n.shape.slice(),r=o.pop();n=L(n,[-1,r]);const i=t.shape.slice(),a=i.pop(),l=i.pop(),c=[...i,a],u=Array.from({length:t.rank},(f,m)=>m===0?t.rank-2:m<=t.rank-2?m-1:m);t=L(vt(t,u),[l,-1]);const h=[...o,...c];return L(ym({a:n,b:t,transposeA:!1,transposeB:!1,bias:s?nd(n.rank,s,yn()):null,activation:e}),h)}}function Dg(n,t,e){return V(()=>(Array.isArray(t)?t=Ge(t,"int32"):t=nt(t,"int32"),Xu(n,t,e)))}function wi(n){return F(n,n)}function nd(n,t,e){const s=t.shape;if(t.rank!==1&&t.rank!==n)throw new D(`Unexpected bias dimensions: ${t.rank}; expected it to be 1 or ${n}`);if(n===5){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1,1,1]):L(t,[1,s[3],s[0],s[1],s[2]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,1,1,s[0]]):L(t,[1].concat(s))}else if(n===4){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1,1]):L(t,[1,s[2],s[0],s[1]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,1,s[0]]):L(t,[1].concat(s))}else if(n===3){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1]):L(t,[1,s[1],s[0]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,s[0]]):L(t,[1].concat(s))}else if(n<3)return t;throw new D(`Unsupported input rank by biasAdd: ${t.rank}`)}function wn(n,t,e){return V(()=>(e==null&&(e=yn()),te(e),J(n,nd(n.rank,t,e))))}function ZN(n,t=1){if(t!==1)throw new bt(`Support for alpha values other than 1 (${t}) is not implemented yet.`);return al(n)}function JN(n){return V(()=>dt(n,J(Ee(n),1)))}function Fg(n,t,e,s){return V(()=>bv(n,t,e,s))}function QN(n){return V(()=>{const t=J(.5,F(.2,n));return Xe(t,0,1)})}function Ci(n,t,e=!1){return e?n():t()}const tT=["fanIn","fanOut","fanAvg"],eT=["normal","uniform","truncatedNormal"];function nT(n){Js(tT,"FanMode",n)}function sT(n){Js(eT,"Distribution",n)}class ln extends _o{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}}class Og extends ln{apply(t,e){return pe(t,e)}}Og.className="Zeros",Z(Og);class sd extends ln{apply(t,e){return ds(t,e)}}sd.className="Ones",Z(sd);class _g extends ln{constructor(t){if(super(),typeof t!="object")throw new D(`Expected argument of type ConstantConfig but got ${t}`);if(t.value===void 0)throw new D(`config must have value set but got ${t}`);this.value=t.value}apply(t,e){return V(()=>F(At(this.value),ds(t,e)))}getConfig(){return{value:this.value}}}_g.className="Constant",Z(_g);class Lg extends ln{constructor(t){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=t.minval||this.DEFAULT_MINVAL,this.maxval=t.maxval||this.DEFAULT_MAXVAL,this.seed=t.seed}apply(t,e){return ui(t,this.minval,this.maxval,e,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}}Lg.className="RandomUniform",Z(Lg);class Mg extends ln{constructor(t){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=t.mean||this.DEFAULT_MEAN,this.stddev=t.stddev||this.DEFAULT_STDDEV,this.seed=t.seed}apply(t,e){if(e=e||"float32",e!=="float32"&&e!=="int32")throw new bt(`randomNormal does not support dType ${e}.`);return Al(t,this.mean,this.stddev,e,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Mg.className="RandomNormal",Z(Mg);class Pg extends ln{constructor(t){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=t.mean||this.DEFAULT_MEAN,this.stddev=t.stddev||this.DEFAULT_STDDEV,this.seed=t.seed}apply(t,e){if(e=e||"float32",e!=="float32"&&e!=="int32")throw new bt(`truncatedNormal does not support dType ${e}.`);return gm(t,this.mean,this.stddev,e,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Pg.className="TruncatedNormal",Z(Pg);class Bg extends ln{constructor(t){super(),this.gain=t.gain!=null?t.gain:1}apply(t,e){return V(()=>{if(t.length!==2||t[0]!==t[1])throw new D("Identity matrix initializer can only be used for 2D square matrices.");return F(this.gain,Hf(t[0]))})}getConfig(){return{gain:this.gain}}}Bg.className="Identity",Z(Bg);function oT(n,t="channelsLast"){let e,s;if(te(t),n.length===2)e=n[0],s=n[1];else if([3,4,5].indexOf(n.length)!==-1){if(t==="channelsFirst"){const o=ms(n,2);e=n[1]*o,s=n[0]*o}else if(t==="channelsLast"){const o=ms(n,0,n.length-2);e=n[n.length-2]*o,s=n[n.length-1]*o}}else{const o=ms(n);e=Math.sqrt(o),s=Math.sqrt(o)}return[e,s]}class He extends ln{constructor(t){if(super(),t.scale<0)throw new D(`scale must be a positive float. Got: ${t.scale}`);this.scale=t.scale==null?1:t.scale,this.mode=t.mode==null?"fanIn":t.mode,nT(this.mode),this.distribution=t.distribution==null?"normal":t.distribution,sT(this.distribution),this.seed=t.seed}apply(t,e){const s=oT(t),o=s[0],r=s[1];let i=this.scale;if(this.mode==="fanIn"?i/=Math.max(1,o):this.mode==="fanOut"?i/=Math.max(1,r):i/=Math.max(1,(o+r)/2),this.distribution==="normal"){const a=Math.sqrt(i);if(e=e||"float32",e!=="float32"&&e!=="int32")throw new bt(`${this.getClassName()} does not support dType ${e}.`);return gm(t,0,a,e,this.seed)}else{const a=Math.sqrt(3*i);return ui(t,-a,a,e,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}}He.className="VarianceScaling",Z(He);class od extends He{constructor(t){super({scale:1,mode:"fanAvg",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return He.className}}od.className="GlorotUniform",Z(od);class rd extends He{constructor(t){super({scale:1,mode:"fanAvg",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return He.className}}rd.className="GlorotNormal",Z(rd);class id extends He{constructor(t){super({scale:2,mode:"fanIn",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return He.className}}id.className="HeNormal",Z(id);class ad extends He{constructor(t){super({scale:2,mode:"fanIn",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return He.className}}ad.className="HeUniform",Z(ad);class ld extends He{constructor(t){super({scale:1,mode:"fanIn",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return He.className}}ld.className="LeCunNormal",Z(ld);class cd extends He{constructor(t){super({scale:1,mode:"fanIn",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return He.className}}cd.className="LeCunUniform",Z(cd);class zg extends ln{constructor(t){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=t.gain==null?this.DEFAULT_GAIN:t.gain,this.seed=t.seed}apply(t,e){return V(()=>{if(t.length<2)throw new bt("Shape must be at least 2D.");if(e!=="int32"&&e!=="float32"&&e!==void 0)throw new TypeError(`Unsupported data type ${e}.`);e=e;const s=q(t.slice(0,-1)),o=t[t.length-1],r=s*o;r>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${r}) elements: Slowness may result.`);const i=[Math.max(o,s),Math.min(o,s)],a=Al(i,0,1,e,this.seed),l=fS.qr(a,!1);let c=l[0];const h=l[1].flatten().stridedSlice([0],[Math.min(o,s)*Math.min(o,s)],[Math.min(o,s)+1]);return c=F(c,h.sign()),s<o&&(c=c.transpose()),F(At(this.gain),c.reshape(t))})}getConfig(){return{gain:this.gain,seed:this.seed}}}zg.className="Orthogonal",Z(zg);const Vg={constant:"Constant",glorotNormal:"GlorotNormal",glorotUniform:"GlorotUniform",heNormal:"HeNormal",heUniform:"HeUniform",identity:"Identity",leCunNormal:"LeCunNormal",leCunUniform:"LeCunUniform",ones:"Ones",orthogonal:"Orthogonal",randomNormal:"RandomNormal",randomUniform:"RandomUniform",truncatedNormal:"TruncatedNormal",varianceScaling:"VarianceScaling",zeros:"Zeros"};function Wg(n,t={}){return xi(n,on.getMap().classNameMap,t,"initializer")}function Kt(n){return Xh(n)}function Ut(n){if(typeof n=="string"){const t=n in Vg?Vg[n]:n;if(t==="GlorotNormal")return new rd;if(t==="GlorotUniform")return new od;if(t==="HeNormal")return new id;if(t==="HeUniform")return new ad;if(t==="LeCunNormal")return new ld;if(t==="LeCunUniform")return new cd;{const e={};return e.className=t,e.config={},Wg(e)}}else return n instanceof ln?n:Wg(n)}function ud(n){return Array.isArray(n)&&Array.isArray(n[0])}function Dl(n){return n.length===0?[]:Array.isArray(n[0])?n:[n]}function ft(n){let t;if(Array.isArray(n)){if(n.length!==1)throw new D(`Expected Tensor length to be 1; got ${n.length}`);t=n[0]}else t=n;return t}function Nt(n){if(Array.isArray(n)&&Array.isArray(n[0])){if(n.length===1)return n=n,n[0];throw new D(`Expected exactly 1 Shape; got ${n.length}`)}else return n}function Fl(n){let t=0;for(const e of n)e.shape.length===0?t+=1:t+=e.shape.reduce((s,o)=>s*o);return t}const Ug="Variable";class rT{constructor(t,e="float32",s=Ug,o=!0,r=null){this.dtype=e??"float32",this.shape=t.shape,this.id=vg(),s=s??Ug,this.originalName=Tg(s),this.name=Eg(this.originalName),this.trainable_=o,this.constraint=r,this.val=fv(t,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(t){return this.assertNotDisposed(),iT(this.val,t),this.val.id!==t.id&&(this.val.assign(t),this.constraint!=null&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw new Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(t){this.trainable_=t,this.val.trainable=t}}function iT(n,t){if(n.shape.toString()!==t.shape.toString())throw new Error("Shape mismatch: "+JSON.stringify(n.shape)+" vs. "+JSON.stringify(t.shape))}function hd(n){return n.map(t=>t.read())}function dd(n){n.forEach(t=>{t[0].write(t[1])})}class ae{constructor(t){this.dtype=t.dtype,this.shape=t.shape,t.shape!=null?this.ndim=t.shape.length:this.ndim=t.ndim,this.maxNDim=t.maxNDim,this.minNDim=t.minNDim,this.axes=t.axes||{}}}class Pn{constructor(t,e,s,o,r,i,a){this.dtype=t,this.shape=e,this.sourceLayer=s,this.inputs=o,this.callArgs=r,this.outputTensorIndex=a,this.id=vg(),i!=null&&(this.originalName=Tg(i),this.name=Eg(this.originalName)),this.rank=e.length}}let aT=0;class Ol{constructor(t,e){this.callArgs=e,this.id=aT++,this.outboundLayer=t.outboundLayer,this.inboundLayers=t.inboundLayers,this.nodeIndices=t.nodeIndices,this.tensorIndices=t.tensorIndices,this.inputTensors=t.inputTensors,this.outputTensors=t.outputTensors,this.inputMasks=t.inputMasks,this.outputMasks=t.outputMasks,this.inputShapes=t.inputShapes,this.outputShapes=t.outputShapes;for(const s of t.inboundLayers)s?.outboundNodes.push(this);t.outboundLayer.inboundNodes.push(this)}getConfig(){const t=[];for(const e of this.inboundLayers)e!=null?t.push(e.name):t.push(null);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:t,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}}let lT=0;class Ct extends _o{constructor(t={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=lT++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let e=t.name;if(!e){const s=this.getClassName();e=Jn(s)+"_"+El(s)}if(this.name=e,this.trainable_=t.trainable==null?!0:t.trainable,t.inputShape!=null||t.batchInputShape!=null){let s;if(t.batchInputShape!=null)s=t.batchInputShape;else if(t.inputShape!=null){let r=null;t.batchSize!=null&&(r=t.batchSize),s=[r].concat(t.inputShape)}this.batchInputShape=s;let o=t.dtype;o==null&&(o=t.inputDType),o==null&&(o="float32"),this.dtype=o}t.weights!=null?this.initialWeights=t.weights:this.initialWeights=null,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(t,e){return t.name+"_ib-"+e.toString()}getNodeAtIndex(t,e){if(this.inboundNodes.length===0)throw new rn(`The layer has never been called and thus has no defined ${e}.`);if(this.inboundNodes.length<=t)throw new D(`Asked to get ${e} at node ${t}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[t]}getInputAt(t){return Be(this.getNodeAtIndex(t,"input").inputTensors)}getOutputAt(t){return Be(this.getNodeAtIndex(t,"output").outputTensors)}get input(){if(this.inboundNodes.length>1)throw new On(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(this.inboundNodes.length===0)throw new On(`Layer ${this.name} is not connected, no input to return.`);return Be(this.getNodeAtIndex(0,"input").inputTensors)}get output(){if(this.inboundNodes.length===0)throw new On(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new On(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return Be(this.getNodeAtIndex(0,"output").outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(t=>t())}get updates(){return this._updates}get built(){return this._built}set built(t){this._built=t}get trainable(){return this.trainable_}set trainable(t){this._trainableWeights.forEach(e=>e.trainable=t),this.trainable_=t}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(t=>t.trainable):[]}set trainableWeights(t){this._trainableWeights=t}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(t=>!t.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(t){this._nonTrainableWeights=t}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw new Error("Cannot call the resetStates() method of a non-stateful Layer object.")}assertInputCompatibility(t){const e=Dt(t);if(this.inputSpec==null||this.inputSpec.length===0)return;const s=Dt(this.inputSpec);if(e.length!==s.length)throw new D(`Layer ${this.name} expects ${s.length} inputs, but it received ${e.length} input tensors. Input received: ${t}`);for(let o=0;o<e.length;o++){const r=e[o],i=s[o];if(i==null)continue;const a=r.rank;if(i.ndim!=null&&a!==i.ndim)throw new D(`Input ${o} is incompatible with layer ${this.name}: expected ndim=${i.ndim}, found ndim=${a}`);if(i.maxNDim!=null&&a>i.maxNDim)throw new D(`Input ${o} is incompatible with layer ${this.name}: expected max_ndim=${i.maxNDim}, found ndim=${a}`);if(i.minNDim!=null&&a<i.minNDim)throw new D(`Input ${o} is incompatible with layer ${this.name}: expected min_ndim=${i.minNDim}, found ndim=${a}.`);if(i.dtype!=null&&r.dtype!==i.dtype)throw new D(`Input ${o} is incompatible with layer ${this.name} : expected dtype=${i.dtype}, found dtype=${r.dtype}.`);if(i.axes){const l=r.shape;for(const c in i.axes){const u=Number(c),h=i.axes[c],d=u>=0?l[u]:l[l.length+u];if(h!=null&&[h,null].indexOf(d)===-1)throw new D(`Input ${o} is incompatible with layer ${this.name}: expected axis ${u} of input shape to have value ${h} but got shape ${l}.`)}}if(i.shape!=null)for(let l=0;l<i.shape.length;++l){const c=i.shape[l],u=r.shape[l];if(c!=null&&u!=null&&c!==u)throw new D(`Input ${o} is incompatible with layer ${this.name}: expected shape=${i.shape}, found shape=${r.shape}.`)}}}call(t,e){return t}invokeCallHook(t,e){this._callHook!=null&&this._callHook(t,e)}setCallHook(t){this._callHook=t}clearCallHook(){this._callHook=null}apply(t,e){e=e||{},this.assertNotDisposed();const s=Dt(t),o=hT(t),r=dT(t);if(o===r)throw new D("Arguments to apply() must be all SymbolicTensors or all Tensors");return Qs(this.name,()=>{if(!this.built){this.assertInputCompatibility(t);const i=[];for(const a of Dt(t))i.push(a.shape);this.build(Be(i)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),this._refCount===null&&r&&(this._refCount=1)}if(this.assertInputCompatibility(t),r){let i=this.call(t,e);this.supportsMasking&&this.setMaskMetadata(t,i);const a=Dt(i),l=[];for(let c of a)s.indexOf(c)!==-1&&(c=c.clone()),l.push(c);if(i=Be(l),this.activityRegularizer!=null)throw new bt("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return i}else{const i=cT(t),a=this.computeOutputShape(i);let l;const c=uT(t);if(this.warnOnIncompatibleInputShape(Array.isArray(t)?i[0]:i),a!=null&&a.length>0&&Array.isArray(a[0])?l=a.map((u,h)=>new Pn(c,u,this,Dt(t),e,this.name,h)):l=new Pn(c,a,this,Dt(t),e,this.name),this.addInboundNode(t,l,null,null,i,a,e),this._refCount++,this.activityRegularizer!=null)throw new bt("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return l}})}warnOnIncompatibleInputShape(t){if(this.batchInputShape!=null)if(t.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(t)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let e=!1;this.batchInputShape.forEach((s,o)=>{s!=null&&t[o]!=null&&t[o]!==s&&(e=!0)}),e&&console.warn(`The shape of the input tensor (${JSON.stringify(t)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}get outputShape(){if(this.inboundNodes==null||this.inboundNodes.length===0)throw new On(`The layer ${this.name} has never been called and thus has no defined output shape.`);const t=[];for(const e of this.inboundNodes){const s=JSON.stringify(e.outputShapes);t.indexOf(s)===-1&&t.push(s)}if(t.length===1){const e=this.inboundNodes[0].outputShapes;return Array.isArray(e)&&Array.isArray(e[0])&&e.length===1?e[0]:e}else throw new On(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new rn(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return Fl(this.weights)}build(t){this.built=!0}getWeights(t=!1){return hd(t?this.trainableWeights:this.weights)}setWeights(t){V(()=>{const e=this.weights;if(e.length!==t.length)throw new D(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${t.length}, but the layer was expecting ${e.length} weights. Provided weights: ${t}...`);if(e.length===0)return;const s=[],o=hd(e);for(let r=0;r<o.length;++r){const i=o[r],a=e[r],l=t[r];if(!Et(i.shape,l.shape))throw new D(`Layer weight shape ${i.shape} not compatible with provided weight shape ${l.shape}`);s.push([a,l])}dd(s)})}addWeight(t,e,s,o,r,i,a,l){if(this._addedWeightNames.indexOf(t)!==-1)throw new D(`Duplicate weight name ${t} for layer ${this.name}`);this._addedWeightNames.push(t),s==null&&(s="float32"),this.fastWeightInitDuringBuild&&(o=l!=null?l():Ut("zeros"));const c=o.apply(e,s),u=new rT(c,s,t,i,a);return c.dispose(),r!=null&&this.addLoss(()=>r.apply(u.read())),i==null&&(i=!0),i?this._trainableWeights.push(u):this._nonTrainableWeights.push(u),u}setFastWeightInitDuringBuild(t){this.fastWeightInitDuringBuild=t}addLoss(t){t==null||Array.isArray(t)&&t.length===0||(t=Dt(t),this._losses!==void 0&&this._losses!==null&&this.losses.push(...t))}computeOutputShape(t){return t}computeMask(t,e){if(!this.supportsMasking){if(e!=null)if(Array.isArray(e))e.forEach(s=>{if(s!=null)throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`);return null}return e}setMaskMetadata(t,e,s){if(!this.supportsMasking)return;const o=this.computeMask(t,s),r=Dt(e),i=Dt(o);if(r.length!==i.length)throw new Error(`${this.name} outputs ${r.length} tensors but ${r.length} masks for those tensors`);for(let a=0;a<r.length;a++)r[a].kerasMask=i[a]}addInboundNode(t,e,s,o,r,i,a=null){const l=Dt(t);e=Dt(e),s=Dt(s),o=Dt(o),r=Dl(r),i=Dl(i);const c=[],u=[],h=[];for(const d of l)c.push(d.sourceLayer),u.push(d.nodeIndex),h.push(d.tensorIndex);new Ol({outboundLayer:this,inboundLayers:c,nodeIndices:u,tensorIndices:h,inputTensors:l,outputTensors:e,inputMasks:s,outputMasks:o,inputShapes:r,outputShapes:i},a);for(let d=0;d<e.length;d++)e[d].sourceLayer=this,e[d].nodeIndex=this.inboundNodes.length-1,e[d].tensorIndex=d}getConfig(){const t={name:this.name,trainable:this.trainable};return this.batchInputShape!=null&&(t.batchInputShape=this.batchInputShape),this.dtype!=null&&(t.dtype=this.dtype),t}disposeWeights(){return this.weights.forEach(t=>t.dispose()),this.weights.length}assertNotDisposed(){if(this._refCount===0)throw new Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw new Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(this._refCount===null)throw new Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let t=0;return--this._refCount===0&&(t=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:t}}}function cT(n){n=Dt(n);const t=[];for(const e of n)t.push(e.shape);return Be(t)}function uT(n){return"float32"}function Gg(n,t,e){if((t==null||e!=null&&e>0)&&(t=n.sourceLayer,e=n.nodeIndex),t.inboundNodes.length===0)return[n];{const s=t.inboundNodes[e];if(s.inboundLayers.length===0)return s.inputTensors;{const o=[];for(let r=0;r<s.inboundLayers.length;r++){const i=s.inputTensors[r],a=s.inboundLayers[r],l=s.nodeIndices[r],c=Gg(i,a,l);for(const u of c)o.indexOf(u)===-1&&o.push(u)}return o}}}function hT(n){let t=!0;for(const e of Dt(n))if(!(e instanceof Pn)){t=!1;break}return t}function dT(n){let t=!0;for(const e of Dt(n))if(e instanceof Pn){t=!1;break}return t}class Ii extends Ct{constructor(t){if(super({dtype:t.dtype,name:t.name!=null?t.name:El("input").toString()}),t.batchSize==null&&(t.batchSize=null),t.sparse==null&&(t.sparse=!1),this.trainable=!1,this.built=!0,this.sparse=t.sparse,t.inputShape!=null&&t.batchInputShape!=null)throw new D("Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.");let e=t.batchInputShape;if(e==null){if(t.inputShape==null)throw new D("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");e=[t.batchSize].concat(t.inputShape)}else if(t.batchSize!=null)throw new D("Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.");const s=t.dtype||"float32";this.batchInputShape=e,this.dtype=s,this.inputSpec=[{shape:e}];const o=new Pn(this.dtype,this.batchInputShape,this,[],{},this.name);o.nodeIndex=0,o.tensorIndex=0,new Ol({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[o],outputTensors:[o],inputMasks:[null],outputMasks:[null],inputShapes:[e],outputShapes:[e]})}apply(t,e){throw new D(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}}Ii.className="InputLayer",Z(Ii);function Hg(n){if(n.batchShape==null&&n.shape==null)throw new Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(n.batchShape!=null&&n.shape!=null)throw new D("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let t=n.batchShape;n.shape!=null&&t==null&&(t=[null].concat(n.shape));let e=n.dtype;return e==null&&(e="float32"),new Ii({batchInputShape:t,name:n.name,dtype:e,sparse:n.sparse}).inboundNodes[0].outputTensors[0]}function pT(n,t){if(n.dtype==null||n.dtype===t.dtype)return t;try{return nt(t,n.dtype)}catch{throw new D(`The dtype of the feed (${t.dtype}) can not be cast to the dtype of the key '${n.name}' (${n.dtype}).`)}}class xs{constructor(t){if(this.id2Value={},this.id2Mask={},this.name2Id={},t instanceof xs)for(const e in t.id2Value)this.id2Value[e]=t.id2Value[e],e in t.id2Mask&&(this.id2Mask[e]=t.id2Mask[e]);else{if(t==null)return;for(const e of t)this.add(e.key,e.value)}}add(t,e,s){if(this.id2Value[t.id]==null)this.id2Value[t.id]=pT(t,e),this.name2Id[t.name]=t.id,s!=null&&(this.id2Mask[t.id]=s);else throw new D(`Duplicate key: name=${t.name}, id=${t.id}`);return this}addFeed(t){this.add(t.key,t.value)}hasKey(t){return this.id2Value[t.id]!=null}names(){return Object.keys(this.name2Id)}getValue(t){if(t instanceof Pn){if(this.id2Value[t.id]==null)throw new D(`Nonexistent key: ${t.name}`);return this.id2Value[t.id]}else{const e=this.name2Id[t];if(e==null)throw new D(`Feed dict has no SymbolicTensor name: ${t}`);return this.id2Value[e]}}getMask(t){if(t instanceof Pn){if(this.id2Value[t.id]==null)throw new D(`Nonexistent key: ${t.name}`);return this.id2Mask[t.id]}else{const e=this.name2Id[t];if(e==null)throw new D(`Feed dict has no SymbolicTensor name: ${t}`);return this.id2Mask[e]}}disposeMasks(){this.id2Mask!=null&&It(this.id2Mask)}}const _l=new Cg,Ll=new Cg;function fT(n){_l?.setMaxEntries(n),Ll?.setMaxEntries(n)}function $i(n,t,e,s){const o=e==null?!1:e.training,r=Array.isArray(n),i=r?n:[n],a=i.map(f=>f.name),l=[],c=t.names();for(const f of a)c.indexOf(f)!==-1?l.push(t.getValue(f)):l.push(null);const u=a.join(",")+"|"+t.names().sort().join(",");let h=_l.get(u),d;if(h==null){const f=mT(i,t);h=f.sorted,d=f.recipientCounts,_l.put(u,h),Ll.put(u,d)}d={},o||Object.assign(d,Ll.get(u));const p=new xs(t);for(let f=0;f<h.length;++f){const m=h[f],g=m.sourceLayer;if(g instanceof Ii)continue;const x=[],b=[],w=[];let y=!1;for(const k of m.inputs){const v=p.getValue(k),I=p.getMask(k);x.push(v),b.push(I),I!=null&&(y=!0),o||(d[k.name]--,d[k.name]===0&&!t.hasKey(k)&&a.indexOf(k.name)===-1&&!v.isDisposed&&k.sourceLayer.stateful!==!0&&w.push(v))}y&&(e=e||{},e.mask=b[0]);const C=Dt(g.apply(x,e));let $=null;g.supportsMasking&&($=g.computeMask(x,b));const N=xT(m),T=Array.isArray(N)?N:[N];for(let k=0;k<T.length;++k){p.hasKey(T[k])||p.add(T[k],C[k],Array.isArray($)?$[0]:$);const v=a.indexOf(T[k].name);v!==-1&&(l[v]=C[k])}o||It(w)}return p.disposeMasks(),r?l:l[0]}function mT(n,t){S(n!=null&&n.length>0,()=>"Expected at least one fetch, got none");let e=[],s={};if(n.length===1){const o=qg(n[0],t);e=o.sorted,s=o.recipientMap}else{const o=new Set;for(const r of n){const{sorted:i,recipientMap:a}=qg(r,t);for(const l of i)o.has(l.name)||(e.push(l),o.add(l.name));for(const l in a)s[l]==null&&(s[l]=new Set),a[l].forEach(c=>s[l].add(c))}}return{sorted:e,recipientCounts:gT(s)}}function gT(n){const t={};for(const e in n)t[e]=n[e].size;return t}function qg(n,t){const e=new Set,s=[],o={};for(const a of t.names())e.add(a);const r=[],i=[];for(r.push(n);r.length>0;){const a=r[r.length-1];if(e.has(a.name)){r.pop();continue}const l=i[i.length-1]===r.length-1;if(a.inputs.length===0||l)r.pop(),s.push(a),e.add(a.name),l&&i.pop();else{i.push(r.length-1);for(const c of a.inputs)o[c.name]==null&&(o[c.name]=new Set),o[c.name].add(a.name),!e.has(c.name)&&r.push(c)}}return{sorted:s,recipientMap:o}}function xT(n){let t;if(n.sourceLayer.inboundNodes.length===1)t=n.sourceLayer.output;else{let e=null;for(let s=0;s<n.sourceLayer.inboundNodes.length;++s)for(const o of n.sourceLayer.inboundNodes[s].outputTensors)if(o.id===n.id){e=s;break}t=n.sourceLayer.getOutputAt(e)}return t}U().registerFlag("TOPOLOGICAL_SORT_CACHE_MAX_ENTRIES",()=>100,fT);function pd(n,t){return V(()=>ke(ut(F(n,n),t,!0)))}class ki extends _o{getConfig(){return{}}}class Kg extends ki{constructor(t){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=t.maxValue!=null?t.maxValue:this.defaultMaxValue,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return V(()=>{const e=pd(t,this.axis),s=Xe(e,0,this.maxValue);return F(t,dt(s,J(ie(),e)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}}Kg.className="MaxNorm",Z(Kg);class jg extends ki{constructor(t){super(),this.defaultAxis=0,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return V(()=>dt(t,J(ie(),pd(t,this.axis))))}getConfig(){return{axis:this.axis}}}jg.className="UnitNorm",Z(jg);class Xg extends ki{apply(t){return Hs(t)}}Xg.className="NonNeg",Z(Xg);class Yg extends ki{constructor(t){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=t.minValue!=null?t.minValue:this.defaultMinValue,this.maxValue=t.maxValue!=null?t.maxValue:this.defaultMaxValue,this.rate=t.rate!=null?t.rate:this.defaultRate,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return V(()=>{const e=pd(t,this.axis),s=J(F(this.rate,Xe(e,this.minValue,this.maxValue)),F(1-this.rate,e));return F(t,dt(s,J(ie(),e)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}}Yg.className="MinMaxNorm",Z(Yg);const Zg={maxNorm:"MaxNorm",minMaxNorm:"MinMaxNorm",nonNeg:"NonNeg",unitNorm:"UnitNorm"};function le(n){return Xh(n)}function Jg(n,t={}){return xi(n,on.getMap().classNameMap,t,"constraint")}function ce(n){if(n==null)return null;if(typeof n=="string"){const e={className:n in Zg?Zg[n]:n,config:{}};return Jg(e)}else return n instanceof ki?n:Jg(n)}async function eo(n){if(n==null)return;const t=[],e=[],s=[];for(const o in n){const r=n[o];if(typeof r!="number"){const i=r;t.push(i.data()),e.push(o),s.push(i)}}if(t.length>0){const o=await Promise.all(t);for(let r=0;r<o.length;++r)n[e[r]]=o[r][0];It(s)}}function Qg(n){if(n!=null)for(const t in n){const e=n[t];typeof e!="number"&&e.dispose()}}var tx;(function(n){n[n.SILENT=0]="SILENT",n[n.VERBOSE=1]="VERBOSE"})(tx||(tx={}));const bT=125;class vi{constructor(){this.validationData=null}setParams(t){this.params=t}async onEpochBegin(t,e){}async onEpochEnd(t,e){}async onBatchBegin(t,e){}async onBatchEnd(t,e){}async onTrainBegin(t){}async onTrainEnd(t){}setModel(t){}}class yT{constructor(t,e=10){t==null&&(t=[]),this.callbacks=t,this.queueLength=e}append(t){this.callbacks.push(t)}setParams(t){for(const e of this.callbacks)e.setParams(t)}setModel(t){for(const e of this.callbacks)e.setModel(t)}async onEpochBegin(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onEpochBegin(t,e)}async onEpochEnd(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onEpochEnd(t,e)}async onBatchBegin(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onBatchBegin(t,e)}async onBatchEnd(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onBatchEnd(t,e)}async onTrainBegin(t){t==null&&(t={});for(const e of this.callbacks)await e.onTrainBegin(t)}async onTrainEnd(t){t==null&&(t={});for(const e of this.callbacks)await e.onTrainEnd(t)}}class wT extends vi{constructor(){super()}async onEpochBegin(t){this.seen=0,this.totals={}}async onBatchEnd(t,e){e==null&&(e={});const s=e.size==null?0:e.size;this.seen+=s;for(const o in e){const r=e[o];if(typeof r=="number")this.totals.hasOwnProperty(o)||(this.totals[o]=0),this.totals[o]=this.totals[o]+r*s;else{let i;o in this.totals?i=this.totals[o]:this.totals[o]=0;const a=V(()=>J(this.totals[o],F(r,s)));this.totals[o]=a,i?.dispose()}}}async onEpochEnd(t,e){if(e!=null)for(const s of this.params.metrics)this.totals[s]!=null&&(typeof this.totals[s]=="number"?e[s]=this.totals[s]/this.seen:V(()=>{const o=F(dt(1,this.seen),this.totals[s]);e[s]=o,this.totals[s].dispose(),Tn(e[s])}))}}class CT extends vi{async onTrainBegin(t){this.epoch=[],this.history={}}async onEpochEnd(t,e){e==null&&(e={}),this.epoch.push(t);for(const s in e)this.history[s]==null&&(this.history[s]=[]),this.history[s].push(e[s])}async syncData(){const t=[],e=[],s=[];for(const r in this.history){const i=this.history[r];for(let a=0;a<i.length;++a)if(typeof i[a]!="number"){const l=i[a];t.push(l.data()),e.push(r),s.push(a)}}const o=await Promise.all(t);for(let r=0;r<o.length;++r)this.history[e[r]][s[r]].dispose(),this.history[e[r]][s[r]]=o[r][0]}}class IT extends vi{constructor(t,e){if(super(),this.currentEpoch=0,this.nowFunc=t.nowFunc,this.nextFrameFunc=t.nextFrameFunc||zm,this.yieldEvery=e||"auto",this.yieldEvery==="auto"&&(this.yieldEvery=bT),this.yieldEvery==="never"&&t.onYield!=null)throw new Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");Tc(this.yieldEvery)&&(this.maybeWait=MN(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=t.onTrainBegin,this.trainEnd=t.onTrainEnd,this.epochBegin=t.onEpochBegin,this.epochEnd=t.onEpochEnd,this.batchBegin=t.onBatchBegin,this.batchEnd=t.onBatchEnd,this.yield=t.onYield}async maybeWait(t,e,s){const o=[];this.yield!=null&&(await eo(s),o.push(this.yield(t,e,s))),o.push(this.nextFrameFunc()),await Promise.all(o)}async onEpochBegin(t,e){this.currentEpoch=t,this.epochBegin!=null&&(await eo(e),await this.epochBegin(t,e))}async onEpochEnd(t,e){const s=[];this.epochEnd!=null&&(await eo(e),s.push(this.epochEnd(t,e))),this.yieldEvery==="epoch"&&s.push(this.nextFrameFunc()),await Promise.all(s)}async onBatchBegin(t,e){this.batchBegin!=null&&(await eo(e),await this.batchBegin(t,e))}async onBatchEnd(t,e){const s=[];this.batchEnd!=null&&(await eo(e),s.push(this.batchEnd(t,e))),this.yieldEvery==="batch"?s.push(this.nextFrameFunc()):Tc(this.yieldEvery)&&s.push(this.maybeWait(this.currentEpoch,t,e)),await Promise.all(s)}async onTrainBegin(t){this.trainBegin!=null&&(await eo(t),await this.trainBegin(t))}async onTrainEnd(t){this.trainEnd!=null&&(await eo(t),await this.trainEnd(t))}}function ex(n,t){return n==null&&(n={}),n instanceof vi?[n]:Array.isArray(n)&&n[0]instanceof vi?n:Dt(n).map(s=>new IT(s,t))}class cn{constructor(){}static registerCallbackConstructor(t,e){S(t>=0&&Number.isInteger(t),()=>`Verbosity level is expected to be an integer >= 0, but got ${t}`),cn.checkForDuplicate(e),cn.constructors[t]==null&&(cn.constructors[t]=[]),cn.constructors[t].push(e)}static checkForDuplicate(t){for(const e in cn.constructors)cn.constructors[+e].forEach(o=>{if(o===t)throw new D("Duplicate callback constructor.")})}static clear(){cn.constructors={}}static createCallbacks(t){const e=[];for(const s in cn.constructors){const o=+s;t>=o&&e.push(...cn.constructors[o])}return e.map(s=>new s)}}cn.constructors={};function nx(n,t,e,s,o,r,i,a,l){const c=new CT,u=[new wT,...cn.createCallbacks(t)];n!=null&&u.push(...n),u.push(c);const h=new yT(u);return h.setParams({epochs:e,initialEpoch:s,samples:o,steps:r,batchSize:i,verbose:t,doValidation:a,metrics:l}),{callbackList:h,history:c}}function Qn(n,t={},e=!1){return xi(n,on.getMap().classNameMap,t,"layer",e)}function Ml(n,t){return V(()=>{n.dtype!=="float32"&&(n=nt(n,"float32"));const e=ut(wi(n),t,!0),s=il(e.shape,ie()),o=ke(hs(e,s));return dt(n,o)})}function Pl(n,t){return V(()=>se(wi(pt(t,n)),-1))}function fd(n,t){return V(()=>se(Ee(pt(t,n)),-1))}function md(n,t){return V(()=>{const e=pt(n,t),s=Xe(Ee(n),ie(),Number.MAX_VALUE),o=Ee(dt(e,s));return F(100,se(o,-1))})}function $T(n,t){return V(()=>{const e=Xe(t,ie(),Number.MAX_VALUE),s=Dn(J(1,e)),o=Xe(n,ie(),Number.MAX_VALUE),r=Dn(J(1,o));return se(wi(pt(s,r)),-1)})}function kT(n,t){return V(()=>{const e=hs(0,pt(1,F(n,t)));return se(wi(e),-1)})}function vT(n,t){return V(()=>{const e=hs(0,pt(1,F(n,t)));return se(e,-1)})}function ST(n,t){return V(()=>{const e=ut(F(n,t),-1),s=mn(F(pt(1,n),t),-1);return hs(0,J(1,pt(s,e)))})}function NT(n,t){return V(()=>{const e=Math.log(2),s=pt(t,n),o=pt(J(s,li(F(-2,s))),e);return se(o,-1)})}function Si(n,t,e=!1){return V(()=>{if(e)t=lh(t);else{const s=ut(t,t.shape.length-1,!0);t=dt(t,s)}return t=Xe(t,ie(),1-ie()),Qt(ut(F(nt(n,"float32"),Dn(t)),t.shape.length-1))})}function Bl(n,t,e=!1){return V(()=>{const s=nt(ul(XN(n)),"int32");t=Xe(t,ie(),1-ie());const o=t.shape,r=L(Yf(s,o[o.length-1]),o);return Si(r,t,e)})}function TT(n,t){if(!Et(n.shape,t.shape))throw new D(`logits and labels must have the same shape, but got shapes ${JSON.stringify(n.shape)} and ${JSON.stringify(t.shape)}`);return V(()=>{const e=Hs(t),s=Qt(Ee(t));return J(pt(e,F(t,n)),qf(An(s)))})}function zl(n,t){return V(()=>{let e;return e=Xe(t,ie(),1-ie()),e=Dn(dt(e,pt(1,e))),se(TT(n,e),-1)})}function ET(n,t){return V(()=>{const e=Xe(n,ie(),1),s=Xe(t,ie(),1);return ut(F(n,Dn(dt(e,s))),-1)})}function RT(n,t){return V(()=>{const e=Dn(J(ie(),t));return se(pt(t,F(n,e)),-1)})}function sx(n,t){return V(()=>{const e=Ml(n,-1),s=Ml(t,-1),o=F(e,s);return Qt(ut(o,-1))})}const Vl={meanSquaredError:Pl,meanAbsoluteError:fd,meanAbsolutePercentageError:md,meanSquaredLogarithmicError:$T,squaredHinge:kT,hinge:vT,categoricalHinge:ST,logcosh:NT,categoricalCrossentropy:Si,sparseCategoricalCrossentropy:Bl,binaryCrossentropy:zl,kullbackLeiblerDivergence:ET,poisson:RT,cosineProximity:sx};function gd(n){if(typeof n=="string"){if(n in Vl)return Vl[n];let t=`Unknown loss ${n}`;throw n.toLowerCase().includes("softmaxcrossentropy")&&(t=`Unknown loss ${n}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new D(t)}else return n}function ox(n,t){return V(()=>{const e=F(.5,sn(t)),s=Ln(Ye(t,e),n.dtype);return se(Rn(n,s),-1)})}function rx(n,t){return V(()=>Ln(Rn(ni(n,-1),ni(t,-1)),"float32"))}function AT(n,t){return V(()=>nt(ut(Kn(Rn(n,1),Rn(t,1))),"float32"))}function DT(n,t){return V(()=>nt(ut(Kn(Rn(n,0),Rn(t,1))),"float32"))}function FT(n,t){return V(()=>{const e=AT(n,t),s=DT(n,t),o=J(e,s);return nt(Re(Ye(o,0),dt(e,o),0),"float32")})}function OT(n,t){return zl(n,t)}function _T(n,t){return n.rank===t.rank&&(n=di(n,[n.rank-1])),t=ni(t,-1),t.dtype!==n.dtype&&(t=nt(t,n.dtype)),nt(Rn(n,t),"float32")}const LT=Pl,MT=Pl,PT=fd,BT=fd,zT=md,VT=md,ix=Si,WT=sx,ax=Bl,Wl={binaryAccuracy:ox,categoricalAccuracy:rx,precision:FT,categoricalCrossentropy:ix,sparseCategoricalCrossentropy:ax,mse:LT,MSE:MT,mae:PT,MAE:BT,mape:zT,MAPE:VT,cosine:WT};function UT(n){if(typeof n=="string"&&n in Wl)return Wl[n];if(typeof n!="string"&&n!=null)return n;throw new D(`Unknown metric ${n}`)}function Ul(n){if(_n(n!==null,`Unknown LossOrMetricFn ${n}`),typeof n=="string")return n;{let t;for(const e of Object.keys(Vl))if(Vl[e]===n){t=e;break}if(t!==void 0)return t;for(const e of Object.keys(Wl))if(Wl[e]===n){t=e;break}return t!==void 0?t:n.name}}function GT(n){const t={Adagrad:()=>Xs.adagrad(.01),Adadelta:()=>Xs.adadelta(1,.95,ie()),Adam:()=>Xs.adam(.001,.9,.999,ie()),Adamax:()=>Xs.adamax(.002,.9,.999,ie(),0),RMSProp:()=>Xs.rmsprop(.001,.9,0,ie()),SGD:()=>Xs.sgd(.01)};if(t.adagrad=t.Adagrad,t.adadelta=t.Adadelta,t.adam=t.Adam,t.adamax=t.Adamax,t.rmsprop=t.RMSProp,t.sgd=t.SGD,n in t)return t[n]();throw new D(`Unknown Optimizer ${n}`)}const lx=1*1024*1024;function cx(n,t,e=!1){if(n==null||typeof n!="object"||Object.getPrototypeOf(n)!==Object.prototype||!xd(n))throw new Error("User-defined metadata is expected to be a JSON object, but is not.");if(e){const s=JSON.stringify(n);s.length>lx&&console.warn(`User-defined metadata of model "${t}" is too large in size (length=${s.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= ${lx}.`)}}function xd(n){if(n===null)return!0;if(typeof n=="object")if(Object.getPrototypeOf(n)===Object.prototype){const t=Object.keys(n);for(const e of t)if(typeof e!="string"||!xd(n[e]))return!1;return!0}else if(Array.isArray(n)){for(const t of n)if(!xd(t))return!1;return!0}else return!1;else{const t=typeof n;return t==="string"||t==="number"||t==="boolean"}}function HT(n,t,e,s=console.log){const o=KT(n),r=["Layer (type)","Input Shape","Output shape","Param #"];o?(t=t||90,e=e||[.32,.61,.89,1]):(t=t||115,e=e||[.24,.48,.7,.8,1]),e[e.length-1]<=1&&(e=e.map(u=>Math.floor(t*u)));let i;if(!o){r.push("Receives inputs"),i=[];for(const u in n.nodesByDepth)i.push(...n.nodesByDepth[u])}s("_".repeat(t)),Gl(r,e,s),s("=".repeat(t));const a=n.layers;for(let u=0;u<a.length;++u)o?jT(a[u],e,s):XT(a[u],e,i,s),s((u===a.length-1?"=":"_").repeat(t));n.checkTrainableWeightsConsistency();const l=qT(n),c=Fl(n.nonTrainableWeights);s(`Total params: ${l+c}`),s(`Trainable params: ${l}`),s(`Non-trainable params: ${c}`),s("_".repeat(t))}function qT(n){let t;return n.collectedTrainableWeights!=null?t=Fl(n.collectedTrainableWeights):t=Fl(n.trainableWeights),t}function KT(n){let t=!0;const e=[],s=[];for(const o in n.nodesByDepth)e.push(n.nodesByDepth[o]);for(const o of e){if(o.length>1||o.length===1&&o[0].inboundLayers.length>1){t=!1;break}s.push(...o)}if(t)for(const o of n.layers){let r=!1;for(const i of o.inboundNodes)if(s.indexOf(i)!==-1)if(r){t=!1;break}else r=!0;if(!t)break}return t}function Gl(n,t,e=console.log){let s="";for(let o=0;o<n.length;++o)o>0&&(s=s.slice(0,s.length-1)+" "),s+=n[o],s=s.slice(0,t[o]),s+=" ".repeat(t[o]-s.length);e(s)}function jT(n,t,e){let s,o;try{o=n.inboundNodes.map(l=>JSON.stringify(l.inputShapes)).join(",")}catch{o="multiple"}try{s=JSON.stringify(n.outputShape)}catch{s="multiple"}const r=n.name,i=n.getClassName(),a=[`${r} (${i})`,o,s,n.countParams().toString()];Gl(a,t,e)}function XT(n,t,e,s){let o,r;try{r=n.inboundNodes.map(h=>JSON.stringify(h.inputShapes)).join(",")}catch{r="multiple"}try{o=JSON.stringify(n.outputShape)}catch{o="multiple"}const i=[];for(const h of n.inboundNodes)if(!(e!=null&&e.length>0&&e.indexOf(h)===-1))for(let d=0;d<h.inboundLayers.length;++d){const p=h.inboundLayers[d].name,f=h.nodeIndices[d],m=h.tensorIndices[d];i.push(`${p}[${f}][${m}]`)}const a=n.name,l=n.getClassName(),c=i.length===0?"":i[0],u=[`${a} (${l})`,r,o,n.countParams().toString(),c];Gl(u,t,s);for(let h=1;h<i.length;++h)Gl(["","","","",i[h]],t,s)}function ux(n,t,e){return(n==="inboundNodes"||n==="outputLayers"||n==="inputLayers")&&t===0&&typeof e=="string"}function bd(n,t){if(n===null)return null;if(typeof n=="string")return Zs(n);if(typeof n=="number"||typeof n=="boolean")return n;if(n instanceof Array){const e=[],s=n.length;for(let o=0;o<s;++o){const r=n[o];ux(t,o,r)?e.push(r):e.push(bd(r,t))}return e}else{const e={};for(const s of Object.keys(n)){const o=n[s];if(s==="name"&&typeof o=="string")e[s]=o;else{const r=Zs(s);e[r]=bd(o,r)}}return e}}function yd(n,t){if(n==null)return null;if(typeof n=="string")return Jn(n);if(typeof n=="number"||typeof n=="boolean")return n;if(n instanceof Array){const e=[],s=n.length;for(let o=0;o<s;++o){const r=n[o];ux(t,o,r)?e.push(r):e.push(yd(r,t))}return e}else{const e={};for(const s of Object.keys(n)){const o=n[s],r=Jn(s);(s==="name"||s==="className")&&typeof o=="string"?e[r]=o:e[r]=yd(o,s)}return e}}const hx="4.22.0";const YT=n=>{const t=Object.keys(n);if(t.length===0)return!1;const e=t[0].split("/");return!isNaN(parseInt(e[e.length-1],10))};class Cn extends Ct{constructor(t){if(super({}),this.containerNodes=new Set,this.name=t.name,this.name==null){const b=this.getClassName().toLowerCase();this.name=El(b)}if(this.supportsMasking=!1,this.trainable_=!0,Array.isArray(t.inputs)?this.inputs=t.inputs.slice():this.inputs=[t.inputs],Array.isArray(t.outputs)?this.outputs=t.outputs.slice():this.outputs=[t.outputs],fs(this.inputs).length!==this.inputs.length)throw new D(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(b=>b.name)}`);fs(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(b=>b.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[];for(const b of this.outputs){const w=b.sourceLayer,y=b.nodeIndex,C=b.tensorIndex;this.outputLayers.push(w),this.outputLayersNodeIndices.push(y),this.outputLayersTensorIndices.push(C)}for(const b of this.inputs){const w=b.sourceLayer,y=b.nodeIndex,C=b.tensorIndex;_n(y===0,"input layer has >1 nodes"),_n(C===0,"input layer has >1 tensors"),this.inputLayers.push(w),this.inputLayersNodeIndices.push(y),this.inputLayersTensorIndices.push(C)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let b=0;b<this.inputLayers.length;b++){const w=this.inputLayers[b];if(!(w instanceof Ii))throw new TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${t.inputs}. Input ${b} (0-based) originates from layer type ${w.getClassName()}.`);this.inputNames.push(w.name),this.feedInputShapes.push(w.batchInputShape),this.feedInputNames.push(w.name)}for(const b of this.outputLayers)this.outputNames.push(b.name);this.internalInputShapes=this.inputs.map(b=>b.shape),this.internalOutputShapes=this.outputs.map(b=>b.shape);const e={},s={},o={},r={},i={},a=[],l=(b,w,y,C,$,N)=>{(C==null||$==null||N==null)&&(C=b.sourceLayer,$=b.nodeIndex,N=b.tensorIndex);const T=C.inboundNodes[$];if(y.indexOf(T)!==-1)throw new rn(`The tensor ${b.name} at layer "${C.name}" is part of a cycle.`);if(w.indexOf(T)!==-1)return;this.containerNodes.add(Cn.nodeKey(C,$)),C.id in i||(i[C.id]=Object.keys(i).length),y.indexOf(T)===-1&&y.push(T);const k=T.inboundLayers.length;for(let v=0;v<k;v++){const I=T.inputTensors[v],R=T.inboundLayers[v],A=T.nodeIndices[v],P=T.tensorIndices[v];l(I,w,y,R,A,P)}for(w.push(T);y.indexOf(T)>=0;)y.splice(y.indexOf(T),1);a.push(T)},c=[],u=[];for(const b of this.outputs)l(b,c,u);const h=a.slice().reverse();for(const b of h){s[b.id]=b,b.id in e||(e[b.id]=0);let w=e[b.id];const y=o[b.outboundLayer.id]==null?0:o[b.outboundLayer.id];w=Math.max(w,y),o[b.outboundLayer.id]=w,r[b.outboundLayer.id]=b.outboundLayer,e[b.id]=w;for(let C=0;C<b.inboundLayers.length;C++){const $=b.inboundLayers[C],N=b.nodeIndices[C],T=$.inboundNodes[N],k=e[T.id]==null?0:e[T.id];e[T.id]=Math.max(w+1,k),s[T.id]=T}}const d={};for(const b in e){const w=e[b];w in d||(d[w]=[]),d[w].push(s[b])}const p={};for(const b in o){const w=o[b];w in p||(p[w]=[]),p[w].push(r[b])}let f=Object.keys(p).map(b=>parseInt(b,10)).sort(Nl);this.layers=[];for(const b of f){const w=p[b];w.sort((y,C)=>{const $=i[y.id],N=i[C.id];return $<N?-1:$>N?1:0});for(const y of w)y instanceof Cn&&this.internalContainerRefs.push(y),this.layers.push(y)}this.layersByDepth=p,f=Object.keys(d).map(b=>parseInt(b,10)).sort(Nl);const m=this.inputs.slice(),g=[];for(const b of f)for(const w of d[b]){const y=w.outboundLayer;if(y!=null){for(const C of w.inputTensors)if(m.indexOf(C)===-1)throw new rn(`Graph disconnected: cannot obtain value for tensor ${C} at layer "${y.name}". The following previous layers were accessed without issue: ${g}`);for(const C of w.outputTensors)m.push(C);g.push(y.name)}}this.nodesByDepth=d;const x=this.layers.map(b=>b.name);for(const b of x){const w=x.filter(y=>y===b).length;if(w!==1)throw new rn(`The name "${b}" is used ${w} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(x))}this.outboundNodes=[],this.inboundNodes=[],new Ol({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(b=>null),outputMasks:this.outputs.map(b=>null),inputShapes:this.inputs.map(b=>b.shape),outputShapes:this.outputs.map(b=>b.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(this._refCount===0)throw new Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();const t={refCountAfterDispose:null,numDisposedVariables:0};if(--this._refCount===0){for(const e of this.layers)t.numDisposedVariables+=e.dispose().numDisposedVariables;for(const e of this.internalContainerRefs)t.numDisposedVariables+=e.dispose().numDisposedVariables}return t.refCountAfterDispose=this._refCount,t}get trainable(){return this.trainable_}set trainable(t){this.layers.forEach(e=>{e._trainableWeights.forEach(s=>s.trainable=t)}),this.trainable_=t}get trainableWeights(){if(this._trainableWeights.length>0)throw new D("Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.");if(!this.trainable)return[];let t=[];for(const e of this.layers)t=t.concat(e.trainableWeights);return t}get nonTrainableWeights(){const t=[];for(const e of this.layers)t.push(...e.nonTrainableWeights);if(!this.trainable){const e=[];for(const s of this.layers)e.push(...s.trainableWeights);return e.concat(t)}return t}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(t,e=!0){const s={};let o=0;const r=YT(t);r&&this.parseWeights(t);for(const a of this.layers)for(const[l,c]of a.weights.entries()){const u=r?`${c.name.split("/").slice(0,-1).join("/")+"/"}${l}`:c.originalName;if(s[u]!=null)throw new D(`Duplicate weight name: ${u}`);s[u]=c,o++}const i=[];for(const a in t){let l=a;if(s[a]==null){const c=a.split("/");l=c.slice(0,-2).concat([c[c.length-1]]).join("/")}if(s[l]!=null)i.push([s[l],t[a]]);else if(e)throw new D(`Provided weight data has no target variable: ${a}`);delete s[l]}if(e){const a=[];for(const l in s)a.push(l);if(a.length>0)throw new D(`${a.length} of ${o} weights are not set: ${a}`)}dd(i)}parseWeights(t){for(const e in Object.keys(t)){const s=e.split("/"),o=["vars","layer_checkpoint_dependencies"],r=s.map(i=>i.startsWith("_")?i.slice(1):i).filter(i=>!o.includes(i)).join("/");r!==e&&(t[r]=t[e],delete t[e])}}updatedConfig(){const t=this.getConfig(),e={};return e.className=this.getClassName(),e.config=t,e.kerasVersion=`tfjs-layers ${hx}`,e.backend="TensorFlow.js",e}toJSON(t,e=!0){const s=yd(this.updatedConfig());return e?JSON.stringify(s):s}call(t,e){return V(()=>{t=Dt(t);const s=new xs;for(let o=0;o<this.inputs.length;++o)s.add(this.inputs[o],t[o]);return $i(this.outputs,s,e)})}computeMask(t,e){return V(()=>{t=Dt(t);let s;return e==null?s=Ys(null,t.length):s=Dt(e),this.runInternalGraph(t,s)[1]})}computeOutputShape(t){const e=Dl(t);if(e.length!==this.inputLayers.length)throw new D(`Invalid inputShape argument ${t}: model has ${this.inputLayers.length} tensor inputs.`);const s={};for(let a=0;a<e.length;a++){const l=this.inputLayers[a],c=e[a],u=l.name+"_0_0";s[u]=c}const o=Object.keys(this.nodesByDepth).map(a=>parseInt(a,10)).sort(Nl);if(o.length>1)for(const a of o){const l=this.nodesByDepth[a];for(const c of l){const u=c.outboundLayer;if(this.inputLayers.map(m=>m.id).indexOf(u.id)!==-1)continue;const h=[];for(let m=0;m<c.inboundLayers.length;m++){const g=c.inboundLayers[m],x=c.nodeIndices[m],b=c.tensorIndices[m],w=`${g.name}_${x}_${b}`,y=s[w];h.push(y)}const d=u.computeOutputShape(Be(h)),p=Dl(d),f=u.inboundNodes.indexOf(c);for(let m=0;m<p.length;m++){const g=`${u.name}_${f}_${m}`;s[g]=p[m]}}}const r=[],i=[];for(let a=0;a<this.outputLayers.length;a++){const l=this.outputLayers[a],c=this.outputLayersNodeIndices[a],u=this.outputLayersTensorIndices[a],h=`${l.name}_${c}_${u}`;i.push(h)}for(let a=0;a<i.length;a++){const l=i[a];_n(l in s),r.push(s[l])}return Be(r)}runInternalGraph(t,e){e==null&&(e=Ys(null,t.length));const s={};for(let l=0;l<this.inputs.length;++l){const c=this.inputs[l],u=t[l],h=e[l];s[c.id]=[u,h]}const o=Object.keys(this.nodesByDepth).map(l=>parseInt(l,10)).sort(Nl);for(const l of o){const c=this.nodesByDepth[l];for(const u of c){const h=u.outboundLayer,d=u.inputTensors,p=u.outputTensors,f=new Array;for(const m of d)m.id in s&&f.push(s[m.id]);if(f.length===d.length){let m={},g,x,b,w;if(u.callArgs!=null&&(m=u.callArgs),f.length===1){const[y,C]=f[0];m.mask==null&&(m.mask=C),b=Dt(h.call(y,m)),w=Dt(h.computeMask(y,C)),g=[y],x=[C]}else g=f.map(y=>y[0]),x=f.map(y=>y[1]),m.mask==null&&(m.mask=x),b=Dt(h.call(g,m)),w=Dt(h.computeMask(g,x));if(h.activityRegularizer)throw new bt("LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.");for(let y=0;y<p.length;++y){const C=p[y],$=b[y],N=w[y];s[C.id]=[$,N]}}}}const r=[],i=[],a=[];for(const l of this.outputs){_n(l.id in s,`Could not compute output ${l.name} : ${l.id}`);const[c,u]=s[l.id];a.push(c.shape),r.push(c),i.push(u)}return[r,i,a]}buildNodeConversionMap(t){const e={};let s;for(const o of this.layers){s=o instanceof Cn?1:0;for(let r=0;r<o.inboundNodes.length;r++){const i=Cn.nodeKey(o,r);this.containerNodes.has(i)&&(e[i]=s,s+=1)}}return e}getLayer(t,e){if(e!=null)return this.findLayer(e);if(t==null)throw new D("Provide either a layer name or layer index");if(typeof t=="number")return this.findLayer(t);for(const s of this.layers)if(s.name===t)return s;throw new D(`No such layer: ${t}`)}findLayer(t){if(this.layers.length<=t)throw new D(`Was asked to retrieve layer at index ${t}, but model only has ${this.layers.length} layer(s).`);return this.layers[t]}calculateLosses(){return V(()=>{const t=[];for(const e of this.layers)for(let s=0;s<e.inboundNodes.length;++s){const o=Cn.nodeKey(e,s);this.containerNodes.has(o)&&t.push(...e.calculateLosses())}return t})}getConfig(){const t={name:this.name},e=this.buildNodeConversionMap(this.layers),s=[];for(const i of this.layers){const a=i.getClassName(),l=i.getConfig(),c=[];for(let h=0;h<i.inboundNodes.length;h++){const d=i.inboundNodes[h],p=Cn.nodeKey(i,h);let f={};if(this.containerNodes.has(p)){if(d.callArgs)try{JSON.stringify(d.callArgs),f=d.callArgs}catch{console.warn(`Layer ${i.name} was passed non-serializable keyword arguments: ${d.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),f={}}if(d.inboundLayers.length>0){const m=[];for(let g=0;g<d.inboundLayers.length;g++){const x=d.inboundLayers[g],b=d.nodeIndices[g],w=d.tensorIndices[g],y=Cn.nodeKey(x,b);let C=e[y];C==null&&(C=0),m.push([x.name,C,w,f])}c.push(m)}}}const u={};u.name=i.name,u.className=a,u.config=l,u.inboundNodes=c,s.push(u)}t.layers=s;const o=[];for(let i=0;i<this.inputLayers.length;i++){const a=this.inputLayers[i],l=this.inputLayersNodeIndices[i],c=Cn.nodeKey(a,l);if(!this.containerNodes.has(c))continue;let u=e[c];u==null&&(u=0);const h=this.inputLayersTensorIndices[i];o.push([a.name,u,h])}t.inputLayers=o;const r=[];for(let i=0;i<this.outputLayers.length;i++){const a=this.outputLayers[i],l=this.outputLayersNodeIndices[i],c=Cn.nodeKey(a,l);if(!this.containerNodes.has(c))continue;let u=e[c];u==null&&(u=0);const h=this.outputLayersTensorIndices[i];r.push([a.name,u,h])}return t.outputLayers=r,t}static fromConfig(t,e,s={},o=!1){const r={},i={};function a(g,x){g.name in i?i[g.name].push(x):i[g.name]=[x]}function l(g,x){const b=[];let w;for(const y of x){const C=y[0],$=y[1],N=y[2];if(w=y[3]==null?{}:y[3],!(C in r)){a(g,x);return}const T=r[C];if(T.inboundNodes.length<=$){a(g,x);return}const k=T.inboundNodes[$];b.push(k.outputTensors[N])}b.length>0&&g.apply(Be(b),w)}function c(g){const x=g.name,b=Qn(g,e.customObjects!=null?e.customObjects:{});b.setFastWeightInitDuringBuild(o),r[x]=b,g.inboundNodes.forEach(y=>{if(!(y instanceof Array))throw new D(`Corrupted configuration, expected array for nodeData: ${y}`);a(b,y)})}const u=e.name,h=e.layers;for(const g of h)c(g);for(;!LN(i);)for(const g of h){const x=r[g.name];if(x.name in i){const b=i[x.name];delete i[x.name];for(const w of b)l(x,w)}}const d=[],p=[],f=e.inputLayers;for(const g of f){const x=g[0],b=g[1],w=g[2];_n(x in r);const C=r[x].inboundNodes[b].outputTensors;d.push(C[w])}const m=e.outputLayers;for(const g of m){const x=g[0],b=g[1],w=g[2];_n(x in r);const C=r[x].inboundNodes[b].outputTensors;p.push(C[w])}return new t({inputs:d,outputs:p,name:u})}get stateful(){if(this._stateful)throw new D("Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.");for(const t of this.layers)if(t.stateful)return!0;return!1}resetStates(){V(()=>{this.layers.forEach(t=>{t.stateful&&t.resetStates()})})}}function ZT(n,t,e){const s=t.length;if(n==null||Array.isArray(n)&&n.length===0)return t.map(o=>null);if(s===1)return Array.isArray(n)&&n.length===1?n:typeof n=="object"&&t[0]in n?[n[t[0]]]:[n];if(Array.isArray(n)){if(n.length!==s)throw new Error(`Provided ${e} is an array of ${n.length} element(s), but the model has ${s} outputs. Make sure a set of weights is provided for each model output.`);return n}else if(typeof n=="object"&&Object.keys(n).length>0&&typeof n[Object.keys(n)[0]]=="object"){const o=[];return t.forEach(r=>{r in n?o.push(n[r]):o.push(null)}),o}else throw new Error(`The model has multiple (${s}) outputs, so ${e} must be either an array with ${s} elements or an object with ${t} keys. Provided ${e} not understood: ${JSON.stringify(n)}`)}function dx(n,t){return ZT(n,t,"classWeight")}async function px(n,t,e,s){if(e!=null){const o=V(()=>{if(n.shape.length===1)return Bs(n);if(n.shape.length===2){if(n.shape[1]>1)return ni(n,1);if(n.shape[1]===1)return L(n,[n.shape[0]]);throw new Error(`Encountered unexpected last-dimension size (${n.shape[1]}) during handling of class weights. The size is expected to be >= 1.`)}else throw new Error(`Unexpected rank of target (y) tensor (${n.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),r=Array.from(await o.data());It(o);const i=[];return r.forEach(a=>{if(e[a]==null)throw new Error(`classWeight must contain all classes in the training data. The class ${a} exists in the data but not in classWeight`);i.push(e[a])}),Ge(i,"float32")}else return null}function JT(n,t){return F(n,t)}const QT=32;function fx(n,t){let e,s;const o=t;e=o.xs,s=o.ys,S(e!=null&&s!=null,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${t}`);const r=mx("input",n.inputNames,e),i=mx("output",n.outputNames,s),a=r[0].shape[0];S(r.length===n.inputs.length,()=>`LayersModel has ${n.inputs.length} inputs, but the dataset provides ${r.length} inputs.  (Expected input keys: ${JSON.stringify(n.inputNames)})`),S(i.length===n.outputs.length,()=>`LayersModel has ${n.outputs.length} outputs, but the dataset provides ${i.length} outputs.  (Expected output keys: ${JSON.stringify(n.outputNames)})`);for(let l=0;l<r.length;l++)S(r[l].shape[0]===a,()=>`Batch size mismatch: input ${n.inputNames[l]} has ${r[l].shape[0]}; expected  ${a} based on input ${n.inputNames[0]}.`);for(let l=0;l<i.length;l++)S(i[l].shape[0]===a,()=>`Batch size mismatch: output ${n.outputNames[l]} has ${i[l].shape[0]}; expected  ${a} based on input ${n.inputNames[0]}.`);return{xs:r,ys:i}}function mx(n,t,e){if(e instanceof oe)return[e];if(Array.isArray(e))return S(e.length===t.length,()=>`Received an array of ${e.length} Tensors, but expected ${t.length} to match the ${n} keys ${t}.`),e;{const s=[];for(const o of t){if(e[o]==null)throw new D(`The feature data generated by the dataset lacks the required ${n} key '${o}'.`);s.push(e[o])}return s}}function tE(n){if(n.length===3)throw new bt("Validation with sample weights is not implemented yet.");return{xs:n[0],ys:n[1]}}async function eE(n,t,e){const s=e.batchesPerEpoch!=null;if(S(n.optimizer!=null,()=>"You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig)."),S(e!=null,()=>"For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call."),S(e.epochs!=null&&e.epochs>0&&Number.isInteger(e.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${e.epochs}`),S(!s||e.batchesPerEpoch>0&&Number.isInteger(e.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${e.batchesPerEpoch}`),S(e.validationSplit==null,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),n.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");n.isTraining=!0;try{const o=e.validationData!=null;let r,i;if(o)if(gx(e.validationData))S(e.validationBatches==null||e.validationBatches>0&&Number.isInteger(e.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${e.validationBatches}`);else{const g=tE(e.validationData);r=g.xs,i=g.ys}const a=n.makeTrainFunction(),l=n.getDedupedMetricsNames();let c;o?c=l.slice().concat(l.map(g=>"val_"+g)):c=l.slice();const u=ex(e.callbacks,e.yieldEvery),h=e.verbose==null?1:e.verbose,{callbackList:d,history:p}=nx(u,h,e.epochs,null,null,nE(t,e),null,o,c);d.setModel(n),n.history=p,await d.onTrainBegin(),n.stopTraining_=!1;let f=e.initialEpoch==null?0:e.initialEpoch,m=await t.iterator();for(;f<e.epochs;){const g={};await d.onEpochBegin(f);let x=0,b=0;for(s||(m=await t.iterator());!s||x<e.batchesPerEpoch;){const w=await m.next();if(s&&w.done){console.warn(`You provided \`batchesPerEpoch\` as ${e.batchesPerEpoch}, but your dataset iterator ran out of data after ${x} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${e.batchesPerEpoch*e.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(w.value!=null){const{xs:y,ys:C}=fx(n,w.value),$={};$.batch=b,$.size=y[0].shape[0],await d.onBatchBegin(b,$);const N=[];if(e.classWeight!=null){const v=dx(e.classWeight,n.outputNames);for(let I=0;I<v.length;++I)N.push(await px(C[I],null,v[I]))}const T=y.concat(C).concat(N),k=a(T);It(T);for(let v=0;v<l.length;++v){const I=l[v],R=k[v];$[I]=R,Tn(R)}await d.onBatchEnd(b,$),Qg($),b++,x++}if(s?x>=e.batchesPerEpoch:w.done){if(o){let y;gx(e.validationData)?y=Dt(await n.evaluateDataset(e.validationData,{batches:e.validationBatches})):y=Dt(n.evaluate(r,i,{batchSize:e.validationBatchSize==null?QT:e.validationBatchSize,verbose:0}));for(let C=0;C<n.metricsNames.length;++C)g[`val_${n.metricsNames[C]}`]=y[C]}break}if(n.stopTraining_)break}if(await d.onEpochEnd(f,g),f++,n.stopTraining_)break}return await d.onTrainEnd(),await n.history.syncData(),n.history}finally{n.isTraining=!1}}function nE(n,t){let e=null;return t.batchesPerEpoch!=null?e=t.batchesPerEpoch:Number.isFinite(n.size)&&(e=n.size),e}function gx(n){return typeof n.iterator=="function"}function sE(n){return typeof n.next=="function"}async function oE(n,t,e){e=e||{};const s=e.batches!=null,o=n.testFunction;let r=[];if(e.verbose>0)throw new bt("Verbose mode is not implemented yet.");S(!s||e.batches>0&&Number.isInteger(e.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(e.batches)}`);const i=sE(t)?t:await t.iterator();let a=0,l=0;for(;!s||l<e.batches;){const c=await i.next();if(r=V(()=>{if(c.value){const{xs:u,ys:h}=fx(n,c.value),d=u.concat(h),p=V(()=>o(d));if(It(d),l===0)for(let m=0;m<p.length;++m)r.push(At(0));const f=d[0].shape[0];for(let m=0;m<p.length;++m){const g=p[m],x=r[m];r[m]=V(()=>J(r[m],F(f,g))),l>0&&It(x)}It(p),a+=f,++l}return r}),c.done){s&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${e.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let c=0;c<r.length;++c){const u=r[c];r[c]=dt(r[c],a),It(u)}return Be(r)}function wd(n){S(n>0&&Number.isInteger(n),()=>`batchSize is required to be a positive integer, but got ${n}`)}function Ni(n,t,e){return n==null?[null]:Array.isArray(n)?n.map(s=>to(s,t,e-t)):to(n,t,e-t)}function Cd(n,t){return V(()=>n==null?null:Array.isArray(n)?n.map(e=>Cd(e,t)):Dg(n,t.dtype==="int32"?t:nt(t,"int32")))}function Id(n,t){const e=[];let s=0,o=null;for(;s<n;)o=s+t,o>=n&&(o=n),e.push([s,o]),s=o;return e}function xx(n){const t=[];n instanceof oe&&(n=[n]);for(let e=0;e<n.length;++e){const s=n[e];if(s.rank===1)t.push(yi(s,1));else{if(s.rank===0)throw new Error("Expected tensor to be at least 1D, but received a 0D tensor (scalar).");t.push(s)}}return t}function In(n,t){if(n==null)return;const e=[];if(t instanceof oe)e.push(t.id);else if(Array.isArray(t))t.forEach(o=>e.push(o.id));else if(t!=null)for(const o in t){const r=t[o];e.push(r.id)}const s=[];if(n instanceof oe)e.indexOf(n.id)===-1&&s.push(n);else if(Array.isArray(n))n.forEach(o=>{e.indexOf(o.id)===-1&&s.push(o)});else if(n!=null)for(const o in n){const r=n[o];e.indexOf(r.id)===-1&&s.push(r)}s.forEach(o=>{o.isDisposed||o.dispose()})}function rE(n){return n instanceof oe}function $d(n){return Array.isArray(n)}function bx(n){return!rE(n)&&!$d(n)}function yx(n,t,e,s=!0,o=""){if(t==null||t.length===0){if(n!=null){let i=!1;if($d(n)&&n.length>0)i=!0;else if(bx(n)){for(const a in n)if(n.hasOwnProperty(a)){i=!0;break}}else i=!0;if(i)throw new D(`Error when checking model ${o} expected no data, but got ${n}`)}return[]}if(n==null)return t.map(i=>null);let r;if(bx(n)){n=n,r=[];for(const i of t){if(n[i]==null)throw new D(`No data provided for "${i}". Need data for each key in: ${t}`);r.push(n[i])}}else if($d(n)){if(n=n,n.length!==t.length)throw new D(`Error when checking model ${o}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${t.length} Tensor(s), but instead got the following list of Tensor(s): ${n}`);r=n}else{if(n=n,t.length>1)throw new D(`The model ${o} expects ${t.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${n.shape}`);r=[n]}if(r=xx(r),e!=null)for(let i=0;i<t.length;++i){if(e[i]==null)continue;const a=r[i];if(a.shape.length!==e[i].length)throw new D(`Error when checking ${o}: expected ${t[i]} to have ${e[i].length} dimension(s). but got array with shape ${a.shape}`);for(let l=0;l<e[i].length;++l){if(l===0&&!s)continue;const c=a.shape[l],u=e[i][l];if(u!=null&&u>=0&&c!==u)throw new D(`${o} expected a batch of elements where each example has shape [${e[i].slice(1,e[i].length)}] (i.e.,tensor shape [*,${e[i].slice(1,e[i].length)}]) but the ${o} received an input with ${a.shape[0]} examples, each with shape [${a.shape.slice(1,a.shape.length)}] (tensor shape [${a.shape}])`)}}return r}function iE(n,t,e){const s=fs(n.map(r=>r.shape[0]));s.sort();const o=fs(t.map(r=>r.shape[0]));if(o.sort(),s.length>1)throw new D(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(n.map(r=>r.shape))}`);if(o.length>1)throw new D(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(t.map(r=>r.shape))}`);if(s.length>0&&o.length>0&&!Et(s,o))throw new D(`Input Tensors should have the same number of samples as target Tensors. Found ${s[0]} input sample(s) and ${o[0]} target sample(s).`)}function aE(n,t,e){const s=[Pl,zl,Si];for(let o=0;o<n.length;++o){const r=n[o],i=t[o],a=e[o];if(i!=null){if(i===Si&&r.shape[r.shape.length-1]===1)throw new D(`You are passing a target array of shape ${r.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(s.indexOf(i)!==-1){const l=r.shape.slice(1),c=a.slice(1);for(let u=0;u<l.length;++u){const h=l[u],d=c[u];if(d!=null&&h!==d)throw new D(`A target Tensor with shape ${r.shape} was passed for an output of shape ${a}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}function wx(n,t,e,s=!0,o=""){let r;if(Array.isArray(n)){if(n.length!==t.length)throw new D(`Error when checking model ${o}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${t.length} Tensor(s), but instead got ${n.length} Tensors(s).`);r=n}else{if(t.length>1)throw new D(`The model expects ${t.length} ${o} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(n.shape)}.`);r=[n]}if(e!=null)for(let i=0;i<t.length;++i){if(e[i]==null)continue;const a=r[i];if(a.shape.length!==e[i].length)throw new D(`Error when checking ${o}: expected ${t[i]} to have ${e[i].length} dimension(s), but got array with shape ${JSON.stringify(a.shape)}`);for(let l=0;l<e[i].length;++l){if(l===0&&!s)continue;const c=a.shape[l],u=e[i][l];if(u!=null&&u!==c)throw new D(`Error when checking ${o}: expected ${t[i]} to have shape ${JSON.stringify(e[i])} but got array with shape ${JSON.stringify(a.shape)}.`)}}}function lE(n,t){if(n==null||Array.isArray(n)&&n.length===0)return t.map(s=>[]);let e;if(typeof n=="string"||typeof n=="function")e=[n];else if(Array.isArray(n)||typeof n=="object")e=n;else throw new TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${n}`);if(Array.isArray(e))return t.map(s=>e);{const s=[];for(const o of t){let r=e.hasOwnProperty(o)?e[o]:[];Array.isArray(r)||(r=[r]),s.push(r)}return s}}const cE="layers-model";class no extends Cn{constructor(t){super(t),this.isTraining=!1}summary(t,e,s=console.log){if(!this.built)throw new D("This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).");HT(this,t,e,s)}compile(t){if(t.loss==null&&(t.loss=[]),this.loss=t.loss,typeof t.optimizer=="string")this.optimizer_=GT(t.optimizer),this.isOptimizerOwned=!0;else{if(!(t.optimizer instanceof ps))throw new D("User-defined optimizer must be an instance of tf.Optimizer.");this.optimizer_=t.optimizer,this.isOptimizerOwned=!1}let e=[];if(!Array.isArray(t.loss)&&typeof t.loss!="string"&&typeof t.loss!="function"){t.loss=t.loss;for(const i in t.loss)if(this.outputNames.indexOf(i)===-1)throw new D(`Unknown entry in loss dictionary: "${i}". Only expected the following keys: ${this.outputNames}`);for(const i of this.outputNames)t.loss[i]==null&&console.warn(`Output "${i}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${i} during training`),e.push(gd(t.loss[i]))}else if(Array.isArray(t.loss)){if(t.loss.length!==this.outputs.length)throw new D(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${t.loss}.`);e=t.loss.map(a=>gd(a))}else{const i=gd(t.loss);this.outputs.forEach(a=>{e.push(i)})}this.lossFunctions=e,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let i=0;i<this.outputs.length;++i){const a=this.internalOutputShapes[i],l=this.outputNames[i];this.feedOutputNames.push(l),this.feedOutputShapes.push(a),this.feedLossFns.push(this.lossFunctions[i])}const s=[];this.metrics=t.metrics,this.metricsNames=["loss"],this.metricsTensors=[],Qs("loss",()=>{for(let i=0;i<this.outputs.length;++i){if(s.indexOf(i)!==-1)continue;const a=this.lossFunctions[i];this.outputs.length>1&&(this.metricsTensors.push([a,i]),this.metricsNames.push(this.outputNames[i]+"_loss"))}});const o=lE(t.metrics,this.outputNames),r=(i,a,l)=>{this.outputNames.length>1&&(a=this.outputNames[i]+"_"+a),this.metricsNames.push(a),this.metricsTensors.push([l,i])};Qs("metric",()=>{for(let i=0;i<this.outputs.length;++i){if(s.indexOf(i)!==-1)continue;const a=o[i];(c=>{let h,d,p;for(const f of c){if(typeof f=="string"&&["accuracy","acc","crossentropy","ce"].indexOf(f)!==-1){const g=this.internalOutputShapes[i];g[g.length-1]===1||this.lossFunctions[i]===zl?["accuracy","acc"].indexOf(f)!==-1?d=ox:["crossentropy","ce"].indexOf(f)!==-1&&(d=OT):this.lossFunctions[i]===Bl?["accuracy","acc"].indexOf(f)!==-1?d=_T:["crossentropy","ce"].indexOf(f)!==-1&&(d=ax):["accuracy","acc"].indexOf(f)!==-1?d=rx:["crossentropy","ce"].indexOf(f)!==-1&&(d=ix);let x;["accuracy","acc"].indexOf(f)!==-1?x="acc":["crossentropy","ce"].indexOf(f)!==-1&&(x="ce"),p=d,h=""+x}else p=UT(f),h=""+Ul(f);let m;Qs(h,()=>{m=p}),r(i,h,m)}})(a)}}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){this.collectedTrainableWeights!=null&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(t,e,s={}){const o=s.batchSize==null?32:s.batchSize;wd(o);const i=this.standardizeUserDataXY(t,e,!0,o);try{const a=i[0].concat(i[1]);this.makeTestFunction();const l=this.testFunction,c=this.testLoop(l,a,o,s.verbose,s.steps);return Be(c)}finally{In(i[0],t),In(i[1],e)}}async evaluateDataset(t,e){return this.makeTestFunction(),oE(this,t,e)}checkNumSamples(t,e,s,o="steps"){let r;if(s!=null){if(r=null,e!=null)throw new D(`If ${o} is set, batchSize must be null or undefined.Got batchSize = ${e}`)}else if(t!=null)Array.isArray(t)?r=t[0].shape[0]:r=t.shape[0];else throw new D(`Either the input data should have a defined shape, or ${o} shoud be specified.`);return r}execute(t,e){if(Array.isArray(e)&&e.length===0)throw new D("`outputs` is an empty Array, which is not allowed.");const s=Array.isArray(e),o=s?e:[e],r=this.retrieveSymbolicTensors(o),i=new xs;if(t instanceof oe&&(t=[t]),Array.isArray(t)){if(t.length!==this.inputs.length)throw new D(`The number of inputs provided (${t.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let l=0;l<this.inputs.length;++l)i.add(this.inputs[l],t[l])}else for(const l of this.inputs){const c=t[l.name];if(c==null)throw new D(`No value is provided for the model's input ${l.name}`);i.add(l,c)}const a=$i(r,i);return s?a:a[0]}retrieveSymbolicTensors(t){const e=Ys(null,t.length);let s=t.length;for(const o of this.layers){const r=Array.isArray(o.output)?o.output:[o.output],i=r.map(a=>a.name);for(let a=0;a<t.length;++a){const l=i.indexOf(t[a]);if(l!==-1&&(e[a]=r[l],s--),s===0)break}if(s===0)break}if(s>0){const o=[];throw e.forEach((r,i)=>{r==null&&o.push(t[i])}),new D(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(o)}`)}return e}predictLoop(t,e=32,s=!1){return V(()=>{const o=this.checkNumSamples(t);if(s)throw new bt("Verbose predictLoop() is not implemented yet.");const r=Id(o,e),i=this.outputs.map(a=>[]);for(let a=0;a<r.length;++a)V(()=>{const c=r[a][0],u=r[a][1],h=Ni(t,c,u),d=[];if(Array.isArray(h))for(let f=0;f<h.length;++f)d.push({key:this.inputs[f],value:h[f]});else d.push({key:this.inputs[0],value:h});const p=new xs(d);return $i(this.outputs,p)}).forEach((c,u)=>i[u].push(c));return Be(i.map(a=>Me(a,0)))})}predict(t,e={}){const s=xx(t);wx(s,this.inputNames,this.feedInputShapes,!1);try{const o=e.batchSize==null?32:e.batchSize;return wd(o),this.predictLoop(s,o)}finally{In(s,t)}}predictOnBatch(t){wx(t,this.inputNames,this.feedInputShapes,!0);const e=(Array.isArray(t)?t[0]:t).shape[0];return this.predictLoop(t,e)}standardizeUserDataXY(t,e,s=!0,o){if(this.optimizer_==null)throw new rn("You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).");const r=[];for(let i=0;i<this.feedOutputShapes.length;++i){const a=this.feedOutputShapes[i];this.feedLossFns[i]===Bl?r.push(a.slice(0,a.length-1).concat([1])):r.push(a)}if(t=yx(t,this.feedInputNames,this.feedInputShapes,!1,"input"),e=yx(e,this.feedOutputNames,r,!1,"target"),iE(t,e),aE(e,this.feedLossFns,this.feedOutputShapes),this.stateful&&o!=null&&o>0&&t[0].shape[0]%o!==0)throw new D(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${o}. Found: ${t[0].shape[0]} sample(s).`);return[t,e]}async standardizeUserData(t,e,s,o,r=!0,i){const[a,l]=this.standardizeUserDataXY(t,e,r,i);if(s!=null)throw new Error("sample weight is not supported yet.");let c=null;if(o!=null){const u=dx(o,this.outputNames);c=[];for(let h=0;h<u.length;++h)c.push(await px(l[h],null,u[h]))}return[a,l,c]}testLoop(t,e,s,o=0,r){return V(()=>{const i=this.checkNumSamples(e,s,r,"steps"),a=[];if(o>0)throw new bt("Verbose mode is not implemented yet.");if(r!=null)throw new bt("steps mode in testLoop() is not implemented yet");{const l=Id(i,s),c=Ge(bn(0,i));for(let u=0;u<l.length;++u){const h=l[u][0],d=l[u][1],p=to(c,h,d-h),f=Cd(e,p),m=t(f);if(u===0)for(let g=0;g<m.length;++g)a.push(At(0));for(let g=0;g<m.length;++g){const x=m[g];a[g]=J(a[g],F(d-h,x))}}for(let u=0;u<a.length;++u)a[u]=dt(a[u],i)}return a})}getDedupedMetricsNames(){const t=this.metricsNames,e=[];for(let s=0;s<t.length;++s){const o=t[s];let r=o;if(Ig(t,o)>1){const i=Ig(t.slice(0,s),o);r+=`_${i}`}e.push(r)}return e}makeTrainFunction(){return t=>{const e=[],s=t.slice(0,this.inputs.length),o=t.slice(this.inputs.length,this.inputs.length+this.outputs.length),r=t.slice(this.inputs.length+this.outputs.length,this.inputs.length+this.outputs.length*2),i=[],a=()=>{const h=[];for(let m=0;m<this.inputs.length;++m)h.push({key:this.inputs[m],value:s[m]});const d=new xs(h),p=$i(this.outputs,d,{training:!0});let f;for(let m=0;m<this.lossFunctions.length;++m){const g=this.lossFunctions[m];let x=g(o[m],p[m]);r[m]!=null&&(x=JT(x,r[m]));const b=se(x);e.push(b),m===0?f=x:f=J(f,x)}for(let m=0;m<this.metricsTensors.length;++m){let g;if(this.outputs.length>1&&m<this.outputs.length)g=e[m];else{const x=this.metricsTensors[m][0],b=this.metricsTensors[m][1];g=se(x(o[b],p[b]))}Tn(g),i.push(g)}return f=se(f),this.calculateLosses().forEach(m=>{f=J(f,m)}),f},l=this.collectedTrainableWeights.map(h=>h.read());return[this.optimizer_.minimize(a,!0,l)].concat(i)}}makeTestFunction(){this.testFunction=t=>V(()=>{const e=[];let s;const o=t.slice(0,this.inputs.length),r=t.slice(this.inputs.length,this.inputs.length+this.outputs.length),i=[];for(let c=0;c<this.inputs.length;++c)i.push({key:this.inputs[c],value:o[c]});const a=new xs(i),l=$i(this.outputs,a);for(let c=0;c<this.lossFunctions.length;++c){const u=this.lossFunctions[c],h=se(u(r[c],l[c]));c===0?s=h:s=J(s,h),e.push(s)}for(let c=0;c<this.metricsTensors.length;++c){const u=this.metricsTensors[c][0],h=this.metricsTensors[c][1],d=se(u(r[h],l[h]));e.push(d)}return e})}async fit(t,e,s={}){if(this.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");this.isTraining=!0;let o,r,i,a,l,c,u,h,d;try{const p=s.batchSize==null?32:s.batchSize;wd(p);const m=await this.standardizeUserData(t,e,s.sampleWeight,s.classWeight,!1,p);o=m[0],r=m[1],d=m[2];let g=!1,x;if(s.validationData!=null&&s.validationData.length>0){if(g=!0,s.validationData.length===2)l=s.validationData[0],c=s.validationData[1];else throw s.validationData.length===3?new bt("validationData including sample weights is not supported yet."):new D(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${s.validationData} is invalid.`);const v=await this.standardizeUserData(l,c,null,null,!0,p);u=v[0],h=v[1],x=u.concat(h)}else if(s.validationSplit!=null&&s.validationSplit>0&&s.validationSplit<1){g=!0;const k=Math.floor(o[0].shape[0]*(1-s.validationSplit)),v=o[0].shape[0];u=Ni(o,k,v),i=o,o=Ni(o,0,k),h=Ni(r,k,v),a=r,r=Ni(r,0,k),x=u.concat(h)}else s.validationSteps!=null&&(g=!0);const b=o.concat(r).concat(d);this.checkTrainableWeightsConsistency();const w=this.makeTrainFunction(),y=this.getDedupedMetricsNames();let C,$;g?(this.makeTestFunction(),C=this.testFunction,$=y.slice().concat(y.map(k=>"val_"+k))):(C=null,x=[],$=y.slice());const N=ex(s.callbacks,s.yieldEvery);return await this.fitLoop(w,b,y,p,s.epochs,s.verbose,N,C,x,s.shuffle,$,s.initialEpoch,null,null)}finally{this.isTraining=!1,In(o,t),In(r,e),In(i,t),In(a,e),In(u,l),In(h,c),d!=null&&It(d)}}async fitLoop(t,e,s,o,r,i,a,l,c,u,h,d,p,f){o==null&&(o=32),r==null&&(r=1),u==null&&(u=!0),d==null&&(d=0);let m=!1;if(l!=null&&c!=null&&(m=!0),f!=null&&(m=!0,p==null))throw new D("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");const g=this.checkNumSamples(e,o,p,"steps_per_epoch");let x;g!=null&&(x=bn(0,g)),i==null&&(i=1);const{callbackList:b,history:w}=nx(a,i,r,d,g,p,o,m,h);b.setModel(this),this.history=w,await b.onTrainBegin(),this.stopTraining_=!1;for(let y=d;y<r;++y){await b.onEpochBegin(y);const C={};if(p!=null)throw new bt("stepsPerEpoch mode is not implemented yet.");{if(u==="batch")throw new bt("batch shuffling is not implemneted yet");u&&Hy(x);const $=Ge(x),N=Id(g,o);for(let T=0;T<N.length;++T){const k={};if(await b.onBatchBegin(T,k),V(()=>{const v=N[T][0],I=N[T][1],R=to($,v,I-v);k.batch=T,k.size=I-v;const A=Cd(e,R),P=t(A);for(let _=0;_<s.length;++_){const B=s[_],z=P[_];k[B]=z,Tn(z)}if(T===N.length-1&&m){const _=this.testLoop(l,c,o);for(let B=0;B<s.length;++B){const z=s[B],W=_[B];Tn(W),C["val_"+z]=W}}}),await b.onBatchEnd(T,k),Qg(k),this.stopTraining_)break}$.dispose()}if(await b.onEpochEnd(y,C),this.stopTraining_)break}return await b.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(t,e){return eE(this,t,e)}async trainOnBatch(t,e){const s=await this.standardizeUserData(t,e),o=s[0],r=s[1],a=this.makeTrainFunction()(o.concat(r)),l=[];for(const c of a){const u=await c.data();l.push(u[0])}return It(a),In(s[0],t),In(s[1],e),Be(l)}getNamedWeights(t){const e=[],s=t!=null&&t.trainableOnly,o=s?this.trainableWeights:this.weights,r=this.getWeights(s);for(let i=0;i<o.length;++i)s&&!o[i].trainable||e.push({name:o[i].originalName,tensor:r[i]});return e}set stopTraining(t){this.stopTraining_=t}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(t){this.optimizer_!==t&&(this.optimizer_=t,this.isOptimizerOwned=!1)}dispose(){const t=super.dispose();if(t.refCountAfterDispose===0&&this.optimizer!=null&&this.isOptimizerOwned){const e=wf().numTensors;this.optimizer_.dispose(),t.numDisposedVariables+=e-wf().numTensors}return t}getLossIdentifiers(){let t;if(typeof this.loss=="string")t=Jn(this.loss);else if(Array.isArray(this.loss)){for(const e of this.loss)if(typeof e!="string")throw new Error("Serialization of non-string loss is not supported.");t=this.loss.map(e=>Jn(e))}else{const e=Object.keys(this.loss);t={};const s=this.loss;for(const o of e)if(typeof s[o]=="string")t[o]=Jn(s[o]);else throw new Error("Serialization of non-string loss is not supported.")}return t}getMetricIdentifiers(){if(typeof this.metrics=="string"||typeof this.metrics=="function")return[Jn(Ul(this.metrics))];if(Array.isArray(this.metrics))return this.metrics.map(t=>Jn(Ul(t)));{const t={};for(const e in this.metrics)t[e]=Jn(Ul(this.metrics[e]));return t}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(t){if(t.weighted_metrics!=null)throw new Error("Loading weight_metrics is not supported yet.");if(t.loss_weights!=null)throw new Error("Loading loss_weights is not supported yet.");if(t.sample_weight_mode!=null)throw new Error("Loading sample_weight_mode is not supported yet.");const e=bd(t.optimizer_config),s=Qn(e);let o;if(typeof t.loss=="string")o=Zs(t.loss);else if(Array.isArray(t.loss))o=t.loss.map(i=>Zs(i));else if(t.loss!=null){o={};for(const i in t.loss)o[i]=Zs(t.loss[i])}let r;if(Array.isArray(t.metrics))r=t.metrics.map(i=>Zs(i));else if(t.metrics!=null){r={};for(const i in t.metrics)r[i]=Zs(t.metrics[i])}this.compile({loss:o,metrics:r,optimizer:s})}async save(t,e){if(typeof t=="string"){const c=Uw(t);if(c.length===0)throw new D(`Cannot find any save handlers for URL '${t}'`);if(c.length>1)throw new D(`Found more than one (${c.length}) save handlers for URL '${t}'`);t=c[0]}if(t.save==null)throw new D("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");const s=await kf(this.getNamedWeights(e)),a={modelTopology:this.toJSON(null,!1),format:cE,generatedBy:`TensorFlow.js tfjs-layers v${hx}`,convertedBy:null};if((e==null?!1:e.includeOptimizer)&&this.optimizer!=null){a.trainingConfig=this.getTrainingConfig();const c="optimizer",{data:u,specs:h}=await kf(await this.optimizer.getWeights(),c);s.specs.push(...h),s.data=Ww([s.data,u])}return this.userDefinedMetadata!=null&&(cx(this.userDefinedMetadata,this.name,!0),a.userDefinedMetadata=this.userDefinedMetadata),a.weightData=s.data,a.weightSpecs=s.specs,t.save(a)}setUserDefinedMetadata(t){cx(t,this.name),this.userDefinedMetadata=t}getUserDefinedMetadata(){return this.userDefinedMetadata}}no.className="Model",Z(no);class Cx extends no{}Cx.className="Functional",Z(Cx);class Ti extends no{constructor(t){if(super({inputs:[],outputs:[]}),t=t||{},this.trainable=!0,this.built=!1,this.name=t.name!=null?t.name:El("sequential_"),t.layers!=null)for(const e of t.layers)this.add(e)}checkShape(t){if(t.inboundNodes[0].outputTensors[0].shape.some(s=>s<0))throw new D(`Negative dimension size caused by adding layer ${t.name} with input shape [${t.inboundNodes[0].inputTensors[0].shape}]`)}add(t){const e=t instanceof Ti||t instanceof no;let s;if(e){if(s=t,s.outputs.length!==1)throw new D("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");if(s.inputs.length!==1)throw new D("All layers in a Sequential model should have a single input tensor. For multi-input layers, use the functional API.")}if(this.outputs.length===0){if(t.inboundNodes.length===0){if(t.batchInputShape==null)throw new D("The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.");const o=Hg({batchShape:t.batchInputShape,dtype:t.dtype,name:t.name+"_input"});t.apply(o)}if(e)this.outputs=s.outputs,this.inputs=s.inputs;else{if(t.inboundNodes.length!==1)throw new D(`A layer added to a Sequential model must not already be connected somewhere else. LayersModel received layer ${t.name} which has ${t.inboundNodes.length} pre-existing inbound connections.`);if(t.inboundNodes[0].outputTensors.length!==1)throw new D("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(t),this.outputs=[t.inboundNodes[0].outputTensors[0]],this.inputs=Gg(this.outputs[0])}this.inboundNodes=[],new Ol({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:Ys(null,this.inputs.length),outputMasks:[null],inputShapes:this.inputs.map(o=>o.shape),outputShapes:this.outputs[0].shape})}else{const o=t.apply(this.outputs[0]);if(Array.isArray(o))throw new TypeError("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(t),this.outputs=[o],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}this.layers.push(t),this.built=!1}pop(){if(this.layers.length===0)throw new TypeError("There are no layers in the model.");if(this.layers.pop(),this.layers.length===0)this.outputs=[],this.inboundNodes=[],this.outboundNodes=[];else{const t=this.layers.length-1;this.layers[t].outboundNodes=[],this.outputs=[this.layers[t].output],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}}call(t,e){return this.model==null&&this.build(),this.model.call(t,e)}build(t){if(Nt(t),this.inputs.length===0||this.outputs.length===0)throw new TypeError("Sequential model cannot be built: model is empty. Add some layers first.");this.model=new no({inputs:this.inputs,outputs:this.outputs[0],name:this.name+"_model"}),this.model.trainable=this.trainable,this.supportsMasking=this.model.supportsMasking,this.inputLayers=this.model.inputLayers,this.inputLayersNodeIndices=this.model.inputLayersNodeIndices,this.inputLayersTensorIndices=this.model.inputLayersTensorIndices,this.outputLayers=this.model.outputLayers,this.outputLayersNodeIndices=this.model.outputLayersNodeIndices,this.outputLayersTensorIndices=this.model.outputLayersTensorIndices,this.nodesByDepth=this.model.nodesByDepth,this.containerNodes=this.model.containerNodes,this.outputNames=this.model.outputNames,this.inputNames=this.model.inputNames,this.built=!0}countParams(){return this.built||this.build(),super.countParams()}summary(t,e,s=console.log){this.built||this.build(),super.summary(t,e,s)}setWeights(t){this.model==null&&this.build(),this.model.setWeights(t)}evaluate(t,e,s={}){if(!this.built)throw new rn("The model needs to be compiled before being used.");return this.model.evaluate(t,e,s)}async evaluateDataset(t,e){if(!this.built)throw new rn("The model needs to be compiled before being used.");return this.model.evaluateDataset(t,e)}predict(t,e={}){return this.model==null&&this.build(),this.model.predict(t,e)}predictOnBatch(t){return this.model==null&&this.build(),this.model.predictOnBatch(t)}compile(t){this.build(),this.model.compile(t),this.optimizer_=this.model.optimizer,this.isOptimizerOwned=this.model.isOptimizerOwned,this.loss=this.model.loss,this.metrics=this.model.metrics,this.metricsTensors=this.model.metricsTensors,this.metricsNames=this.model.metricsNames}get optimizer(){return this.model==null?void 0:this.model.optimizer}set optimizer(t){this.model.optimizer=t}async fit(t,e,s={}){if(!this.built)throw new rn("The model needs to be compiled before being used.");return this.model.fit(t,e,s)}async fitDataset(t,e){if(!this.built)throw new rn("The model needs to be compiled before being used.");return this.model.fitDataset(t,e)}async trainOnBatch(t,e){return this.model.trainOnBatch(t,e)}static fromConfig(t,e,s={},o=!1){let r,i={};if(e instanceof Array){if(e[0].className==null||e[0].className==="Merge")throw new D("Legacy serialization format not supported yet.");r=e}else S(e.layers!=null,()=>"When the config data for a Sequential model is not an Array, it must be an Object that contains the 'layers' field."),r=e.layers,delete e.layers,i=e;const a=new t(i);if(!(a instanceof Ti))throw new bt(`Sequential.fromConfig called on non-Sequential input: ${a}`);for(const l of r){const u=Qn(l,void 0,o);o&&u.setFastWeightInitDuringBuild(!0),a.add(u)}return a}set stopTraining(t){if(this.model==null)throw new D("Cannot set the stopTraining property of a sequential model before it is compiled.");this.model.stopTraining=t}get stopTraining(){if(this.model==null)throw new D("Cannot get the stopTraining property of a sequential model before it is compiled.");return this.model.stopTraining}getConfig(){const t=[];for(const e of this.layers){const s={};s.className=e.getClassName(),s.config=e.getConfig(),t.push(s)}return{name:this.name,layers:t}}}Ti.className="Sequential",Z(Ti);function uE(n){return new no(n)}function hE(n){return Hg(n)}let ve=class extends _o{getConfig(){return{}}};class Ix extends ve{apply(t,e=1){return ZN(t,e)}}Ix.className="elu",Z(Ix);class $x extends ve{apply(t){return cm(t)}}$x.className="selu",Z($x);class kx extends ve{apply(t){return Hs(t)}}kx.className="relu",Z(kx);class vx extends ve{apply(t){return V(()=>ci(6,Hs(t)))}}vx.className="relu6",Z(vx);class Sx extends ve{apply(t){return t}}Sx.className="linear",Z(Sx);class Nx extends ve{apply(t){return Eo(t)}}Nx.className="sigmoid",Z(Nx);class Tx extends ve{apply(t){return QN(t)}}Tx.className="hardSigmoid",Z(Tx);class Ex extends ve{apply(t){return li(t)}}Ex.className="softplus",Z(Ex);class Rx extends ve{apply(t){return JN(t)}}Rx.className="softsign",Z(Rx);class Ax extends ve{apply(t){return ol(t)}}Ax.className="tanh",Z(Ax);let kd=class extends ve{apply(t,e=-1){return lh(t,e)}};kd.className="softmax",Z(kd);class Dx extends ve{apply(t,e=-1){return Kf(t,e)}}Dx.className="logSoftmax",Z(Dx);class Fx extends ve{apply(t){return V(()=>V(()=>{const e=Math.sqrt(2),s=F(.5,J(1,Wf(dt(t,e))));return F(t,s)}))}}Fx.className="gelu",Z(Fx);class Ox extends ve{apply(t){return V(()=>F(.5,F(t,J(1,ol(F(ke(dt(2,Math.PI)),J(t,F(.044715,Us(t,3)))))))))}}Ox.className="gelu_new",Z(Ox);class _x extends ve{apply(t){return V(()=>F(t,ol(li(t))))}}_x.className="mish",Z(_x);class Lx extends ve{apply(t,e=1){return V(()=>F(Eo(F(t,e)),t))}}Lx.className="swish",Z(Lx);function bs(n){return n.getClassName()}function vd(n,t={}){return xi(n,on.getMap().classNameMap,t,"activation")}function ys(n){if(n==null){const t={};return t.className="linear",t.config={},vd(t)}if(typeof n=="string"){const t={};return t.className=n,t.config={},vd(t)}else return n instanceof ve?n:vd(n)}function Mx(n){if(n!=null&&typeof n!="object")throw new Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${n}`)}class Px extends _o{}class Sd extends Px{constructor(t){super(),Mx(t),this.l1=t==null||t.l1==null?.01:t.l1,this.l2=t==null||t.l2==null?.01:t.l2,this.hasL1=this.l1!==0,this.hasL2=this.l2!==0}apply(t){return V(()=>{let e=pe([1]);return this.hasL1&&(e=J(e,ut(F(this.l1,Ee(t))))),this.hasL2&&(e=J(e,ut(F(this.l2,wi(t))))),L(e,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(t,e){return new t({l1:e.l1,l2:e.l2})}}Sd.className="L1L2",Z(Sd);function dE(n){return Mx(n),new Sd({l2:n!=null?n.l2:null,l1:0})}const Bx={l1l2:"L1L2"};function Lt(n){return Xh(n)}function zx(n,t={}){return xi(n,on.getMap().classNameMap,t,"regularizer")}function Gt(n){if(n==null)return null;if(typeof n=="string"){const e={className:n in Bx?Bx[n]:n,config:{}};return zx(e)}else return n instanceof Px?n:zx(n)}class Vx extends Ct{constructor(t){super(t??{}),this.supportsMasking=!0,t!=null&&(this.maxValue=t.maxValue)}call(t,e){t=ft(t);let s=Hs(t);return this.maxValue!=null&&(s=Xe(s,0,this.maxValue)),s}computeOutputShape(t){return t}getConfig(){const t={maxValue:this.maxValue},e=super.getConfig();return Object.assign(t,e),t}}Vx.className="ReLU",Z(Vx);class Wx extends Ct{constructor(t){super(t??{}),this.DEFAULT_ALPHA=.3,t==null&&(t={}),this.alpha=t.alpha==null?this.DEFAULT_ALPHA:t.alpha}call(t,e){const s=ft(t);return Zu(s,this.alpha)}computeOutputShape(t){return t}getConfig(){const t={alpha:this.alpha},e=super.getConfig();return Object.assign(t,e),t}}Wx.className="LeakyReLU",Z(Wx);class Ux extends Ct{constructor(t){if(super(t??{}),this.DEFAULT_ALPHA_INITIALIZER="zeros",t==null&&(t={}),this.supportsMasking=!0,this.alphaInitializer=Ut(t.alphaInitializer||this.DEFAULT_ALPHA_INITIALIZER),this.alphaRegularizer=Gt(t.alphaRegularizer),this.alphaConstraint=ce(t.alphaConstraint),t.sharedAxes==null)this.sharedAxes=null;else if(Array.isArray(t.sharedAxes))this.sharedAxes=t.sharedAxes;else if(typeof t.sharedAxes=="number")this.sharedAxes=[t.sharedAxes];else throw new D(`Expected sharedAxes to be a number or an array of numbers, but got ${t.sharedAxes}`)}build(t){t=Nt(t);const e=t.slice(1);if(this.sharedAxes!=null)for(const o of this.sharedAxes)e[o-1]=1;this.alpha=this.addWeight("alpha",e,"float32",this.alphaInitializer,this.alphaRegularizer,!0,this.alphaConstraint);const s={};if(this.sharedAxes!=null)for(let o=1;o<t.length;++o)s[o]=t[o];this.inputSpec=[new ae({ndim:t.length,axes:s})],this.built=!0}call(t,e){return t=ft(t),sh(t,this.alpha.read())}getConfig(){const t={alphaInitializer:Kt(this.alphaInitializer),alphaRegularizer:Lt(this.alphaRegularizer),alphaConstraint:le(this.alphaConstraint),sharedAxes:this.sharedAxes},e=super.getConfig();return Object.assign(t,e),t}}Ux.className="PReLU",Z(Ux);let Gx=class extends Ct{constructor(t){if(super(t??{}),this.DEFAULT_ALPHA=1,t==null&&(t={}),t.alpha!=null&&t.alpha!==this.DEFAULT_ALPHA)throw new bt(`Non-default alpha value (${t.alpha}) is not supported by the ELU layer yet.`);this.alpha=t.alpha==null?this.DEFAULT_ALPHA:t.alpha}call(t,e){const s=ft(t);return al(s)}computeOutputShape(t){return t}getConfig(){const t={alpha:this.alpha},e=super.getConfig();return Object.assign(t,e),t}};Gx.className="ELU",Z(Gx);class Hx extends Ct{constructor(t){super(t??{}),this.DEFAULT_THETA=1,t==null&&(t={}),this.theta=t.theta==null?this.DEFAULT_THETA:t.theta}call(t,e){const s=ft(t);return F(s,nt(Ye(s,this.theta),"float32"))}computeOutputShape(t){return t}getConfig(){const t={theta:this.theta},e=super.getConfig();return Object.assign(t,e),t}}Hx.className="ThresholdedReLU",Z(Hx);class qx extends Ct{constructor(t){super(t??{}),this.DEFAULT_AXIS=1,t==null&&(t={}),this.softmax=new kd().apply,this.axis=t.axis==null?this.DEFAULT_AXIS:t.axis}call(t,e){return V(()=>{let s=ft(t);const o=e.mask;if(o!=null){const r=F(pt(ds(s.shape),nt(o,s.dtype)),At(-1e9));s=J(s,r)}return this.axis instanceof Array?this.axis.length>1?An(pt(s,jf(s,this.axis,!0))):this.softmax(s,this.axis[0]):this.softmax(s,this.axis)})}computeOutputShape(t){return t}getConfig(){const t={axis:this.axis},e=super.getConfig();return Object.assign(t,e),t}}qx.className="Softmax",Z(qx);function Po(n,t,e){if(typeof n=="number")return Ys(n,t);if(n.length!==t)throw new D(`The ${e} argument must be an integer or tuple of ${t} integers. Received: ${n.length} elements.`);for(let s=0;s<t;++s){const o=n[s];if(!KN(o))throw new D(`The ${e} argument must be an integer or tuple of ${t} integers. Received: ${JSON.stringify(n)} including a non-integer number ${o}`)}return n}function $n(n,t,e,s,o=1){if(n==null)return n;const r=t+(t-1)*(o-1);let i;return e==="same"?i=n:i=n-r+1,Math.floor((i+s-1)/s)}function Bn(n,t,e,s){if(n==null)return null;if(s==="valid")n=n*t+gs([e-t,0]);else if(s==="same")n=n*t;else throw new D(`Unsupport padding mode: ${s}.`);return n}function Nd(n,t){return V(()=>(te(t),t==="channelsFirst"?vt(n,[0,2,3,1]):n))}function Kx(n,t){return V(()=>(te(t),t==="channelsFirst"?vt(n,[0,2,3,4,1]):n))}function pE(n,t,e,s=1,o="valid",r,i=1){return V(()=>{if(r==null&&(r=yn()),te(r),n.shape.length!==3)throw new D(`The input of a conv1dWithBias operation should be 3, but is ${n.shape.length} instead.`);if(t.shape.length!==3)throw new D(`The kernel for a conv1dWithBias operation should be 3, but is ${t.shape.length} instead`);if(e!=null&&e.shape.length!==1)throw new D(`The bias for a conv1dWithBias operation should be 1, but is ${e.shape.length} instead`);if(r==="channelsFirst"&&(n=vt(n,[0,2,1])),o==="causal")throw new bt("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");let a=Lf(n,t,s,o==="same"?"same":"valid","NWC",i);return e!=null&&(a=wn(a,e)),a})}function jx(n,t,e,s=[1,1],o="valid",r,i,a=null){return V(()=>{if(r==null&&(r=yn()),te(r),n.rank!==3&&n.rank!==4)throw new D(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${n.rank}.`);if(t.rank!==3&&t.rank!==4)throw new D(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${n.rank}.`);let l=Nd(n,r);if(o==="causal")throw new bt("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");return l=Cv({x:l,filter:t,strides:s,pad:o==="same"?"same":"valid",dilations:i,dataFormat:"NHWC",bias:e,activation:a}),r==="channelsFirst"&&(l=vt(l,[0,3,1,2])),l})}function fE(n,t,e,s=[1,1,1],o="valid",r,i){return V(()=>{if(r==null&&(r=yn()),te(r),n.rank!==4&&n.rank!==5)throw new D(`conv3dWithBias expects input to be of rank 4 or 5, but received ${n.rank}.`);if(t.rank!==4&&t.rank!==5)throw new D(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${n.rank}.`);let a=Kx(n,r);if(o==="causal")throw new bt("The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.");return a=bI(a,t,s,o==="same"?"same":"valid","NDHWC",i),e!=null&&(a=wn(a,e)),r==="channelsFirst"&&(a=vt(a,[0,4,1,2,3])),a})}class Hl extends Ct{constructor(t,e){if(super(e),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",Hl.verifyArgs(e),this.rank=t,fe(this.rank,"rank"),this.rank!==1&&this.rank!==2&&this.rank!==3)throw new bt(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=Po(e.kernelSize,t,"kernelSize"),this.strides=Po(e.strides==null?1:e.strides,t,"strides"),this.padding=e.padding==null?"valid":e.padding,Je(this.padding),this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,te(this.dataFormat),this.activation=ys(e.activation),this.useBias=e.useBias==null?!0:e.useBias,this.biasInitializer=Ut(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=ce(e.biasConstraint),this.biasRegularizer=Gt(e.biasRegularizer),this.activityRegularizer=Gt(e.activityRegularizer),this.dilationRate=Po(e.dilationRate==null?1:e.dilationRate,t,"dilationRate"),this.rank===1&&Array.isArray(this.dilationRate)&&this.dilationRate.length!==1)throw new D(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(this.rank===2){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==2)throw new D(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(this.rank===3){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==3)throw new D(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(t){if(_n("kernelSize"in t,"required key 'kernelSize' not in config"),typeof t.kernelSize!="number"&&!Zh(t.kernelSize,"number",1,3))throw new D(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(t.kernelSize)}.`)}getConfig(){const t={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:bs(this.activation),useBias:this.useBias,biasInitializer:Kt(this.biasInitializer),biasRegularizer:Lt(this.biasRegularizer),activityRegularizer:Lt(this.activityRegularizer),biasConstraint:le(this.biasConstraint)},e=super.getConfig();return Object.assign(t,e),t}}class Bo extends Hl{constructor(t,e){super(t,e),this.kernel=null,Bo.verifyArgs(e),this.filters=e.filters,fe(this.filters,"filters"),this.kernelInitializer=Ut(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=ce(e.kernelConstraint),this.kernelRegularizer=Gt(e.kernelRegularizer)}build(t){t=Nt(t);const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new D(`The channel dimension of the input should be defined. Found ${t[e]}`);const s=t[e],o=this.kernelSize.concat([s,this.filters]);this.kernel=this.addWeight("kernel",o,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[e]:s}}],this.built=!0}call(t,e){return V(()=>{t=ft(t);let s;const o=this.bias==null?null:this.bias.read(),r=kg(this.activation.getClassName());if(r!=null&&this.rank===2)s=jx(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate,r);else{if(this.rank===1)s=pE(t,this.kernel.read(),o,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(this.rank===2)s=jx(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(this.rank===3)s=fE(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new bt("convolutions greater than 3D are not implemented yet.");this.activation!=null&&(s=this.activation.apply(s))}return s})}computeOutputShape(t){t=Nt(t);const e=[],s=this.dataFormat==="channelsLast"?t.slice(1,t.length-1):t.slice(2);for(let r=0;r<s.length;++r){const i=$n(s[r],this.kernelSize[r],this.padding,this.strides[r],typeof this.dilationRate=="number"?this.dilationRate:this.dilationRate[r]);e.push(i)}let o=[t[0]];return this.dataFormat==="channelsLast"?(o=o.concat(e),o.push(this.filters)):(o.push(this.filters),o=o.concat(e)),o}getConfig(){const t={filters:this.filters,kernelInitializer:Kt(this.kernelInitializer),kernelRegularizer:Lt(this.kernelRegularizer),kernelConstraint:le(this.kernelConstraint)},e=super.getConfig();return Object.assign(t,e),t}static verifyArgs(t){if(!("filters"in t)||typeof t.filters!="number"||t.filters<1)throw new D(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(t.filters)}`)}}class Ei extends Bo{constructor(t){super(2,t),Ei.verifyArgs(t)}getConfig(){const t=super.getConfig();return delete t.rank,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!Zh(t.kernelSize,"number",1,2))throw new D(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(t.kernelSize)}.`)}}Ei.className="Conv2D",Z(Ei);class Ri extends Bo{constructor(t){super(3,t),Ri.verifyArgs(t)}getConfig(){const t=super.getConfig();return delete t.rank,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!(Array.isArray(t.kernelSize)&&(t.kernelSize.length===1||t.kernelSize.length===3)))throw new D(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(t.kernelSize)}.`)}}Ri.className="Conv3D",Z(Ri);class Xx extends Ei{constructor(t){if(super(t),this.inputSpec=[new ae({ndim:4})],this.padding!=="same"&&this.padding!=="valid")throw new D(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(t){if(t=Nt(t),t.length!==4)throw new D("Input should have rank 4; Received input shape: "+JSON.stringify(t));const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new D("The channel dimension of the inputs should be defined. Found `None`.");const s=t[e],o=this.kernelSize.concat([this.filters,s]);this.kernel=this.addWeight("kernel",o,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new ae({ndim:4,axes:{[e]:s}})],this.built=!0}call(t,e){return V(()=>{let s=ft(t);if(s.shape.length!==4)throw new D(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${s.shape.length}`);const o=s.shape,r=o[0];let i,a;this.dataFormat==="channelsFirst"?(i=2,a=3):(i=1,a=2);const l=o[i],c=o[a],u=this.kernelSize[0],h=this.kernelSize[1],d=this.strides[0],p=this.strides[1],f=Bn(l,d,u,this.padding),m=Bn(c,p,h,this.padding),g=[r,f,m,this.filters];this.dataFormat!=="channelsLast"&&(s=vt(s,[0,2,3,1]));let x=Mf(s,this.kernel.read(),g,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(x=vt(x,[0,3,1,2])),this.bias!=null&&(x=wn(x,this.bias.read(),this.dataFormat)),this.activation!=null&&(x=this.activation.apply(x)),x})}computeOutputShape(t){t=Nt(t);const e=t.slice();let s,o,r;this.dataFormat==="channelsFirst"?(s=1,o=2,r=3):(s=3,o=1,r=2);const i=this.kernelSize[0],a=this.kernelSize[1],l=this.strides[0],c=this.strides[1];return e[s]=this.filters,e[o]=Bn(e[o],l,i,this.padding),e[r]=Bn(e[r],c,a,this.padding),e}getConfig(){const t=super.getConfig();return delete t.dilationRate,t}}Xx.className="Conv2DTranspose",Z(Xx);class Yx extends Ri{constructor(t){if(super(t),this.inputSpec=[new ae({ndim:5})],this.padding!=="same"&&this.padding!=="valid")throw new D(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(t){if(t=Nt(t),t.length!==5)throw new D("Input should have rank 5; Received input shape: "+JSON.stringify(t));const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new D("The channel dimension of the inputs should be defined. Found `None`.");const s=t[e],o=this.kernelSize.concat([this.filters,s]);this.kernel=this.addWeight("kernel",o,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new ae({ndim:5,axes:{[e]:s}})],this.built=!0}call(t,e){return V(()=>{let s=ft(t);if(s.shape.length!==5)throw new D(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${s.shape.length}`);const o=s.shape,r=o[0];let i,a,l;this.dataFormat==="channelsFirst"?(l=2,i=3,a=4):(l=1,i=2,a=3);const c=o[l],u=o[i],h=o[a],d=this.kernelSize[0],p=this.kernelSize[1],f=this.kernelSize[2],m=this.strides[0],g=this.strides[1],x=this.strides[2],b=Bn(c,m,d,this.padding),w=Bn(u,g,p,this.padding),y=Bn(h,x,f,this.padding),C=[r,b,w,y,this.filters];this.dataFormat!=="channelsLast"&&(s=vt(s,[0,2,3,4,1]));let $=CI(s,this.kernel.read(),C,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&($=vt($,[0,4,1,2,3])),this.bias!==null&&($=wn($,this.bias.read(),this.dataFormat)),this.activation!==null&&($=this.activation.apply($)),$})}computeOutputShape(t){t=Nt(t);const e=t.slice();let s,o,r,i;this.dataFormat==="channelsFirst"?(s=1,o=2,r=3,i=4):(s=4,o=1,r=2,i=3);const a=this.kernelSize[0],l=this.kernelSize[1],c=this.kernelSize[2],u=this.strides[0],h=this.strides[1],d=this.strides[2];return e[s]=this.filters,e[o]=Bn(e[o],u,a,this.padding),e[r]=Bn(e[r],h,l,this.padding),e[i]=Bn(e[i],d,c,this.padding),e}getConfig(){const t=super.getConfig();return delete t.dilationRate,t}}Yx.className="Conv3DTranspose",Z(Yx);class Zx extends Bo{constructor(t,e){if(super(t,e),this.DEFAULT_DEPTHWISE_INITIALIZER="glorotUniform",this.DEFAULT_POINTWISE_INITIALIZER="glorotUniform",this.depthwiseKernel=null,this.pointwiseKernel=null,e.filters==null)throw new D("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(e.kernelInitializer!=null||e.kernelRegularizer!=null||e.kernelConstraint!=null)throw new D("Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.");if(e.padding!=null&&e.padding!=="same"&&e.padding!=="valid")throw new D(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(e.padding)}`);this.depthMultiplier=e.depthMultiplier==null?1:e.depthMultiplier,this.depthwiseInitializer=Ut(e.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=Gt(e.depthwiseRegularizer),this.depthwiseConstraint=ce(e.depthwiseConstraint),this.pointwiseInitializer=Ut(e.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=Gt(e.pointwiseRegularizer),this.pointwiseConstraint=ce(e.pointwiseConstraint)}build(t){if(t=Nt(t),t.length<this.rank+2)throw new D(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(t)}`);const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null||t[e]<0)throw new D(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(t[e])}`);const s=t[e],o=this.kernelSize.concat([s,this.depthMultiplier]),r=[];for(let a=0;a<this.rank;++a)r.push(1);r.push(s*this.depthMultiplier,this.filters);const i=!0;this.depthwiseKernel=this.addWeight("depthwise_kernel",o,"float32",this.depthwiseInitializer,this.depthwiseRegularizer,i,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight("pointwise_kernel",r,"float32",this.pointwiseInitializer,this.pointwiseRegularizer,i,this.pointwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,i,this.biasConstraint):this.bias=null,this.inputSpec=[new ae({ndim:this.rank+2,axes:{[e]:s}})],this.built=!0}call(t,e){return V(()=>{t=ft(t);let s;if(this.rank===1)throw new bt("1D separable convolution is not implemented yet.");return this.rank===2&&(this.dataFormat==="channelsFirst"&&(t=vt(t,[0,2,3,1])),s=um(t,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,"NHWC")),this.useBias&&(s=wn(s,this.bias.read(),this.dataFormat)),this.activation!=null&&(s=this.activation.apply(s)),this.dataFormat==="channelsFirst"&&(s=vt(s,[0,3,1,2])),s})}getConfig(){const t=super.getConfig();return delete t.rank,delete t.kernelInitializer,delete t.kernelRegularizer,delete t.kernelConstraint,t.depthwiseInitializer=Kt(this.depthwiseInitializer),t.pointwiseInitializer=Kt(this.pointwiseInitializer),t.depthwiseRegularizer=Lt(this.depthwiseRegularizer),t.pointwiseRegularizer=Lt(this.pointwiseRegularizer),t.depthwiseConstraint=le(this.depthwiseConstraint),t.pointwiseConstraint=le(this.pointwiseConstraint),t}}Zx.className="SeparableConv";class Jx extends Zx{constructor(t){super(2,t)}}Jx.className="SeparableConv2D",Z(Jx);class ql extends Bo{constructor(t){super(1,t),ql.verifyArgs(t),this.inputSpec=[{ndim:3}]}getConfig(){const t=super.getConfig();return delete t.rank,delete t.dataFormat,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!Zh(t.kernelSize,"number",1,1))throw new D(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(t.kernelSize)}.`)}}ql.className="Conv1D",Z(ql);class Qx extends Ct{constructor(t){super(t),typeof t.cropping=="number"?this.cropping=[[t.cropping,t.cropping],[t.cropping,t.cropping]]:typeof t.cropping[0]=="number"?this.cropping=[[t.cropping[0],t.cropping[0]],[t.cropping[1],t.cropping[1]]]:this.cropping=t.cropping,this.dataFormat=t.dataFormat===void 0?"channelsLast":t.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(t){return this.dataFormat==="channelsFirst"?[t[0],t[1],t[2]-this.cropping[0][0]-this.cropping[0][1],t[3]-this.cropping[1][0]-this.cropping[1][1]]:[t[0],t[1]-this.cropping[0][0]-this.cropping[0][1],t[2]-this.cropping[1][0]-this.cropping[1][1],t[3]]}call(t,e){return V(()=>{if(t=ft(t),this.dataFormat==="channelsLast"){const s=Rl(t,this.cropping[0][0],t.shape[1]-this.cropping[0][0]-this.cropping[0][1],2);return Rl(s,this.cropping[1][0],t.shape[2]-this.cropping[1][1]-this.cropping[1][0],3)}else{const s=Rl(t,this.cropping[0][0],t.shape[2]-this.cropping[0][0]-this.cropping[0][1],3);return Rl(s,this.cropping[1][0],t.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)}})}getConfig(){const t={cropping:this.cropping,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}Qx.className="Cropping2D",Z(Qx);class tb extends Ct{constructor(t){super(t),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=t.size==null?this.DEFAULT_SIZE:t.size,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,te(this.dataFormat),this.interpolation=t.interpolation==null?"nearest":t.interpolation,GN(this.interpolation)}computeOutputShape(t){if(this.dataFormat==="channelsFirst"){const e=t[2]==null?null:this.size[0]*t[2],s=t[3]==null?null:this.size[1]*t[3];return[t[0],t[1],e,s]}else{const e=t[1]==null?null:this.size[0]*t[1],s=t[2]==null?null:this.size[1]*t[2];return[t[0],e,s,t[3]]}}call(t,e){return V(()=>{let s=ft(t);const o=s.shape;if(this.dataFormat==="channelsFirst"){s=vt(s,[0,2,3,1]);const r=this.size[0]*o[2],i=this.size[1]*o[3],a=this.interpolation==="nearest"?Xn.resizeNearestNeighbor(s,[r,i]):Xn.resizeBilinear(s,[r,i]);return vt(a,[0,3,1,2])}else{const r=this.size[0]*o[1],i=this.size[1]*o[2];return this.interpolation==="nearest"?Xn.resizeNearestNeighbor(s,[r,i]):Xn.resizeBilinear(s,[r,i])}})}getConfig(){const t={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation},e=super.getConfig();return Object.assign(t,e),t}}tb.className="UpSampling2D",Z(tb);function mE(n,t,e=[1,1],s="valid",o,r){return V(()=>{o==null&&(o=yn()),te(o);let i=Nd(n,o);if(n.rank!==4)throw new D(`Input for depthwiseConv2d is required to be 4-D, but is instead ${n.rank}-D`);if(t.rank!==4)throw new D(`depthwiseKernel is required to be 4-D, but is instead ${t.rank}-D`);return i=Ku(i,t,e,s==="same"?"same":"valid","NHWC",r),o==="channelsFirst"&&(i=vt(i,[0,3,1,2])),i})}class eb extends Hl{constructor(t){super(2,t),this.depthwiseKernel=null,this.depthMultiplier=t.depthMultiplier==null?1:t.depthMultiplier,this.depthwiseInitializer=Ut(t.depthwiseInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.depthwiseConstraint=ce(t.depthwiseConstraint),this.depthwiseRegularizer=Gt(t.depthwiseRegularizer)}build(t){if(t=Nt(t),t.length<4)throw new D(`Inputs to DepthwiseConv2D should have rank 4. Received input shape: ${JSON.stringify(t)}.`);const e=this.dataFormat==="channelsFirst"?1:3;if(t[e]==null||t[e]<0)throw new D(`The channel dimension of the inputs to DepthwiseConv2D should be defined, but is not (${t[e]}).`);const s=t[e],o=[this.kernelSize[0],this.kernelSize[1],s,this.depthMultiplier];this.depthwiseKernel=this.addWeight("depthwise_kernel",o,null,this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[s*this.depthMultiplier],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return V(()=>{t=ft(t);let s=mE(t,this.depthwiseKernel.read(),this.strides,this.padding,this.dataFormat,null);return this.useBias&&(s=wn(s,this.bias.read(),this.dataFormat)),this.activation!=null&&(s=this.activation.apply(s)),s})}computeOutputShape(t){t=Nt(t);const e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2],o=this.dataFormat==="channelsFirst"?t[1]*this.depthMultiplier:t[3]*this.depthMultiplier,r=$n(e,this.kernelSize[0],this.padding,this.strides[0]),i=$n(s,this.kernelSize[1],this.padding,this.strides[1]);return this.dataFormat==="channelsFirst"?[t[0],o,r,i]:[t[0],r,i,o]}getConfig(){const t=super.getConfig();return t.depthMultiplier=this.depthMultiplier,t.depthwiseInitializer=Kt(this.depthwiseInitializer),t.depthwiseRegularizer=Lt(this.depthwiseRegularizer),t.depthwiseConstraint=le(this.depthwiseRegularizer),t}}eb.className="DepthwiseConv2D",Z(eb);function nb(n,t,e,s){if(Array.isArray(n)){if(t!=null||e!=null)throw new D("When inputs is an array, neither initialState or constants should be provided");s!=null&&(e=n.slice(n.length-s,n.length),n=n.slice(0,n.length-s)),n.length>1&&(t=n.slice(1,n.length)),n=n[0]}function o(r){return r==null||Array.isArray(r)?r:[r]}return t=o(t),e=o(e),{inputs:n,initialState:t,constants:e}}function sb(n,t,e,s=!1,o,r,i=!1,a=!1){return V(()=>{const l=t.shape.length;if(l<3)throw new D(`Input should be at least 3D, but is ${l}D.`);const c=[1,0].concat(bn(2,l));t=vt(t,c),i&&console.warn("Backend rnn(): the unroll = true option is not applicable to the imperative deeplearn.js backend."),o!=null&&(o=nt(nt(o,"bool"),"float32"),o.rank===l-1&&(o=Pe(o,-1)),o=vt(o,c)),s&&(t=qs(t,0),o!=null&&(o=qs(o,0)));const u=[];let h,d=e;const p=t.shape[0],f=js(t);let m;o!=null&&(m=js(o));for(let x=0;x<p;++x){const b=f[x],w=V(()=>n(b,d));if(o==null)h=w[0],d=w[1];else{const y=V(()=>{const C=m[x],$=pt(sn(C),C),N=J(F(w[0],C),F(d[0],$)),T=d.map((k,v)=>J(F(w[1][v],C),F(k,$)));return{output:N,newStates:T}});h=y.output,d=y.newStates}a&&u.push(h)}let g;return a&&(g=jn(u,1)),[h,g,d]})}class ws extends Ct{constructor(t){super(t);let e;if(t.cell==null)throw new D("cell property is missing for the constructor of RNN.");if(Array.isArray(t.cell)?e=new Ad({cells:t.cell}):e=t.cell,e.stateSize==null)throw new D("The RNN cell should have an attribute `stateSize` (tuple of integers, one integer per RNN state).");this.cell=e,this.returnSequences=t.returnSequences==null?!1:t.returnSequences,this.returnState=t.returnState==null?!1:t.returnState,this.goBackwards=t.goBackwards==null?!1:t.goBackwards,this._stateful=t.stateful==null?!1:t.stateful,this.unroll=t.unroll==null?!1:t.unroll,this.supportsMasking=!0,this.inputSpec=[new ae({ndim:3})],this.stateSpec=null,this.states_=null,this.numConstants=null,this.keptStates=[]}getStates(){if(this.states_==null){const t=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;return bn(0,t).map(e=>null)}else return this.states_}setStates(t){this.states_=t}computeOutputShape(t){ud(t)&&(t=t[0]),t=t;let e=this.cell.stateSize;Array.isArray(e)||(e=[e]);const s=e[0];let o;if(this.returnSequences?o=[t[0],t[1],s]:o=[t[0],s],this.returnState){const r=[];for(const i of e)r.push([t[0],i]);return[o].concat(r)}else return o}computeMask(t,e){return V(()=>{Array.isArray(e)&&(e=e[0]);const s=this.returnSequences?e:null;if(this.returnState){const o=this.states.map(r=>null);return[s].concat(o)}else return s})}get states(){if(this.states_==null){const t=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1,e=[];for(let s=0;s<t;++s)e.push(null);return e}else return this.states_}set states(t){this.states_=t}build(t){if(this.numConstants!=null)throw new bt("Constants support is not implemented in RNN yet.");ud(t)&&(t=t[0]),t=t;const e=this.stateful?t[0]:null,s=t.slice(2);this.inputSpec[0]=new ae({shape:[e,null,...s]});const o=[t[0]].concat(t.slice(2));this.cell.build(o);let r;if(Array.isArray(this.cell.stateSize)?r=this.cell.stateSize:r=[this.cell.stateSize],this.stateSpec!=null){if(!Et(this.stateSpec.map(i=>i.shape[i.shape.length-1]),r))throw new D(`An initialState was passed that is not compatible with cell.stateSize. Received stateSpec=${this.stateSpec}; However cell.stateSize is ${this.cell.stateSize}`)}else this.stateSpec=r.map(i=>new ae({shape:[null,i]}));this.stateful&&this.resetStates()}resetStates(t,e=!1){V(()=>{if(!this.stateful)throw new On("Cannot call resetStates() on an RNN Layer that is not stateful.");const s=this.inputSpec[0].shape[0];if(s==null)throw new D("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.states_==null)Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(o=>pe([s,o])):this.states_=[pe([s,this.cell.stateSize])];else if(t==null)It(this.states_),this.keptStates!=null&&(It(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(o=>pe([s,o])):this.states_[0]=pe([s,this.cell.stateSize]);else{if(Array.isArray(t)||(t=[t]),t.length!==this.states_.length)throw new D(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${t.length} state value(s). Input received: ${t}`);e===!0?this.keptStates.push(this.states_.slice()):It(this.states_);for(let o=0;o<this.states_.length;++o){const r=t[o],i=Array.isArray(this.cell.stateSize)?this.cell.stateSize[o]:this.cell.stateSize,a=[s,i];if(!Et(r.shape,a))throw new D(`State ${o} is incompatible with layer ${this.name}: expected shape=${a}, received shape=${r.shape}`);this.states_[o]=r}}this.states_=this.states_.map(o=>Tn(o.clone()))})}apply(t,e){let s=e==null?null:e.initialState,o=e==null?null:e.constants;e==null&&(e={});const r=nb(t,s,o,this.numConstants);t=r.inputs,s=r.initialState,o=r.constants;let i=[],a=[];if(s!=null){e.initialState=s,i=i.concat(s),this.stateSpec=[];for(const c of s)this.stateSpec.push(new ae({shape:c.shape}));a=a.concat(this.stateSpec)}if(o!=null&&(e.constants=o,i=i.concat(o),this.numConstants=o.length),i[0]instanceof Pn){const c=[t].concat(i),u=this.inputSpec.concat(a),h=this.inputSpec;this.inputSpec=u;const d=super.apply(c,e);return this.inputSpec=h,d}else return super.apply(t,e)}call(t,e){return V(()=>{const s=e==null?null:e.mask,o=e==null?null:e.training;let r=e==null?null:e.initialState;t=ft(t),r==null&&(this.stateful?r=this.states_:r=this.getInitialState(t));const i=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;if(r.length!==i)throw new D(`RNN Layer has ${i} state(s) but was passed ${r.length} initial state(s).`);this.unroll&&console.warn("Ignoring unroll = true for RNN layer, due to imperative backend.");const a={training:o},c=sb((f,m)=>{const g=this.cell.call([f].concat(m),a);return[g[0],g.slice(1)]},t,r,this.goBackwards,s,null,this.unroll,this.returnSequences),u=c[0],h=c[1],d=c[2];this.stateful&&this.resetStates(d,o);const p=this.returnSequences?h:u;return this.returnState?[p].concat(d):p})}getInitialState(t){return V(()=>{let e=pe(t.shape);return e=ut(e,[1,2]),e=yi(e),Array.isArray(this.cell.stateSize)?this.cell.stateSize.map(s=>s>1?ed(e,[1,s]):e):this.cell.stateSize>1?[ed(e,[1,this.cell.stateSize])]:[e]})}get trainableWeights(){return this.trainable?this.cell.trainableWeights:[]}get nonTrainableWeights(){return this.trainable?this.cell.nonTrainableWeights:this.cell.weights}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.cell!=null&&this.cell.setFastWeightInitDuringBuild(t)}getConfig(){const t=super.getConfig(),e={returnSequences:this.returnSequences,returnState:this.returnState,goBackwards:this.goBackwards,stateful:this.stateful,unroll:this.unroll};this.numConstants!=null&&(e.numConstants=this.numConstants);const s=this.cell.getConfig();return this.getClassName()===ws.className&&(e.cell={className:this.cell.getClassName(),config:s}),Object.assign(Object.assign(Object.assign({},s),t),e)}static fromConfig(t,e,s={}){const o=e.cell,r=Qn(o,s);return new t(Object.assign(e,{cell:r}))}}ws.className="RNN",Z(ws);class Kl extends Ct{}class Td extends Kl{constructor(t){super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=t.units,fe(this.units,"units"),this.activation=ys(t.activation==null?this.DEFAULT_ACTIVATION:t.activation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Ut(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Ut(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Ut(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=Gt(t.kernelRegularizer),this.recurrentRegularizer=Gt(t.recurrentRegularizer),this.biasRegularizer=Gt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Mo([1,gs([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Mo([1,gs([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){t=Nt(t),this.kernel=this.addWeight("kernel",[t[t.length-1],this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return V(()=>{if(t=t,t.length!==2)throw new D(`SimpleRNNCell expects 2 input Tensors, got ${t.length}.`);let s=t[1];t=t[0];const o=e.training==null?!1:e.training;0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=Cs({ones:()=>sn(t),rate:this.dropout,training:o,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=Cs({ones:()=>sn(s),rate:this.recurrentDropout,training:o,dropoutFunc:this.dropoutFunc}));let r;const i=this.dropoutMask,a=this.recurrentDropoutMask;i!=null?r=Mn(F(t,i),this.kernel.read()):r=Mn(t,this.kernel.read()),this.bias!=null&&(r=wn(r,this.bias.read())),a!=null&&(s=F(s,a));let l=J(r,Mn(s,this.recurrentKernel.read()));return this.activation!=null&&(l=this.activation.apply(l)),[l,l]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:bs(this.activation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Lt(this.kernelRegularizer),recurrentRegularizer:Lt(this.recurrentRegularizer),biasRegularizer:Lt(this.biasRegularizer),activityRegularizer:Lt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout};return Object.assign(Object.assign({},t),e)}}Td.className="SimpleRNNCell",Z(Td);class ob extends ws{constructor(t){t.cell=new Td(t),super(t)}call(t,e){return V(()=>{this.cell.dropoutMask!=null&&(It(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(It(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return new t(e)}}ob.className="SimpleRNN",Z(ob);class Ed extends Kl{constructor(t){if(super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",t.resetAfter)throw new D("GRUCell does not support reset_after parameter set to true.");this.units=t.units,fe(this.units,"units"),this.activation=ys(t.activation===void 0?this.DEFAULT_ACTIVATION:t.activation),this.recurrentActivation=ys(t.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:t.recurrentActivation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Ut(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Ut(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Ut(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=Gt(t.kernelRegularizer),this.recurrentRegularizer=Gt(t.recurrentRegularizer),this.biasRegularizer=Gt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Mo([1,gs([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Mo([1,gs([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.implementation=t.implementation,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){t=Nt(t);const e=t[t.length-1];this.kernel=this.addWeight("kernel",[e,this.units*3],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units*3],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[this.units*3],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return V(()=>{if(t=t,t.length!==2)throw new D(`GRUCell expects 2 input Tensors (inputs, h, c), got ${t.length}.`);const s=e.training==null?!1:e.training;let o=t[1];t=t[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=Cs({ones:()=>sn(t),rate:this.dropout,training:s,count:3,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=Cs({ones:()=>sn(o),rate:this.recurrentDropout,training:s,count:3,dropoutFunc:this.dropoutFunc}));const r=this.dropoutMask,i=this.recurrentDropoutMask;let a,l,c;0<this.dropout&&this.dropout<1&&(t=F(t,r[0]));let u=Mn(t,this.kernel.read());this.useBias&&(u=wn(u,this.bias.read())),0<this.recurrentDropout&&this.recurrentDropout<1&&(o=F(o,i[0]));const h=this.recurrentKernel.read(),[d,p]=Ze(h,[2*this.units,this.units],h.rank-1),f=Mn(o,d),[m,g,x]=Ze(u,3,u.rank-1),[b,w]=Ze(f,2,f.rank-1);a=this.recurrentActivation.apply(J(m,b)),l=this.recurrentActivation.apply(J(g,w));const y=Mn(F(l,o),p);c=this.activation.apply(J(x,y));const C=J(F(a,o),F(J(1,Qt(a)),c));return[C,C]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:bs(this.activation),recurrentActivation:bs(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Lt(this.kernelRegularizer),recurrentRegularizer:Lt(this.recurrentRegularizer),biasRegularizer:Lt(this.biasRegularizer),activityRegularizer:Lt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation,resetAfter:!1};return Object.assign(Object.assign({},t),e)}}Ed.className="GRUCell",Z(Ed);class rb extends ws{constructor(t){t.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),t.cell=new Ed(t),super(t)}call(t,e){return V(()=>{this.cell.dropoutMask!=null&&(It(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(It(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return e.implmentation===0&&(e.implementation=1),new t(e)}}rb.className="GRU",Z(rb);class jl extends Kl{constructor(t){super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=t.units,fe(this.units,"units"),this.activation=ys(t.activation===void 0?this.DEFAULT_ACTIVATION:t.activation),this.recurrentActivation=ys(t.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:t.recurrentActivation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Ut(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Ut(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Ut(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.unitForgetBias=t.unitForgetBias,this.kernelRegularizer=Gt(t.kernelRegularizer),this.recurrentRegularizer=Gt(t.recurrentRegularizer),this.biasRegularizer=Gt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Mo([1,gs([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Mo([1,gs([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.implementation=t.implementation,this.stateSize=[this.units,this.units],this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){var e;t=Nt(t);const s=t[t.length-1];this.kernel=this.addWeight("kernel",[s,this.units*4],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units*4],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint);let o;if(this.useBias){if(this.unitForgetBias){const r=this.biasInitializer,i=this.units;o=new(e=class extends ln{apply(l,c){const u=r.apply([i]),h=new sd().apply([i]),d=r.apply([i*2]);return Ag(Ag(u,h),d)}},e.className="CustomInit",e)}else o=this.biasInitializer;this.bias=this.addWeight("bias",[this.units*4],null,o,this.biasRegularizer,!0,this.biasConstraint)}else this.bias=null;this.built=!0}call(t,e){return V(()=>{const s=e.training==null?!1:e.training;if(t=t,t.length!==3)throw new D(`LSTMCell expects 3 input Tensors (inputs, h, c), got ${t.length}.`);let o=t[1];const r=t[2];t=t[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=Cs({ones:()=>sn(t),rate:this.dropout,training:s,count:4,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=Cs({ones:()=>sn(o),rate:this.recurrentDropout,training:s,count:4,dropoutFunc:this.dropoutFunc}));const i=this.dropoutMask,a=this.recurrentDropoutMask;let l,c,u,h;0<this.dropout&&this.dropout<1&&(t=F(t,i[0]));let d=Mn(t,this.kernel.read());0<this.recurrentDropout&&this.recurrentDropout<1&&(o=F(o,a[0])),d=J(d,Mn(o,this.recurrentKernel.read())),this.useBias&&(d=wn(d,this.bias.read()));const[p,f,m,g]=Ze(d,4,d.rank-1);l=this.recurrentActivation.apply(p),c=this.recurrentActivation.apply(f),u=J(F(c,r),F(l,this.activation.apply(m))),h=this.recurrentActivation.apply(g);const x=F(h,this.activation.apply(u));return[x,x,u]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:bs(this.activation),recurrentActivation:bs(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),unitForgetBias:this.unitForgetBias,kernelRegularizer:Lt(this.kernelRegularizer),recurrentRegularizer:Lt(this.recurrentRegularizer),biasRegularizer:Lt(this.biasRegularizer),activityRegularizer:Lt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation};return Object.assign(Object.assign({},t),e)}}jl.className="LSTMCell",Z(jl);class Rd extends ws{constructor(t){t.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),t.cell=new jl(t),super(t)}call(t,e){return V(()=>{this.cell.dropoutMask!=null&&(It(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(It(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return e.implmentation===0&&(e.implementation=1),new t(e)}}Rd.className="LSTM",Z(Rd);class Ad extends Kl{constructor(t){super(t),this.cells=t.cells}get stateSize(){const t=[];for(const e of this.cells.slice().reverse())Array.isArray(e.stateSize)?t.push(...e.stateSize):t.push(e.stateSize);return t}call(t,e){return V(()=>{t=t;let s=t.slice(1);const o=[];for(const a of this.cells.slice().reverse())Array.isArray(a.stateSize)?o.push(s.splice(0,a.stateSize.length)):o.push(s.splice(0,1));o.reverse();const r=[];let i;for(let a=0;a<this.cells.length;++a){const l=this.cells[a];s=o[a],a===0?i=[t[0]].concat(s):i=[i[0]].concat(s),i=l.call(i,e),r.push(i.slice(1))}s=[];for(const a of r.slice().reverse())s.push(...a);return[i[0]].concat(s)})}build(t){ud(t)&&(t=t[0]),t=t;let e;this.cells.forEach((s,o)=>{Qs(`RNNCell_${o}`,()=>{s.build(t),Array.isArray(s.stateSize)?e=s.stateSize[0]:e=s.stateSize,t=[t[0],e]})}),this.built=!0}getConfig(){const t=super.getConfig(),e=r=>({className:r.getClassName(),config:r.getConfig()}),o={cells:this.cells.map(e)};return Object.assign(Object.assign({},t),o)}static fromConfig(t,e,s={}){const o=[];for(const r of e.cells)o.push(Qn(r,s));return new t({cells:o})}get trainableWeights(){if(!this.trainable)return[];const t=[];for(const e of this.cells)t.push(...e.trainableWeights);return t}get nonTrainableWeights(){const t=[];for(const e of this.cells)t.push(...e.nonTrainableWeights);if(!this.trainable){const e=[];for(const s of this.cells)e.push(...s.trainableWeights);return e.concat(t)}return t}getWeights(){const t=[];for(const e of this.cells)t.push(...e.weights);return hd(t)}setWeights(t){const e=[];for(const s of this.cells){const o=s.weights.length,r=t.splice(o);for(let i=0;i<s.weights.length;++i)e.push([s.weights[i],r[i]])}dd(e)}}Ad.className="StackedRNNCells",Z(Ad);function Cs(n){const{ones:t,rate:e,training:s=!1,count:o=1,dropoutFunc:r}=n,i=()=>r!=null?r(t(),e):Fg(t(),e),a=()=>Ci(i,t,s);return!o||o<=1?Tn(a().clone()):Array(o).fill(void 0).map(a).map(c=>Tn(c.clone()))}var gE=function(n,t){var e={};for(var s in n)Object.prototype.hasOwnProperty.call(n,s)&&t.indexOf(s)<0&&(e[s]=n[s]);if(n!=null&&typeof Object.getOwnPropertySymbols=="function")for(var o=0,s=Object.getOwnPropertySymbols(n);o<s.length;o++)t.indexOf(s[o])<0&&Object.prototype.propertyIsEnumerable.call(n,s[o])&&(e[s[o]]=n[s[o]]);return e};class ib extends ws{constructor(t){if(t.unroll)throw new bt("Unrolling is not possible with convolutional RNNs.");if(Array.isArray(t.cell))throw new bt("It is not possible at the moment to stack convolutional cells.");super(t),this.inputSpec=[new ae({ndim:5})]}call(t,e){return V(()=>{if(this.cell.dropoutMask!=null&&(It(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(It(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null),e&&e.constants)throw new D("ConvRNN2D cell does not support constants");const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}computeOutputShape(t){let e=this.computeSingleOutputShape(t);return this.returnSequences||(e=[e[0],...e.slice(2)]),this.returnState&&(e=[e,...Array(2).fill([t[0],...e.slice(-3)])]),e}getInitialState(t){return V(()=>{const{stateSize:e}=this.cell,s=t.shape,o=this.computeSingleOutputShape(s),r=[o[0],...o.slice(2)],i=pe(r);return Array.isArray(e)?Array(e.length).fill(i):[i]})}resetStates(t,e=!1){V(()=>{if(!this.stateful)throw new On("Cannot call resetStates() on an RNN Layer that is not stateful.");const s=this.inputSpec[0].shape,o=this.computeSingleOutputShape(s),r=[o[0],...o.slice(2)];if(s[0]==null)throw new D("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.getStates()==null)Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>pe(r)):this.states_=[pe(r)];else if(t==null)It(this.states_),this.keptStates!=null&&(It(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>pe(r)):this.states_[0]=pe(r);else{if(Array.isArray(t)||(t=[t]),t.length!==this.states_.length)throw new D(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${t.length} state value(s). Input received: ${t}`);e?this.keptStates.push(this.states_.slice()):It(this.states_);for(let a=0;a<this.states_.length;++a){const l=t[a],c=r;if(!Et(l.shape,c))throw new D(`State ${a} is incompatible with layer ${this.name}: expected shape=${c}, received shape=${l.shape}`);this.states_[a]=l}}this.states_=this.states_.map(a=>Tn(a.clone()))})}computeSingleOutputShape(t){const{dataFormat:e,filters:s,kernelSize:o,padding:r,strides:i,dilationRate:a}=this.cell,l=e==="channelsFirst",c=t[l?3:2],u=t[l?4:3],h=$n(c,o[0],r,i[0],a[0]),d=$n(u,o[1],r,i[1],a[1]);return[...t.slice(0,2),...l?[s,h,d]:[h,d,s]]}}ib.className="ConvRNN2D";class Dd extends jl{constructor(t){const{filters:e,kernelSize:s,strides:o,padding:r,dataFormat:i,dilationRate:a}=t;super(Object.assign(Object.assign({},t),{units:e})),this.filters=e,fe(this.filters,"filters"),this.kernelSize=Po(s,2,"kernelSize"),this.kernelSize.forEach(l=>fe(l,"kernelSize")),this.strides=Po(o||1,2,"strides"),this.strides.forEach(l=>fe(l,"strides")),this.padding=r||"valid",Je(this.padding),this.dataFormat=i||"channelsLast",te(this.dataFormat),this.dilationRate=Po(a||1,2,"dilationRate"),this.dilationRate.forEach(l=>fe(l,"dilationRate"))}build(t){var e;t=Nt(t);const s=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[s]==null)throw new D(`The channel dimension of the input should be defined. Found ${t[s]}`);const o=t[s],r=4,i=this.kernelSize.concat([o,this.filters*r]);this.kernel=this.addWeight("kernel",i,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint);const a=this.kernelSize.concat([this.filters,this.filters*r]);if(this.recurrentKernel=this.addWeight("recurrent_kernel",a,null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias){let l;if(this.unitForgetBias){const c=this.biasInitializer,u=this.filters;l=new(e=class extends ln{apply(d,p){const f=c.apply([u]),m=ds([u]),g=c.apply([u*2]);return td([f,m,g])}},e.className="CustomInit",e)}else l=this.biasInitializer;this.bias=this.addWeight("bias",[this.filters*r],null,l,this.biasRegularizer,!0,this.biasConstraint)}this.built=!0}call(t,e){return V(()=>{if(t.length!==3)throw new D(`ConvLSTM2DCell expects 3 input Tensors (inputs, h, c), got ${t.length}.`);const s=e.training||!1,o=t[0],r=t[1],i=t[2],a=4;0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=Cs({ones:()=>sn(o),rate:this.dropout,training:s,count:a,dropoutFunc:this.dropoutFunc}));const l=this.dropoutMask,c=(K,j,X)=>!j||!j[X]?K:F(j[X],K);let u=c(o,l,0),h=c(o,l,1),d=c(o,l,2),p=c(o,l,3);0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=Cs({ones:()=>sn(r),rate:this.recurrentDropout,training:s,count:a,dropoutFunc:this.dropoutFunc}));const f=this.recurrentDropoutMask;let m=c(r,f,0),g=c(r,f,1),x=c(r,f,2),b=c(r,f,3);const w=3,[y,C,$,N]=Ze(this.kernel.read(),a,w),[T,k,v,I]=this.useBias?Ze(this.bias.read(),a):[null,null,null,null];u=this.inputConv(u,y,T,this.padding),h=this.inputConv(h,C,k,this.padding),d=this.inputConv(d,$,v,this.padding),p=this.inputConv(p,N,I,this.padding);const[R,A,P,_]=Ze(this.recurrentKernel.read(),a,w);m=this.recurrentConv(m,R),g=this.recurrentConv(g,A),x=this.recurrentConv(x,P),b=this.recurrentConv(b,_);const B=this.recurrentActivation.apply(J(u,m)),z=this.recurrentActivation.apply(J(h,g)),W=J(F(z,i),F(B,this.activation.apply(J(d,x)))),H=F(this.recurrentActivation.apply(J(p,b)),this.activation.apply(W));return[H,H,W]})}getConfig(){const t=super.getConfig(),{units:e}=t,s=gE(t,["units"]),o={filters:this.filters,kernelSize:this.kernelSize,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,strides:this.strides};return Object.assign(Object.assign({},s),o)}inputConv(t,e,s,o){const r=Ws(t,e,this.strides,o||"valid",this.dataFormat==="channelsFirst"?"NCHW":"NHWC",this.dilationRate);return s?wn(r,s,this.dataFormat):r}recurrentConv(t,e){return Ws(t,e,1,"same",this.dataFormat==="channelsFirst"?"NCHW":"NHWC")}}Dd.className="ConvLSTM2DCell",Z(Dd);class ab extends ib{constructor(t){const e=new Dd(t);super(Object.assign(Object.assign({},t),{cell:e}))}static fromConfig(t,e){return new t(e)}}ab.className="ConvLSTM2D",Z(ab);class Xl extends Ct{constructor(t){super(t),this.rate=Math.max(Math.min(t.rate,1),0),this.noiseShape=t.noiseShape,this.seed=t.seed,this.supportsMasking=!0}getNoiseShape(t){if(this.noiseShape==null)return this.noiseShape;const e=t.shape,s=[];for(let o=0;o<this.noiseShape.length;++o)s.push(this.noiseShape[o]==null?e[o]:this.noiseShape[o]);return s}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t);if(0<this.rate&&this.rate<1){const o=e.training==null?!1:e.training,r=this.getNoiseShape(s);return Ci(()=>Fg(s,this.rate,r,this.seed),()=>s,o)}return t})}getConfig(){const t={rate:this.rate,noiseShape:this.noiseShape,seed:this.seed},e=super.getConfig();return Object.assign(t,e),t}dispose(){return super.dispose()}}Xl.className="Dropout",Z(Xl);class lb extends Xl{constructor(t){super(t),this.inputSpec=[{ndim:3}]}getNoiseShape(t){const e=t.shape;return[e[0],1,e[2]]}}lb.className="SpatialDropout1D",Z(lb);class Fd extends Ct{constructor(t){if(super(t),this.activation=null,this.useBias=!0,this.kernel=null,this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",t.batchInputShape==null&&t.inputShape==null&&t.inputDim!=null){let e=null;t.batchSize!=null&&(e=t.batchSize),this.batchInputShape=[e,t.inputDim]}this.units=t.units,fe(this.units,"units"),this.activation=ys(t.activation),t.useBias!=null&&(this.useBias=t.useBias),this.kernelInitializer=Ut(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.biasInitializer=Ut(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelConstraint=ce(t.kernelConstraint),this.biasConstraint=ce(t.biasConstraint),this.kernelRegularizer=Gt(t.kernelRegularizer),this.biasRegularizer=Gt(t.biasRegularizer),this.activityRegularizer=Gt(t.activityRegularizer),this.supportsMasking=!0,this.inputSpec=[{minNDim:2}]}build(t){t=Nt(t);const e=t[t.length-1];this.kernel==null&&(this.kernel=this.addWeight("kernel",[e,this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint))),this.inputSpec=[{minNDim:2,axes:{[-1]:e}}],this.built=!0}computeOutputShape(t){t=Nt(t);const e=t.slice();return e[e.length-1]=this.units,e}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t),o=kg(this.activation.getClassName());let r;return o!=null?r=Mn(s,this.kernel.read(),o,this.bias?this.bias.read():null):(r=Mn(s,this.kernel.read()),this.bias!=null&&(r=wn(r,this.bias.read())),this.activation!=null&&(r=this.activation.apply(r))),r})}getConfig(){const t={units:this.units,activation:bs(this.activation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Lt(this.kernelRegularizer),biasRegularizer:Lt(this.biasRegularizer),activityRegularizer:Lt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),biasConstraint:le(this.biasConstraint)},e=super.getConfig();return Object.assign(t,e),t}}Fd.className="Dense",Z(Fd);class cb extends Ct{constructor(t){t=t||{},super(t),this.inputSpec=[{minNDim:3}],this.dataFormat=t.dataFormat}computeOutputShape(t){t=Nt(t);for(const e of t.slice(1))if(e==null)throw new D(`The shape of the input to "Flatten" is not fully defined (got ${t.slice(1)}). Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.`);return[t[0],ms(t,1)]}call(t,e){return V(()=>{this.invokeCallHook(t,e);let s=ft(t);if(this.dataFormat==="channelsFirst"&&s.rank>1){const o=[0];for(let r=2;r<s.rank;++r)o.push(r);o.push(1),s=vt(s,o)}return YN(s)})}getConfig(){const t={};this.dataFormat!=null&&(t.dataFormat=this.dataFormat);const e=super.getConfig();return Object.assign(t,e),t}}cb.className="Flatten",Z(cb);class ub extends Ct{constructor(t){super(t),this.supportsMasking=!0,this.activation=ys(t.activation)}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t);return this.activation.apply(s)})}getConfig(){const t={activation:bs(this.activation)},e=super.getConfig();return Object.assign(t,e),t}}ub.className="Activation",Z(ub);class hb extends Ct{constructor(t){super(t),this.n=t.n,this.inputSpec=[{ndim:2}]}computeOutputShape(t){return[t[0],this.n,t[1]]}call(t,e){return V(()=>(t=ft(t),jN(t,this.n)))}getConfig(){const t={n:this.n},e=super.getConfig();return Object.assign(t,e),t}}hb.className="RepeatVector",Z(hb);class db extends Ct{constructor(t){super(t),this.targetShape=t.targetShape;for(let e=0;e<this.targetShape.length;++e)this.isUnknown(this.targetShape[e])&&(this.targetShape[e]=null)}isUnknown(t){return t<0||t==null}fixUnknownDimension(t,e){const s="Total size of new array must be unchanged.",o=e.slice();let r=1,i=null;for(let l=0;l<o.length;++l){const c=o[l];if(this.isUnknown(c))if(i===null)i=l;else throw new D("Can only specifiy one unknown dimension.");else r*=c}const a=ms(t);if(i!==null){if(r===0||a%r!==0)throw new D(s);o[i]=a/r}else if(a!==r)throw new D(s);return o}computeOutputShape(t){let e=!1;for(let s=0;s<t.length;++s)if(this.isUnknown(t[s])){e=!0;break}return e?t.slice(0,1).concat(this.targetShape):t.slice(0,1).concat(this.fixUnknownDimension(t.slice(1),this.targetShape))}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t),o=s.shape,r=o.slice(0,1).concat(this.fixUnknownDimension(o.slice(1),this.targetShape));return L(s,r)})}getConfig(){const t={targetShape:this.targetShape},e=super.getConfig();return Object.assign(t,e),t}}db.className="Reshape",Z(db);class pb extends Ct{constructor(t){if(super(t),t.dims==null)throw new Error("Required configuration field `dims` is missing during Permute constructor call.");if(!Array.isArray(t.dims))throw new Error(`Permute constructor requires \`dims\` to be an Array, but received ${t.dims} instead.`);const e=bn(1,t.dims.length+1);if(!Et(t.dims.slice().sort(),e))throw new Error("Invalid permutation `dims`: "+JSON.stringify(t.dims)+" `dims` must contain consecutive integers starting from 1.");this.dims=t.dims,this.dimsIncludingBatch=[0].concat(this.dims),this.inputSpec=[new ae({ndim:this.dims.length+1})]}computeOutputShape(t){t=Nt(t);const e=t.slice();return this.dims.forEach((s,o)=>{e[o+1]=t[s]}),e}call(t,e){return vt(ft(t),this.dimsIncludingBatch)}getConfig(){const t={dims:this.dims},e=super.getConfig();return Object.assign(t,e),t}}pb.className="Permute",Z(pb);class fb extends Ct{constructor(t){super(t??{}),this.supportsMasking=!0,t!=null?this.maskValue=t.maskValue==null?0:t.maskValue:this.maskValue=0}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={maskValue:this.maskValue};return Object.assign(e,t),e}computeMask(t,e){const s=ft(t);return Bu(dl(s,this.maskValue),-1)}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t),i=Bu(dl(s,this.maskValue),-1,!0);return F(s,nt(i,s.dtype))})}}fb.className="Masking",Z(fb);class mb extends Ct{constructor(t){if(super(t),this.embeddings=null,this.DEFAULT_EMBEDDINGS_INITIALIZER="randomUniform",t.batchInputShape==null&&t.inputShape==null){let e=null;t.batchSize!=null&&(e=t.batchSize),t.inputLength==null?this.batchInputShape=[e,null]:this.batchInputShape=[e].concat(Dt(t.inputLength))}this.inputDim=t.inputDim,fe(this.inputDim,"inputDim"),this.outputDim=t.outputDim,fe(this.outputDim,"outputDim"),this.embeddingsInitializer=Ut(t.embeddingsInitializer||this.DEFAULT_EMBEDDINGS_INITIALIZER),this.embeddingsRegularizer=Gt(t.embeddingsRegularizer),this.activityRegularizer=Gt(t.activityRegularizer),this.embeddingsConstraint=ce(t.embeddingsConstraint),this.maskZero=t.maskZero,this.supportsMasking=t.maskZero,this.inputLength=t.inputLength}build(t){this.embeddings=this.addWeight("embeddings",[this.inputDim,this.outputDim],this.dtype,this.embeddingsInitializer,this.embeddingsRegularizer,!0,this.embeddingsConstraint),this.built=!0}warnOnIncompatibleInputShape(t){}computeMask(t,e){return V(()=>this.maskZero?(t=ft(t),dl(t,kt(t))):null)}computeOutputShape(t){if(t=Nt(t),this.inputLength==null)return[...t,this.outputDim];const e=Dt(this.inputLength);if(e.length!==t.length-1)throw new D(`"inputLength" is ${this.inputLength}, but received input shape has shape ${t}`);{let s=0;for(let o=0;o<e.length;++o){const r=e[o],i=t[o+1];if(r!=null&&i!=null&&r!==i)throw new D(`"inputLength" is ${this.inputLength}, but received input shape has shape ${t}`);r==null&&(e[s]=i),s++}}return[t[0],...e,this.outputDim]}call(t,e){return V(()=>{this.invokeCallHook(t,e);let s=ft(t);s.dtype!=="int32"&&(s=Ln(s,"int32"));const o=Dg(this.embeddings.read(),L(s,[s.size]));return L(o,Nt(this.computeOutputShape(s.shape)))})}getConfig(){const t={inputDim:this.inputDim,outputDim:this.outputDim,embeddingsInitializer:Kt(this.embeddingsInitializer),embeddingsRegularizer:Lt(this.embeddingsRegularizer),activityRegularizer:Lt(this.activityRegularizer),embeddingsConstraint:le(this.embeddingsConstraint),maskZero:this.maskZero,inputLength:this.inputLength},e=super.getConfig();return Object.assign(t,e),t}}mb.className="Embedding",Z(mb);class so extends Ct{constructor(t){super(t||{}),this.supportsMasking=!0}mergeFunction(t){throw new bt}computeElementwiseOpOutputShape(t,e){if(t==null||e==null)return null;if(t.length<e.length)return this.computeElementwiseOpOutputShape(e,t);if(e.length===0)return t;const s=t.slice(0,t.length-e.length);for(let o=0;o<e.length;++o){const r=t[t.length-e.length+o],i=e[o];if(r==null||i==null||r<0||i<0)s.push(null);else if(r===1)s.push(i);else if(i===1)s.push(r);else{if(r!==i)throw new D("Operands could not be broadcast together with shapes "+JSON.stringify(t)+" "+JSON.stringify(e));s.push(r)}}return s}build(t){if(Array.isArray(t)&&!Array.isArray(t[0])&&(t=[Nt(t)]),t=t,t.length<2)throw new D(`A merge layer should be called on an Array of at least 2 inputs. Got ${t.length} input(s).`);let e=[];for(const r of t)r!=null&&r[0]!==null&&e.push(r[0]);if(e=fs(e),e.length>1)throw new D(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(t)}.`);let s=t[0]==null?null:t[0].slice(1);for(let r=1;r<t.length;++r){const i=t[r]==null?null:t[r].slice(1);s=this.computeElementwiseOpOutputShape(s,i)}const o=t.map(r=>r.length);t.indexOf(null)===-1&&fs(o).length===1?this.reshapeRequired=!1:this.reshapeRequired=!0}call(t,e){return V(()=>{if(t=t,this.reshapeRequired){const s=[],o=t.map(r=>r.rank);if(o.indexOf(null)===-1){const r=gs(o);for(let i of t){const a=i.rank;for(let l=0;l<r-a;++l)i=yi(i,1);s.push(i)}return this.mergeFunction(s)}else{let r=!1;for(const l of t){const c=l.rank;if(c==null){const u=l.shape,h=u[0],d=u.slice(1).concat([h]);let p=L(l,[h].concat(ms(u.slice(1))));p=vt(p,[1,0]),p=L(p,d),s.push(p),r=!0}else if(c>1){const u=bn(1,c).concat([0]);s.push(vt(l,u)),r=!0}else s.push(l)}let i=this.mergeFunction(s);const a=i.rank;if(r){if(a==null){const l=i.shape,c=l.length,u=l[c-1],h=[u].concat(l.slice(0,l.length-1));i=L(vt(L(i,[-1,u]),[1,0]),h)}else if(a>1){const l=[a-1].concat(bn(0,a-1));i=vt(i,l)}}return i}}else return this.mergeFunction(t)})}computeOutputShape(t){t=t;let e;t[0]==null?e=null:e=t[0].slice(1);for(let o=1;o<t.length;++o){const r=t[o]==null?null:t[o].slice(1);e=this.computeElementwiseOpOutputShape(e,r)}let s=[];for(const o of t)o!=null&&o[0]!==null&&s.push(o[0]);return s=fs(s),s.length===1?e=s.concat(e):e=[null].concat(e),e}computeMask(t,e){return V(()=>{if(e==null)return null;if(!Array.isArray(e))throw new D("`mask` should be an Array");if(!Array.isArray(t))throw new D("`inputs` should be an Array");if(e.length!==t.length)throw new D(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${t.length} vs ${e.length})`);if(e.every(o=>o==null))return null;e=e.map(o=>o==null?o:Pe(o,0));let s=e[0];for(let o=1;o<e.length-1;++o)s=Kn(s,e[o]);return s})}}class gb extends so{constructor(t){super(t)}mergeFunction(t){return V(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=J(e,t[s]);return e})}}gb.className="Add",Z(gb);class Od extends so{constructor(t){super(t)}mergeFunction(t){return V(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=F(e,t[s]);return e})}}Od.className="Multiply",Z(Od);class xb extends so{constructor(t){super(t)}mergeFunction(t){return V(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=J(e,t[s]);return F(1/t.length,e)})}}xb.className="Average",Z(xb);class bb extends so{constructor(t){super(t)}mergeFunction(t){return V(()=>{let e=t[0];for(let s=1;s<t.length;++s)e=hs(e,t[s]);return e})}}bb.className="Maximum",Z(bb);class yb extends so{constructor(t){super(t)}mergeFunction(t){return V(()=>{let e=t[0];for(let s=1;s<t.length;++s)e=ci(e,t[s]);return e})}}yb.className="Minimum",Z(yb);class wb extends so{constructor(t){super(t),this.DEFAULT_AXIS=-1,t==null&&(t={}),this.axis=t.axis==null?this.DEFAULT_AXIS:t.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(t){if(!(Array.isArray(t)&&Array.isArray(t[0]))||t.length===1)throw new D("A `Concatenate` layer should be called on a list of at least 2 inputs");t=t;let e=!0;for(const o of t)if(o!=null){e=!1;break}if(e)return;const s=[];for(let o=0;o<t.length;++o){const r=t[o].slice();r.splice(this.axis,1);let i=!1;for(const a of s)if(Et(a,r)){i=!0;break}i||s.push(r)}if(s.length>1)throw new D("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(t))}mergeFunction(t){return V(()=>td(t,this.axis))}computeOutputShape(t){if(!(Array.isArray(t)&&Array.isArray(t[0])))throw new D("A `Concatenate` layer should be called on a list of inputs.");const e=t,s=e[0].slice(),o=this.axis<0?s.length+this.axis:this.axis;for(const r of e.slice(1)){if(s[o]==null||r[o]==null){s[o]=null;break}s[o]+=r[o]}return s}computeMask(t,e){if(e==null)return null;if(!Array.isArray(e))throw new D("`mask` should be an array for Concatenate");if(!Array.isArray(t))throw new D("`inputs` should be an array for Concatenate");if(e.length!==t.length)throw new D(`Mismatch in the length of mask (${e.length}) and the legnth of inputs (${t.length})`);return V(()=>{let s=!0;if(e.forEach(i=>{if(i!=null){s=!1;return}}),s)return null;const o=[];for(let i=0;i<t.length;++i)e[i]==null?o.push(nt(sn(t[i]),"bool")):e[i].rank<t[i].rank?o.push(Pe(e[i],-1)):o.push(e[i]);const r=Me(o,this.axis);return _f(r,-1,!1)})}getConfig(){const t={axis:this.axis},e=super.getConfig();return Object.assign(t,e),t}}wb.className="Concatenate",Z(wb);function Ai(n,t){for(;n<0;)n+=t;return n}function xE(n,t,e){if(n.shape.length>3||t.shape.length>3)throw new bt("batchDot is not implemented for tensors of 4D or higher rank yet");if(S(n.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${n.shape.length}`),S(n.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${t.shape.length}`),typeof e=="number"&&(e=[e,e]),n.dtype==="complex64"||t.dtype==="complex64")throw new bt("batchDot is not implemented for complex64-type Tensors yet.");const s=n.shape.length,o=t.shape.length;e==null&&(e=[s-1,o-2]);const r=e;return V(()=>{let i;if(s>o){i=s-o;const l=[];for(let c=0;c<i;++c)l.push(1);t=L(t,t.shape.concat(l))}else if(o>s){i=o-s;const l=[];for(let c=0;c<i;++c)l.push(1);n=L(n,n.shape.concat(l))}else i=0;let a;if(n.shape.length===2&&t.shape.length===2)r[0]===r[1]?a=ut(F(n,t),r[0]):a=ut(F(vt(n,[1,0]),t),r[1]);else{const l=r[0]!==n.shape.length-1,c=r[1]===t.shape.length-1;a=Rt(n,t,l,c)}if(i>0){let l;s>o?l=s+o-3:l=s-1;const c=[];for(let u=l;u<l+i;++u)c.push(u);a=di(a,c)}return a.shape.length===1&&(a=Pe(a,1)),a})}class Cb extends so{constructor(t){super(t),this.axes=t.axes,this.normalize=t.normalize==null?!1:t.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(t){S(Array.isArray(t)&&t.length===2&&Array.isArray(t[0])&&Array.isArray(t[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const e=t[0],s=t[1];if(e.length>3||s.length>3)throw new bt("Dot layer does not support tensors of 4D or higher rank yet.");const o=this.interpretAxes(e,s);if(e[o[0]]!==s[o[1]])throw new D(`Dimension incompatibility: ${e[o[0]]} !== ${s[o[1]]}`)}mergeFunction(t){if(t.length!==2)throw new D(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${t.length} input(s).`);let e=t[0],s=t[1],o;return Array.isArray(this.axes)?o=this.axes.map((r,i)=>Ai(r,t[i].shape.length)):o=[Ai(this.axes,e.shape.length),Ai(this.axes,s.shape.length)],this.normalize&&(e=Ml(e,o[0]),s=Ml(s,o[1])),xE(e,s,o)}interpretAxes(t,e){let s;return Array.isArray(this.axes)?s=this.axes:s=[Ai(this.axes,t.length),Ai(this.axes,e.length)],s}computeOutputShape(t){S(Array.isArray(t)&&t.length===2&&Array.isArray(t[0])&&Array.isArray(t[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const e=t[0].slice(),s=t[1].slice();if(e.length>3||s.length>3)throw new bt("Dot layer does not support tensors of 4D or higher rank yet.");const o=this.interpretAxes(e,s);e.splice(o[0],1),s.splice(o[1],1),s.splice(0,1);const r=e.concat(s);return r.length===1&&r.push(1),r}computeMask(t,e){return null}getConfig(){const t={axes:this.axes,normalize:this.normalize},e=super.getConfig();return Object.assign(t,e),t}}Cb.className="Dot",Z(Cb);class Ib extends Ct{constructor(t){super(t),this.supportsMasking=!0,this.stddev=t.stddev}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={stddev:this.stddev};return Object.assign(e,t),e}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t);return Ci(()=>J(Al(s.shape,0,this.stddev),s),()=>s,e.training||!1)})}}Ib.className="GaussianNoise",Z(Ib);class $b extends Ct{constructor(t){super(t),this.supportsMasking=!0,this.rate=t.rate}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={rate:this.rate};return Object.assign(e,t),e}call(t,e){return V(()=>{this.invokeCallHook(t,e);const s=ft(t);return this.rate>0&&this.rate<1?Ci(()=>{const r=Math.sqrt(this.rate/(1-this.rate));return F(s,Al(s.shape,1,r))},()=>s,e.training||!1):s})}}$b.className="GaussianDropout",Z($b);class kb extends Ct{constructor(t){super(t),this.supportsMasking=!0,this.rate=t.rate,this.noiseShape=t.noiseShape}_getNoiseShape(t){return this.noiseShape||ft(t).shape}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={rate:this.rate};return Object.assign(e,t),e}call(t,e){return V(()=>{if(this.rate<1&&this.rate>0){const s=this._getNoiseShape(t);return Ci(()=>{const r=ft(t),a=-1.6732632423543772*1.0507009873554805;let l=Gs(ui(s),this.rate);l=Ln(l,"float32");const c=((1-this.rate)*(1+this.rate*a**2))**-.5,u=-c*a*this.rate,h=J(F(r,l),F(J(l,-1),a));return J(F(h,c),u)},()=>ft(t),e.training||!1)}return t})}}kb.className="AlphaDropout",Z(kb);function Di(n,t,e,s,o,r=.001){let i;if(n.rank===2)i=jC(n,t,e,s,o,r);else if(n.rank===3)i=YC(n,t,e,s,o,r);else if(n.rank===4)i=JC(n,t,e,s,o,r);else throw new bt(`batchNormalization is not implemented for array of rank ${n.rank} yet`);return i}function bE(n,t,e,s,o=.001){return V(()=>{const r=th(n,s),i=r.mean,a=r.variance;return[Di(n,i,a,e,t,o),i,a]})}function yE(n,t,e,s,o=.001){return V(()=>{const r=th(n,s),i=r.mean,a=r.variance,l=[];for(const f of bn(0,n.rank))s.indexOf(f)!==-1?l.push(1):l.push(n.shape[f]);const c=L(i,l),u=L(a,l),h=t==null?null:L(t,l),d=e==null?null:L(e,l);return[Di(n,c,u,d,h,o),i,a]})}function wE(n,t,e,s,o=.001){return Et(s.slice().sort(),bn(0,n.rank-1))?bE(n,t,e,s,o):yE(n,t,e,s,o)}class _d extends Ct{constructor(t){t==null&&(t={}),super(t),this.supportsMasking=!0,this.axis=t.axis==null?-1:t.axis,this.momentum=t.momentum==null?.99:t.momentum,this.epsilon=t.epsilon==null?.001:t.epsilon,this.center=t.center==null?!0:t.center,this.scale=t.scale==null?!0:t.scale,this.betaInitializer=Ut(t.betaInitializer||"zeros"),this.gammaInitializer=Ut(t.gammaInitializer||"ones"),this.movingMeanInitializer=Ut(t.movingMeanInitializer||"zeros"),this.movingVarianceInitializer=Ut(t.movingVarianceInitializer||"ones"),this.betaConstraint=ce(t.betaConstraint),this.gammaConstraint=ce(t.gammaConstraint),this.betaRegularizer=Gt(t.betaRegularizer),this.gammaRegularizer=Gt(t.gammaRegularizer)}build(t){t=Nt(t);const e=this.axis>=0?this.axis:this.axis+t.length,s=t[e];if(s==null)throw new D(`Axis ${e} of input tensor should have a defined dimension but the layer received an input with shape ${JSON.stringify(t)}.`);this.inputSpec=[new ae({ndim:t.length,axes:{[e]:s}})];const o=[s];this.scale&&(this.gamma=this.addWeight("gamma",o,null,this.gammaInitializer,this.gammaRegularizer,!0,this.gammaConstraint)),this.center&&(this.beta=this.addWeight("beta",o,null,this.betaInitializer,this.betaRegularizer,!0,this.betaConstraint)),this.movingMean=this.addWeight("moving_mean",o,null,this.movingMeanInitializer,null,!1),this.movingVariance=this.addWeight("moving_variance",o,null,this.movingVarianceInitializer,null,!1),this.built=!0}call(t,e){return V(()=>{const s=e.training==null?!1:e.training,o=ft(t),r=o.shape,i=r.length,a=bn(0,i),l=this.axis>=0?this.axis:this.axis+i;a.splice(l,1);const c=Ys(1,i);c[l]=r[l];const u=a.slice();u.sort();const h=!Et(u,bn(0,i).slice(0,i-1)),d=()=>{if(h){const b=L(this.movingMean.read(),c),w=L(this.movingVariance.read(),c),y=this.center?L(this.beta.read(),c):null,C=this.scale?L(this.gamma.read(),c):null;return Di(o,b,w,y,C,this.epsilon)}else return Di(o,this.movingMean.read(),this.movingVariance.read(),this.beta==null?null:this.beta.read(),this.gamma==null?null:this.gamma.read(),this.epsilon)};if(!s)return d();const[p,f,m]=wE(o,this.gamma.read(),this.beta.read(),a,this.epsilon),g=(b,w,y)=>{V(()=>{const C=1-y,$=b.read(),N=F(pt($,w),C);b.write(pt($,N))})};return g(this.movingMean,f,this.momentum),g(this.movingVariance,m,this.momentum),p})}getConfig(){const t={axis:this.axis,momentum:this.momentum,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Kt(this.betaInitializer),gammaInitializer:Kt(this.gammaInitializer),movingMeanInitializer:Kt(this.movingMeanInitializer),movingVarianceInitializer:Kt(this.movingVarianceInitializer),betaRegularizer:Lt(this.betaRegularizer),gammaRegularizer:Lt(this.gammaRegularizer),betaConstraint:le(this.betaConstraint),gammaConstraint:le(this.gammaConstraint)},e=super.getConfig();return Object.assign(t,e),t}}_d.className="BatchNormalization",Z(_d);class vb extends Ct{constructor(t){if(t==null&&(t={}),super(t),this.axis=t.axis==null?-1:t.axis,typeof this.axis=="number"){if(!Number.isInteger(this.axis))throw new Error(`Expected axis to be an integer, but received ${this.axis}`)}else if(Array.isArray(this.axis)){for(const e of this.axis)if(!Number.isInteger(e))throw new Error(`Expected axis to be an array of integers, but received ${JSON.stringify(this.axis)}`)}else throw new Error(`Expected axis to be an integer or an array of integers, but received ${JSON.stringify(this.axis)}`);this.epsilon=t.epsilon==null?.001:t.epsilon,this.center=t.center==null?!0:t.center,this.scale=t.scale==null?!0:t.scale,this.betaInitializer=Ut(t.betaInitializer||"zeros"),this.gammaInitializer=Ut(t.gammaInitializer||"ones"),this.betaRegularizer=Gt(t.betaRegularizer),this.gammaRegularizer=Gt(t.gammaRegularizer),this.supportsMasking=!0}build(t){t=Nt(t);const e=t.length;typeof this.axis=="number"&&(this.axis=[this.axis]);for(let r=0;r<this.axis.length;++r)this.axis[r]<0&&(this.axis[r]+=e);for(const r of this.axis)if(r<0||r>=e)throw new Error(`Invalid axis: ${r}`);if(this.axis.length!==fs(this.axis).length)throw new Error(`Found duplicate axes in: ${this.axis}`);const s=this.axis.map(r=>t[r]),o=!0;this.scale?this.gamma=this.addWeight("gamma",s,"float32",this.gammaInitializer,this.gammaRegularizer,o):this.gamma=null,this.center?this.beta=this.addWeight("beta",s,"float32",this.betaInitializer,this.betaRegularizer,o):this.beta=null,this.built=!0}call(t,e){const s=ft(t),o=s.shape,r=o.length;return V(()=>{let{mean:a,variance:l}=th(s,this.axis,!0);const c=Ys(1,r);for(const m of this.axis)c[m]=o[m];const u=m=>m!=null&&m.shape.length!==r?L(m,c):m;let h=this.scale?u(this.gamma.read()):null,d=this.center?u(this.beta.read()):null;const p=[],f=[];for(let m=0;m<r;++m)this.axis.indexOf(m)!==-1?(p.push(o[m]),f.push(1)):(p.push(1),f.push(o[m]));return a=gn(a,p),l=gn(l,p),h!=null&&(h=gn(h,f)),d!=null&&(d=gn(d,f)),Di(s,a,l,d,h,this.epsilon)})}getConfig(){const t={axis:this.axis,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Kt(this.betaInitializer),gammaInitializer:Kt(this.gammaInitializer),betaRegularizer:Lt(this.betaRegularizer),gammaRegularizer:Lt(this.gammaRegularizer)},e=super.getConfig();return Object.assign(t,e),t}}vb.className="LayerNormalization",Z(vb);function CE(n,t,e){return V(()=>{if(n.rank!==4)throw new D(`temporalPadding expects input tensor to be 4-D, but received a ${n.rank}-D tensor.`);if(t==null&&(t=[[1,1],[1,1]]),t.length!==2||t[0].length!==2||t[1].length!==2)throw new D("spatial2dPadding expects `padding` to be an Array of two Arrays, each of which is an Array of two integers.");if(e==null&&(e=yn()),e!=="channelsLast"&&e!=="channelsFirst")throw new D(`Unknown data format: ${e}. Supported data formats are 'channelsLast' and 'channelsFirst.`);let s;return e==="channelsFirst"?s=[[0,0],[0,0],t[0],t[1]]:s=[[0,0],t[0],t[1],[0,0]],eh(n,s)})}class Sb extends Ct{constructor(t){if(t==null&&(t={}),super(t),this.dataFormat=t.dataFormat==null?yn():t.dataFormat,t.padding==null)this.padding=[[1,1],[1,1]];else if(typeof t.padding=="number")this.padding=[[t.padding,t.padding],[t.padding,t.padding]];else{if(t.padding=t.padding,t.padding.length!==2)throw new D(`ZeroPadding2D expects padding to be a length-2 array, but received a length-${t.padding.length} array.`);let e,s;if(typeof t.padding[0]=="number")e=[t.padding[0],t.padding[0]],s=[t.padding[1],t.padding[1]];else{if(t.padding=t.padding,t.padding[0].length!==2)throw new D(`ZeroPadding2D expects height padding to be a length-2 array, but received a length-${t.padding[0].length} array.`);if(e=t.padding[0],t.padding[1].length!==2)throw new D(`ZeroPadding2D expects width padding to be a length-2 array, but received a length-${t.padding[1].length} array.`);s=t.padding[1]}this.padding=[e,s]}this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){t=Nt(t);let e,s;return this.dataFormat==="channelsFirst"?(t[2]!=null&&t[2]>=0?e=t[2]+this.padding[0][0]+this.padding[0][1]:e=null,t[3]!=null&&t[3]>=0?s=t[3]+this.padding[1][0]+this.padding[1][1]:s=null,[t[0],t[1],e,s]):(t[1]!=null&&t[1]>=0?e=t[1]+this.padding[0][0]+this.padding[0][1]:e=null,t[2]!=null&&t[2]>=0?s=t[2]+this.padding[1][0]+this.padding[1][1]:s=null,[t[0],e,s,t[3]])}call(t,e){return V(()=>CE(ft(t),this.padding,this.dataFormat))}getConfig(){const t={padding:this.padding,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}Sb.className="ZeroPadding2D",Z(Sb);function Yl(n,t,e,s,o,r){return V(()=>{te(o),Sg(r),Je(s),e==null&&(e=[1,1]),s==null&&(s="valid"),o==null&&(o=yn()),r==null&&(r="max"),n=Nd(n,o);let i;const a=s==="same"?"same":"valid";return r==="max"?i=Qu(n,t,e,a):i=Wu(n,t,e,a),o==="channelsFirst"&&(i=vt(i,[0,3,1,2])),i})}function Nb(n,t,e,s,o,r){return V(()=>{te(o),Sg(r),Je(s),e==null&&(e=[1,1,1]),s==null&&(s="valid"),o==null&&(o=yn()),r==null&&(r="max"),n=Kx(n,o);let i;const a=s==="same"?"same":"valid";return r==="max"?i=L$(n,t,e,a):i=PC(n,t,e,a),o==="channelsFirst"&&(i=vt(i,[0,4,1,2,3])),i})}class Tb extends Ct{constructor(t){if(t.poolSize==null&&(t.poolSize=2),super(t),typeof t.poolSize=="number")this.poolSize=[t.poolSize];else if(Array.isArray(t.poolSize)&&t.poolSize.length===1&&typeof t.poolSize[0]=="number")this.poolSize=t.poolSize;else throw new D(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(t.poolSize)}`);if(fe(this.poolSize,"poolSize"),t.strides==null)this.strides=this.poolSize;else if(typeof t.strides=="number")this.strides=[t.strides];else if(Array.isArray(t.strides)&&t.strides.length===1&&typeof t.strides[0]=="number")this.strides=t.strides;else throw new D(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(t.strides)}`);fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,Je(this.padding),this.inputSpec=[new ae({ndim:3})]}computeOutputShape(t){t=Nt(t);const e=$n(t[1],this.poolSize[0],this.padding,this.strides[0]);return[t[0],e,t[2]]}call(t,e){return V(()=>{this.invokeCallHook(t,e),t=yi(ft(t),2);const s=this.poolingFunction(ft(t),[this.poolSize[0],1],[this.strides[0],1],this.padding,"channelsLast");return di(s,[2])})}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides},e=super.getConfig();return Object.assign(t,e),t}}class Eb extends Tb{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Yl(t,e,s,o,r,"max")}}Eb.className="MaxPooling1D",Z(Eb);class Rb extends Tb{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Yl(t,e,s,o,r,"avg")}}Rb.className="AveragePooling1D",Z(Rb);class Ab extends Ct{constructor(t){if(t.poolSize==null&&(t.poolSize=[2,2]),super(t),this.poolSize=Array.isArray(t.poolSize)?t.poolSize:[t.poolSize,t.poolSize],t.strides==null)this.strides=this.poolSize;else if(Array.isArray(t.strides)){if(t.strides.length!==2)throw new D(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${t.strides.length}.`);this.strides=t.strides}else this.strides=[t.strides,t.strides];fe(this.poolSize,"poolSize"),fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,te(this.dataFormat),Je(this.padding),this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){t=Nt(t);let e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2];return e=$n(e,this.poolSize[0],this.padding,this.strides[0]),s=$n(s,this.poolSize[1],this.padding,this.strides[1]),this.dataFormat==="channelsFirst"?[t[0],t[1],e,s]:[t[0],e,s,t[3]]}call(t,e){return V(()=>(this.invokeCallHook(t,e),this.poolingFunction(ft(t),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class Db extends Ab{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Yl(t,e,s,o,r,"max")}}Db.className="MaxPooling2D",Z(Db);class Fb extends Ab{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Yl(t,e,s,o,r,"avg")}}Fb.className="AveragePooling2D",Z(Fb);class Ob extends Ct{constructor(t){if(t.poolSize==null&&(t.poolSize=[2,2,2]),super(t),this.poolSize=Array.isArray(t.poolSize)?t.poolSize:[t.poolSize,t.poolSize,t.poolSize],t.strides==null)this.strides=this.poolSize;else if(Array.isArray(t.strides)){if(t.strides.length!==3)throw new D(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${t.strides.length}.`);this.strides=t.strides}else this.strides=[t.strides,t.strides,t.strides];fe(this.poolSize,"poolSize"),fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,te(this.dataFormat),Je(this.padding),this.inputSpec=[new ae({ndim:5})]}computeOutputShape(t){t=Nt(t);let e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2],o=this.dataFormat==="channelsFirst"?t[4]:t[3];return e=$n(e,this.poolSize[0],this.padding,this.strides[0]),s=$n(s,this.poolSize[1],this.padding,this.strides[1]),o=$n(o,this.poolSize[2],this.padding,this.strides[2]),this.dataFormat==="channelsFirst"?[t[0],t[1],e,s,o]:[t[0],e,s,o,t[4]]}call(t,e){return V(()=>(this.invokeCallHook(t,e),this.poolingFunction(ft(t),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class _b extends Ob{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Nb(t,e,s,o,r,"max")}}_b.className="MaxPooling3D",Z(_b);class Lb extends Ob{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return te(r),Je(o),Nb(t,e,s,o,r,"avg")}}Lb.className="AveragePooling3D",Z(Lb);class Mb extends Ct{constructor(t){super(t),this.inputSpec=[new ae({ndim:3})]}computeOutputShape(t){return[t[0],t[2]]}call(t,e){throw new bt}}class Ld extends Mb{constructor(t){super(t||{})}call(t,e){return V(()=>{const s=ft(t);return se(s,1)})}}Ld.className="GlobalAveragePooling1D",Z(Ld);class Pb extends Mb{constructor(t){super(t||{})}call(t,e){return V(()=>{const s=ft(t);return mn(s,1)})}}Pb.className="GlobalMaxPooling1D",Z(Pb);class Bb extends Ct{constructor(t){super(t),this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,te(this.dataFormat),this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){return t=t,this.dataFormat==="channelsLast"?[t[0],t[3]]:[t[0],t[1]]}call(t,e){throw new bt}getConfig(){const t={dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class zb extends Bb{call(t,e){return V(()=>{const s=ft(t);return this.dataFormat==="channelsLast"?se(s,[1,2]):se(s,[2,3])})}}zb.className="GlobalAveragePooling2D",Z(zb);class Vb extends Bb{call(t,e){return V(()=>{const s=ft(t);return this.dataFormat==="channelsLast"?mn(s,[1,2]):mn(s,[2,3])})}}Vb.className="GlobalMaxPooling2D",Z(Vb);class Wb extends Ct{constructor(t){super(t),this.layer=t.layer}build(t){this.built=!0}get trainable(){return this.layer!=null?this.layer.trainable:!1}set trainable(t){this.layer!=null&&(this.layer.trainable=t)}get trainableWeights(){return this.layer.trainableWeights}get nonTrainableWeights(){return this.layer.nonTrainableWeights}get updates(){return this.layer._updates}get losses(){return this.layer.losses}getWeights(){return this.layer.getWeights()}setWeights(t){this.layer.setWeights(t)}getConfig(){const t={layer:{className:this.layer.getClassName(),config:this.layer.getConfig()}},e=super.getConfig();return Object.assign(t,e),t}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.layer!=null&&this.layer.setFastWeightInitDuringBuild(t)}static fromConfig(t,e,s={}){const o=e.layer,r=Qn(o,s);delete e.layer;const i={layer:r};return Object.assign(i,e),new t(i)}}class Ub extends Wb{constructor(t){super(t),this.supportsMasking=!0}build(t){if(t=Nt(t),t.length<3)throw new D(`TimeDistributed layer expects an input shape >= 3D, but received input shape ${JSON.stringify(t)}`);this.inputSpec=[{shape:t}];const e=[t[0]].concat(t.slice(2));this.layer.built||(this.layer.build(e),this.layer.built=!0),super.build(t)}computeOutputShape(t){t=Nt(t);const e=[t[0]].concat(t.slice(2)),s=this.layer.computeOutputShape(e),o=t[1];return[s[0],o].concat(s.slice(1))}call(t,e){return V(()=>(t=ft(t),sb((i,a)=>[ft(this.layer.call(i,e)),[]],t,[],!1,null,null,!1,!0)[1]))}}Ub.className="TimeDistributed",Z(Ub);function IE(n){Js(UN,"BidirectionalMergeMode",n)}const $E="concat";class Gb extends Wb{constructor(t){super(t);const e=t.layer.getConfig(),s={};s.className=t.layer.getClassName(),s.config=e,this.forwardLayer=Qn(s),e.goBackwards=e.goBackwards!==!0;const o={};if(o.className=t.layer.getClassName(),o.config=e,this.backwardLayer=Qn(o),this.forwardLayer.name="forward_"+this.forwardLayer.name,this.backwardLayer.name="backward_"+this.backwardLayer.name,this.mergeMode=t.mergeMode===void 0?$E:t.mergeMode,IE(this.mergeMode),t.weights)throw new bt("weights support is not implemented for Bidirectional layer yet.");this._stateful=t.layer.stateful,this.returnSequences=t.layer.returnSequences,this.returnState=t.layer.returnState,this.supportsMasking=!0,this._trainable=!0,this.inputSpec=t.layer.inputSpec,this.numConstants=null}get trainable(){return this._trainable}set trainable(t){this._trainable=t,this.forwardLayer!=null&&(this.forwardLayer.trainable=t),this.backwardLayer!=null&&(this.backwardLayer.trainable=t)}getWeights(){return this.forwardLayer.getWeights().concat(this.backwardLayer.getWeights())}setWeights(t){const e=t.length,s=Math.floor(e/2);this.forwardLayer.setWeights(t.slice(0,s)),this.backwardLayer.setWeights(t.slice(s))}computeOutputShape(t){let e=this.forwardLayer.computeOutputShape(t);Array.isArray(e)&&Array.isArray(e[0])||(e=[e]),e=e;let s,o,r;return this.returnState&&(r=e.slice(1)),s=e[0],s=s,this.mergeMode==="concat"?(s[s.length-1]*=2,o=[s]):this.mergeMode==null?o=[s,s.slice()]:o=[s],this.returnState?this.mergeMode==null?o.concat(r).concat(r.slice()):[s].concat(r).concat(r.slice()):Be(o)}apply(t,e){let s=e==null?null:e.initialState,o=e==null?null:e.constants;e==null&&(e={});const r=nb(t,s,o,this.numConstants);if(t=r.inputs,s=r.initialState,o=r.constants,Array.isArray(t)&&(s=t.slice(1),t=t[0]),(s==null||s.length===0)&&o==null)return super.apply(t,e);const i=[],a=[];if(s!=null){const c=s.length;if(c%2>0)throw new D("When passing `initialState` to a Bidrectional RNN, the state should be an Array containing the states of the underlying RNNs.");e.initialState=s,i.push(...s);const u=s.map(h=>new ae({shape:h.shape}));this.forwardLayer.stateSpec=u.slice(0,c/2),this.backwardLayer.stateSpec=u.slice(c/2),a.push(...u)}if(o!=null)throw new bt("Support for constants in Bidirectional layers is not implemented yet.");const l=i[0]instanceof Pn;for(const c of i)if(c instanceof Pn!==l)throw new D("The initial state of a Bidirectional layer cannot be specified as a mix of symbolic and non-symbolic tensors");if(l){const c=[t].concat(i),u=this.inputSpec.concat(a),h=this.inputSpec;this.inputSpec=u;const d=super.apply(c,e);return this.inputSpec=h,d}else return super.apply(t,e)}call(t,e){return V(()=>{const s=e.initialState;let o,r;if(s==null)o=this.forwardLayer.call(t,e),r=this.backwardLayer.call(t,e);else{const l=s.slice(0,s.length/2),c=s.slice(s.length/2);o=this.forwardLayer.call(t,Object.assign(e,{initialState:l})),r=this.backwardLayer.call(t,Object.assign(e,{initialState:c}))}let i;this.returnState&&(Array.isArray(o)&&(i=o.slice(1).concat(r.slice(1))),o=o[0],r=r[0]),this.returnSequences&&(r=qs(r,1));let a;return this.mergeMode==="concat"?a=td([o,r]):this.mergeMode==="sum"?a=J(o,r):this.mergeMode==="ave"?a=F(.5,J(o,r)):this.mergeMode==="mul"?a=F(o,r):this.mergeMode==null&&(a=[o,r]),this.returnState?this.mergeMode==null?a.concat(i):[a].concat(i):a})}resetStates(t){this.forwardLayer.resetStates(),this.backwardLayer.resetStates()}build(t){Qs(this.forwardLayer.name,()=>{this.forwardLayer.build(t)}),Qs(this.backwardLayer.name,()=>{this.backwardLayer.build(t)}),this.built=!0}computeMask(t,e){Array.isArray(e)&&(e=e[0]);let s;if(this.returnSequences?this.mergeMode==null?s=[e,e]:s=e:this.mergeMode==null?s=[null,null]:s=null,this.returnState){const r=this.forwardLayer.states.map(i=>null);return Array.isArray(s)?s.concat(r).concat(r):[s].concat(r).concat(r)}else return s}get trainableWeights(){return this.forwardLayer.trainableWeights.concat(this.backwardLayer.trainableWeights)}get nonTrainableWeights(){return this.forwardLayer.nonTrainableWeights.concat(this.backwardLayer.nonTrainableWeights)}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.forwardLayer!=null&&this.forwardLayer.setFastWeightInitDuringBuild(t),this.backwardLayer!=null&&this.backwardLayer.setFastWeightInitDuringBuild(t)}getConfig(){const t={mergeMode:this.mergeMode},e=super.getConfig();return Object.assign(t,e),t}static fromConfig(t,e){const s=Qn(e.layer);if(delete e.layer,e.numConstants!=null)throw new bt("Deserialization of a Bidirectional layer with numConstants present is not supported yet.");const o=e;return o.layer=s,new t(o)}}Gb.className="Bidirectional",Z(Gb);class Hb extends Ct{constructor(t){super(t),this.scale=t.scale,t.offset?this.offset=t.offset:this.offset=0}getConfig(){const t={scale:this.scale,offset:this.offset},e=super.getConfig();return Object.assign(t,e),t}call(t,e){return V(()=>(t=ft(t),t.dtype!=="float32"&&(t=Ln(t,"float32")),J(F(t,this.scale),this.offset)))}}Hb.className="Rescaling",Z(Hb);const{resizeBilinear:kE,cropAndResize:vE}=Xn;class qb extends Ct{constructor(t){super(t),this.height=t.height,this.width=t.width}centerCrop(t,e,s,o,r,i,a,l){return V(()=>{let c,u=!1;const h=e/i,d=s/a,p=(o+e)/i,f=(r+s)/a,m=[h,d,p,f],g=[];t.rank===3?(u=!0,c=jn([t])):c=t;for(let C=0;C<c.shape[0];C++)g.push(m);const x=yf(g,[g.length,4]),b=hi(0,g.length,1,"int32"),y=vE(c,x,b,[o,r],"nearest");return Ln(u?ft(js(y)):y,l)})}upsize(t,e,s,o){return V(()=>{const r=kE(t,[e,s]);return Ln(r,o)})}call(t,e){return V(()=>{const s=ft(t),o=s.dtype,r=s.shape,i=r[r.length-3],a=r[r.length-2];let l=0;i!==this.height&&(l=Math.floor((i-this.height)/2));let c=0;return a!==this.width&&(c=Math.floor((a-this.width)/2),c===0&&(c=1)),l>=0&&c>=0?this.centerCrop(s,l,c,this.height,this.width,i,a,o):this.upsize(t,this.height,this.width,o)})}getConfig(){const t={height:this.height,width:this.width},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){t=Nt(t);const e=t.length-3,s=t.length-2;return t[e]=this.height,t[s]=this.width,t}}qb.className="CenterCrop",Z(qb);function SE(n,t,e,s){let o=ft(n);if(o.dtype!=="int32"&&(o=Ln(o,"int32")),t==="int")return o;const r=o.shape;if(o.rank===0&&(o=Pe(o,-1)),t==="oneHot"&&o.shape[o.shape.length-1]!==1&&(o=Pe(o,-1)),o.rank>2)throw new D(`When outputMode is not int, maximum output rank is 2 Received outputMode ${t} and input shape ${r} which would result in output rank ${o.rank}.`);const i=["multiHot","oneHot"].includes(t),a=o;let l;if(typeof s<"u"&&t==="count"?l=Vf(a,s,e,i):l=Vf(a,[],e,i),t!=="tfIdf")return l;if(s)return F(l,s);throw new D("When outputMode is 'tfIdf', weights must be provided.")}class Kb extends Ct{constructor(t){super(t),this.numTokens=t.numTokens,t.outputMode?this.outputMode=t.outputMode:this.outputMode="multiHot"}getConfig(){const t={numTokens:this.numTokens,outputMode:this.outputMode},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){return t=Nt(t),t==null?[this.numTokens]:this.outputMode==="oneHot"&&t[t.length-1]!==1?(t.push(this.numTokens),t):(t[t.length-1]=this.numTokens,t)}call(t,e){return V(()=>{t=ft(t),t.dtype!=="int32"&&(t=Ln(t,"int32"));let s;if(typeof e.countWeights<"u"){if(this.outputMode!=="count")throw new D(`countWeights is not used when outputMode !== count.
              Received countWeights=${e.countWeights}`);s=ft(e.countWeights)}const o=mn(t),r=ll(t),i=Ye(this.numTokens,o).bufferSync().get(0),a=Gs(r,0).bufferSync().get(0);if(!(i&&a))throw new D(`Input values must be between 0 < values <= numTokens with numTokens=${this.numTokens}`);return SE(t,this.outputMode,this.numTokens,s)})}}Kb.className="CategoryEncoding",Z(Kb);const NE=["bilinear","nearest"],jb=new Set(NE);class Xb extends Ct{constructor(t){if(super(t),this.height=t.height,this.width=t.width,t.interpolation)if(jb.has(t.interpolation))this.interpolation=t.interpolation;else throw new D(`Invalid interpolation parameter: ${t.interpolation} is not implemented`);else this.interpolation="bilinear";this.cropToAspectRatio=!!t.cropToAspectRatio}computeOutputShape(t){t=Nt(t);const e=t[2];return[this.height,this.width,e]}getConfig(){const t={height:this.height,width:this.width,interpolation:this.interpolation,cropToAspectRatio:this.cropToAspectRatio},e=super.getConfig();return Object.assign(t,e),t}call(t,e){return V(()=>{const s=[this.height,this.width];if(this.interpolation==="bilinear")return Xn.resizeBilinear(t,s,!this.cropToAspectRatio);if(this.interpolation==="nearest")return Xn.resizeNearestNeighbor(t,s,!this.cropToAspectRatio);throw new Error(`Interpolation is ${this.interpolation} but only ${[...jb]} are supported`)})}}Xb.className="Resizing",Z(Xb);class Yb{constructor(t){this.seed=t}next(){if(this.seed!==void 0)return this.seed++}}Yb.className="RandomSeed";class Zb extends Ct{constructor(t){super(t),this.randomGenerator=new Yb(t.seed)}getConfig(){const t={seed:this.randomGenerator.seed},e=super.getConfig();return Object.assign(t,e),t}}Zb.className="BaseRandomLayer";const TE=["bilinear","nearest"],Jb=new Set(TE);class Qb extends Zb{constructor(t){super(t);const{factor:e,interpolation:s="bilinear"}=t;if(this.factor=e,Array.isArray(this.factor)&&this.factor.length===2)this.widthLower=this.factor[0],this.widthUpper=this.factor[1];else if(!Array.isArray(this.factor)&&this.factor>0)this.widthLower=-this.factor,this.widthUpper=this.factor;else throw new D(`Invalid factor: ${this.factor}. Must be positive number or tuple of 2 numbers`);if(this.widthLower<-1||this.widthUpper<-1)throw new D(`factor must have values larger than -1. Got: ${this.factor}`);if(this.widthUpper<this.widthLower)throw new D(`factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `);if(s)if(Jb.has(s))this.interpolation=s;else throw new D(`Invalid interpolation parameter: ${s} is not implemented`)}getConfig(){const t={factor:this.factor,interpolation:this.interpolation},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){t=Nt(t);const e=t[2];return[this.imgHeight,-1,e]}call(t,e){return V(()=>{const s=ft(t);this.imgHeight=s.shape[s.shape.length-3];const o=s.shape[s.shape.length-2];this.widthFactor=ui([1],1+this.widthLower,1+this.widthUpper,"float32",this.randomGenerator.next());let r=this.widthFactor.dataSync()[0]*o;r=Math.round(r);const i=[this.imgHeight,r];switch(this.interpolation){case"bilinear":return Xn.resizeBilinear(t,i);case"nearest":return Xn.resizeNearestNeighbor(t,i);default:throw new Error(`Interpolation is ${this.interpolation}
          but only ${[...Jb]} are supported`)}})}}Qb.className="RandomWidth",Z(Qb);function Fi(n){return new Fd(n)}function EE(n){return new Xl(n)}function RE(n){return new Od(n)}function AE(n){return new _d(n)}function DE(n){return new Ld(n)}function t0(n){return new Rd(n)}function FE(n){return dE(n)}U().registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,n=>{n&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});var e0;(function(n){n[n.DT_INVALID=0]="DT_INVALID",n[n.DT_FLOAT=1]="DT_FLOAT",n[n.DT_DOUBLE=2]="DT_DOUBLE",n[n.DT_INT32=3]="DT_INT32",n[n.DT_UINT8=4]="DT_UINT8",n[n.DT_INT16=5]="DT_INT16",n[n.DT_INT8=6]="DT_INT8",n[n.DT_STRING=7]="DT_STRING",n[n.DT_COMPLEX64=8]="DT_COMPLEX64",n[n.DT_INT64=9]="DT_INT64",n[n.DT_BOOL=10]="DT_BOOL",n[n.DT_QINT8=11]="DT_QINT8",n[n.DT_QUINT8=12]="DT_QUINT8",n[n.DT_QINT32=13]="DT_QINT32",n[n.DT_BFLOAT16=14]="DT_BFLOAT16",n[n.DT_QINT16=15]="DT_QINT16",n[n.DT_QUINT16=16]="DT_QUINT16",n[n.DT_UINT16=17]="DT_UINT16",n[n.DT_COMPLEX128=18]="DT_COMPLEX128",n[n.DT_HALF=19]="DT_HALF",n[n.DT_RESOURCE=20]="DT_RESOURCE",n[n.DT_VARIANT=21]="DT_VARIANT",n[n.DT_UINT32=22]="DT_UINT32",n[n.DT_UINT64=23]="DT_UINT64",n[n.DT_FLOAT_REF=101]="DT_FLOAT_REF",n[n.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",n[n.DT_INT32_REF=103]="DT_INT32_REF",n[n.DT_UINT8_REF=104]="DT_UINT8_REF",n[n.DT_INT16_REF=105]="DT_INT16_REF",n[n.DT_INT8_REF=106]="DT_INT8_REF",n[n.DT_STRING_REF=107]="DT_STRING_REF",n[n.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",n[n.DT_INT64_REF=109]="DT_INT64_REF",n[n.DT_BOOL_REF=110]="DT_BOOL_REF",n[n.DT_QINT8_REF=111]="DT_QINT8_REF",n[n.DT_QUINT8_REF=112]="DT_QUINT8_REF",n[n.DT_QINT32_REF=113]="DT_QINT32_REF",n[n.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",n[n.DT_QINT16_REF=115]="DT_QINT16_REF",n[n.DT_QUINT16_REF=116]="DT_QUINT16_REF",n[n.DT_UINT16_REF=117]="DT_UINT16_REF",n[n.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",n[n.DT_HALF_REF=119]="DT_HALF_REF",n[n.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",n[n.DT_VARIANT_REF=121]="DT_VARIANT_REF",n[n.DT_UINT32_REF=122]="DT_UINT32_REF",n[n.DT_UINT64_REF=123]="DT_UINT64_REF"})(e0||(e0={}));var n0;(function(n){(function(t){t[t.LEGACY=0]="LEGACY",t[t.V1=1]="V1",t[t.V2=2]="V2"})(n.CheckpointFormatVersion||(n.CheckpointFormatVersion={}))})(n0||(n0={}));var s0;(function(n){n[n.FAIL=0]="FAIL",n[n.SHORTEST=1]="SHORTEST",n[n.LONGEST=2]="LONGEST"})(s0||(s0={}));function rt(n,t){Array.isArray(n)||(n=[n]),n.forEach(e=>{e!=null&&S(e.dtype!=="complex64",()=>`${t} does not support complex64 tensors in the CPU backend.`)})}const OE=bm;class Zl extends $c{nextDataId(){return Zl.nextDataId++}constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new gp(this,Nn())}write(t,e,s){this.firstUse&&(this.firstUse=!1,U().get("IS_NODE")&&je(`
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================`));const o={id:this.nextDataId()};return this.data.set(o,{values:t,dtype:s,refCount:1}),o}makeTensorInfo(t,e,s){let o;if(e==="string"&&s!=null&&s.length>0&&er(s[0])){const r=s.map(i=>is(i));o=this.write(r,t,e)}else o=this.write(s,t,e);return{dataId:o,shape:t,dtype:e}}refCount(t){return this.data.has(t)?this.data.get(t).refCount:0}incRef(t){const e=this.data.get(t);e.refCount++}decRef(t){if(this.data.has(t)){const e=this.data.get(t);e.refCount--}}move(t,e,s,o,r){this.data.set(t,{values:e,dtype:o,refCount:r})}numDataIds(){return this.data.numDataIds()}async read(t){return this.readSync(t)}readSync(t){const{dtype:e,complexTensorInfos:s}=this.data.get(t);if(e==="complex64"){const o=this.readSync(s.real.dataId),r=this.readSync(s.imag.dataId);return Yn(o,r)}return Zy(this.data.get(t).values,e)}bufferSync(t){const e=this.readSync(t.dataId);if(t.dtype==="string")try{const s=e.map(o=>as(o));return wt(t.shape,t.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return wt(t.shape,t.dtype,e)}makeOutput(t,e,s){return Nn().makeTensorFromTensorInfo(this.makeTensorInfo(e,s,t),this)}disposeData(t,e=!1){if(this.data.has(t)){if(this.data.get(t).refCount--,!e&&this.data.get(t).refCount>0)return!1;const{complexTensorInfos:s}=this.data.get(t);s!=null&&(this.disposeData(s.real.dataId,!0),this.disposeData(s.imag.dataId,!0)),this.data.delete(t)}return!0}disposeIntermediateTensorInfo(t){this.disposeData(t.dataId)}async time(t){const e=Oe();return t(),{kernelMs:Oe()-e}}memory(){return{unreliable:!0,reasons:["The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less."]}}where(t){rt([t],"where");const e=this.readSync(t.dataId);return OE(t.shape,e)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}}Zl.nextDataId=0;function o0(n){const t=new Float32Array(n.length);for(let e=0;e<n.length;++e)t[e]=Math.abs(n[e]);return t}const _E={kernelName:Zi,backendName:"cpu",kernelFunc:n=>{const{x:t}=n.inputs,e=n.backend;rt(t,"abs");let s=new Float32Array(q(t.shape));const o=e.data.get(t.dataId).values;return s=o0(o),e.makeOutput(s,t.shape,t.dtype)}};function ee(n){return(t,e,s,o,r)=>{const i=mt(t,e),a=i.length,l=lt(i),c=q(i),u=Ce(r,c),h=t.length,d=e.length,p=lt(t),f=lt(e),m=Ro(t,i),g=Ro(e,i);if(m.length+g.length===0)for(let x=0;x<u.length;++x)u[x]=n(s[x%s.length],o[x%o.length]);else for(let x=0;x<u.length;++x){const b=wo(x,a,l),w=b.slice(-h);m.forEach(N=>w[N]=0);const y=Sn(w,h,p),C=b.slice(-d);g.forEach(N=>C[N]=0);const $=Sn(C,d,f);u[x]=n(s[y],o[$])}return[u,i]}}function qe(n){const{inputs:t,backend:e}=n,{real:s,imag:o}=t,r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,a=e.makeTensorInfo(s.shape,"complex64"),l=e.data.get(a.dataId);return l.complexTensorInfos={real:e.makeTensorInfo(s.shape,"float32",r),imag:e.makeTensorInfo(o.shape,"float32",i)},a}const LE={kernelName:Wc,backendName:"cpu",kernelFunc:qe};function Jl(n,t,e="float32"){if(e==="complex64"){const o=Jl(n,t,"float32"),r=Jl(n,t,"float32");return qe({inputs:{real:o,imag:r},backend:n})}const s=Ie(q(t),e);return n.makeTensorInfo(t,e,s)}function zn(n){const{inputs:t,backend:e}=n,{x:s}=t;return e.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const ME={kernelName:Ir,backendName:"cpu",kernelFunc:zn};function oo(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.data.get(s.dataId).complexTensorInfos.real,r=e.data.get(o.dataId).values;return e.makeTensorInfo(o.shape,o.dtype,r)}const PE={kernelName:fu,backendName:"cpu",kernelFunc:oo};function r0(n,t,e,s){if(s==="int32"){const o=Int32Array.from(n);return[t,"int32",o]}if(s==="bool"){const o=Os([0],e),[r,i]=ee((a,l)=>a!==l?1:0)(t,[],n,o,"bool");return[i,"bool",r]}throw new Error(`Error in Cast: failed to cast ${e} to ${s}`)}function Is(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return zn({inputs:{x:o},backend:e});const u=Jl(e,o.shape,o.dtype),h=Is({inputs:{x:o},backend:e,attrs:{dtype:"float32"}}),d=qe({inputs:{real:h,imag:u},backend:e});return e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),d}if(o.dtype==="complex64"){const u=oo({inputs:{input:o},backend:e}),h=Is({inputs:{x:u},backend:e,attrs:{dtype:r}});return e.disposeIntermediateTensorInfo(u),h}if(!yp(o.dtype,r)){const u=zn({inputs:{x:o},backend:e});return{dataId:u.dataId,shape:u.shape,dtype:r}}const i=e.data.get(o.dataId).values,[a,l,c]=r0(i,o.shape,o.dtype,r);return e.makeTensorInfo(a,l,c)}const BE={kernelName:cr,backendName:"cpu",kernelFunc:Is};function ue(n,t,e,s){return e==null?({inputs:o,backend:r})=>{const{a:i,b:a}=o,l=r;rt([i,a],n);const c=l.data.get(i.dataId).values,u=l.data.get(a.dataId).values,h=i.dtype==="string"?Zn(c):c,d=i.dtype==="string"?Zn(u):u,p=s||i.dtype,[f,m]=t(i.shape,a.shape,h,d,p);return l.makeTensorInfo(m,p,f)}:({inputs:o,backend:r})=>{const{a:i,b:a}=o,l=r;if(i.dtype==="complex64"||a.dtype==="complex64"){const c=Is({inputs:{x:i},backend:l,attrs:{dtype:"complex64"}}),u=l.data.get(c.dataId),h=u.complexTensorInfos.real,d=u.complexTensorInfos.imag,p=l.data.get(h.dataId).values,f=l.data.get(d.dataId).values,m=Is({inputs:{x:a},backend:l,attrs:{dtype:"complex64"}}),g=l.data.get(m.dataId),x=g.complexTensorInfos.real,b=g.complexTensorInfos.imag,w=l.data.get(x.dataId).values,y=l.data.get(b.dataId).values,[C,$,N]=e(i.shape,a.shape,p,f,w,y),T=l.makeTensorInfo(N,"float32",C),k=l.makeTensorInfo(N,"float32",$),v=qe({inputs:{real:T,imag:k},backend:l});return l.disposeIntermediateTensorInfo(c),l.disposeIntermediateTensorInfo(m),l.disposeIntermediateTensorInfo(T),l.disposeIntermediateTensorInfo(k),v}else{const c=l.data.get(i.dataId).values,u=l.data.get(a.dataId).values,h=s||i.dtype,[d,p]=t(i.shape,a.shape,c,u,h);return l.makeTensorInfo(p,h,d)}}}function Md(n){return(t,e,s,o,r,i)=>{const a=mt(t,e),l=q(a),c=a.length,u=lt(a),h=Ce("float32",l),d=Ce("float32",l),p=Ro(t,a),f=Ro(e,a),m=Yn(s,o),g=Yn(r,i),x=t.length,b=lt(t),w=e.length,y=lt(e);if(p.length+f.length===0)for(let C=0;C<h.length;C++){const $=C%m.length,N=C%g.length,T=n(m[$*2],m[$*2+1],g[N*2],g[N*2+1]);h[C]=T.real,d[C]=T.imag}else for(let C=0;C<h.length;C++){const $=wo(C,c,u),N=$.slice(-x);p.forEach(R=>N[R]=0);const T=Sn(N,x,b),k=$.slice(-w);f.forEach(R=>k[R]=0);const v=Sn(k,w,y),I=n(m[T*2],m[T*2+1],g[v*2],g[v*2+1]);h[C]=I.real,d[C]=I.imag}return[h,d,a]}}const i0=ee(((n,t)=>n+t)),zE=Md(((n,t,e,s)=>({real:n+e,imag:t+s}))),zo=ue(Co,i0,zE),VE={kernelName:Co,backendName:"cpu",kernelFunc:zo};function Pd(n,t,e,s,o){const r=q(s),i=Ie(o,e);for(let a=0;a<n.length;a++){const l=n[a];if(l<0)throw new Error("Input x must be non-negative!");l>=o||(r>0?i[l]+=t[a]:i[l]+=1)}return i}function a0(n,t,e,s=!1){const o=n.shape[0],r=n.shape[1],i=wt([o,e],t.dtype);for(let a=0;a<o;a++)for(let l=0;l<r;l++){const c=n.get(a,l);if(c<0)throw new Error("Input x must be non-negative!");c>=e||(s?i.set(1,a,c):t.size>0?i.set(i.get(a,c)+t.get(a,l),a,c):i.set(i.get(a,c)+1,a,c))}return i}const l0=ee(((n,t)=>n&t)),WE=ue(Vc,l0),UE={kernelName:Vc,backendName:"cpu",kernelFunc:WE};function Vn(n){return(t,e,s)=>{const o=Yt(e,t.length);for(let r=0;r<t.length;++r)o[r]=n(t[r],s);return o}}function Ft(n,t,e){const s=Vn(t);return $s(n,s,e)}function $s(n,t,e){return({inputs:s,attrs:o,backend:r})=>{const{x:i}=s;rt(i,n);const a=r,l=a.data.get(i.dataId).values;let c;if(i.dtype==="string"){if(!Array.isArray(l))throw new Error("String tensor's value was not an instance of Array");c=Zn(l)}else c=l;const u=e||i.dtype,h=t(c,u,o);return a.makeTensorInfo(i.shape,u,h)}}const c0=Vn(n=>Math.ceil(n)),GE=$s(ur,c0),HE={kernelName:ur,backendName:"cpu",kernelFunc:GE};function u0(n,t,e,s){const o=Yt(e,q(t));if(s&&e!=="string"){let r=0;n.forEach(i=>{const a=q(i.shape);o.set(i.vals,r),r+=a})}else{let r=0;n.forEach(i=>{const a=e==="string"?Zn(i.vals):i.vals;let l=0;for(let c=0;c<i.shape[0];++c){const u=c*t[1]+r;for(let h=0;h<i.shape[1];++h)o[u+h]=a[l++]}r+=i.shape[1]})}return o}const h0=ee((n,t)=>n===t?1:0),d0=ue(da,h0,null,"bool"),qE={kernelName:da,backendName:"cpu",kernelFunc:d0};const p0=Vn(n=>Math.exp(n)),f0=$s(xr,p0,"float32"),KE={kernelName:xr,backendName:"cpu",kernelFunc:f0};const m0=Vn(n=>Math.expm1(n)),jE=$s(br,m0),XE={kernelName:br,backendName:"cpu",kernelFunc:jE};const g0=Vn(n=>Math.floor(n)),YE=$s(yr,g0),ZE={kernelName:yr,backendName:"cpu",kernelFunc:YE};const x0=ee((n,t)=>Math.floor(n/t)),JE=ue(wr,x0,null,"int32"),QE={kernelName:wr,backendName:"cpu",kernelFunc:JE};function b0(n,t,e,s,o,r,i,a,l){const c=wt([s,r],e);for(let u=0;u<s;u++){const h=[];let d=0;for(let p=0;p<o;p++){const f=n[u*o+p];d+=f*i[p],h.push(f)}if(d<0||d>=l/r)throw new Error(`Invalid indices: ${h} does not index into ${a}`);for(let p=0;p<r;p++)c.values[u*r+p]=t.get(...t.indexToLoc(d*r+p))}return c}function y0(n,t,e){const s=wt(e,n.dtype);for(let o=0;o<s.size;++o){const i=s.indexToLoc(o).slice(),a=i[0],l=i[2],c=t.locToIndex([a,l]);i[2]=t.values[c];const u=n.locToIndex(i);0<=u&&u<n.values.length&&(s.values[o]=n.values[u])}return s}const w0=ee((n,t)=>n>t?1:0),tR=ue(ga,w0,null,"bool"),eR={kernelName:ga,backendName:"cpu",kernelFunc:tR};const C0=ee((n,t)=>n>=t?1:0),nR=ue(Cr,C0,null,"bool"),sR={kernelName:Cr,backendName:"cpu",kernelFunc:nR};const I0=ee((n,t)=>n<t?1:0),oR=ue(ba,I0,null,"bool"),rR={kernelName:ba,backendName:"cpu",kernelFunc:oR};const $0=ee((n,t)=>n<=t?1:0),iR=ue(ya,$0,null,"bool"),aR={kernelName:ya,backendName:"cpu",kernelFunc:iR};function k0(n,t,e){const s=(t-n)/(e-1),o=Ie(e,"float32");o[0]=n;for(let r=1;r<o.length;r++)o[r]=o[r-1]+s;return o}const v0=Vn(n=>Math.log(n)),lR=$s(Sr,v0),cR={kernelName:Sr,backendName:"cpu",kernelFunc:lR};function S0(n,t,e,s){const o=Ce(s,q(e));for(let r=0;r<o.length;++r){const i=r*t;let a=n[i];for(let l=0;l<t;++l){const c=n[i+l];(Number.isNaN(c)||c>a)&&(a=c)}o[r]=a}return o}const N0=ee(((n,t)=>Math.max(n,t))),uR=ue(Tr,N0),hR={kernelName:Tr,backendName:"cpu",kernelFunc:uR};const T0=ee(((n,t)=>Math.min(n,t))),dR=ue(Er,T0),pR={kernelName:Er,backendName:"cpu",kernelFunc:dR};const Bd=ee(((n,t)=>n*t)),fR=Md(((n,t,e,s)=>({real:n*e-t*s,imag:n*s+t*e}))),Ql=ue(Ar,Bd,fR),mR={kernelName:Ar,backendName:"cpu",kernelFunc:Ql};function E0(n,t,e){const s=rs(-1,e);return Bd([],t,s,n,e)}function gR(n){const{inputs:t,backend:e}=n,{x:s}=t;rt(s,"neg");const o=e.data.get(s.dataId).values,[r,i]=E0(o,s.shape,s.dtype);return e.makeTensorInfo(i,s.dtype,r)}const xR={kernelName:Ra,backendName:"cpu",kernelFunc:gR};const R0=ee(((n,t)=>n!==t?1:0)),bR=ue(Aa,R0,null,"bool"),yR={kernelName:Aa,backendName:"cpu",kernelFunc:bR};function zd(n,t,e,s,o){const r=t.length,i=q(t),a=lt(t),l=lt(o),c=Ce(e,q(o));for(let u=0;u<i;++u){const h=wo(u,r,a),d=new Array(h.length);for(let f=0;f<d.length;f++)d[f]=h[s[f]];const p=Sn(d,r,l);c[p]=n[u]}return c}function ze(n){const{inputs:t,attrs:e,backend:s}=n,{x:o}=t,{perm:r}=e;rt(o,"transpose");const i=o.shape.length,a=new Array(i);for(let h=0;h<a.length;h++)a[h]=o.shape[r[h]];const l=s.data.get(o.dataId).values,c=zd(l,o.shape,o.dtype,r,a);return{dataId:s.write(c,a,o.dtype),shape:a,dtype:o.dtype}}const wR={kernelName:Io,backendName:"cpu",kernelFunc:ze};function A0(n,t,e,s){const[o,r]=de(n,s),i=Ue(t,"int32"),a=Ie(q(o),i),l=q(r);for(let c=0;c<a.length;++c){const u=c*l;let h=1;for(let d=0;d<l;++d)h*=e[u+d];a[c]=h}return{outVals:a,outShape:o,outDtype:i}}function CR(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;rt(o,"prod");const a=o.shape.length,l=yt(r,o.shape),c=qt(l,a);let u=l,h=o;const d=[];c!=null&&(h=ze({inputs:{x:o},backend:e,attrs:{perm:c}}),d.push(h),u=Jt(u.length,a));const p=e.data.get(h.dataId).values,{outVals:f,outShape:m,outDtype:g}=A0(h.shape,h.dtype,p,u);let x=m;return i&&(x=ne(m,l)),d.forEach(b=>e.disposeIntermediateTensorInfo(b)),e.makeTensorInfo(x,g,f)}const IR={kernelName:Ma,backendName:"cpu",kernelFunc:CR};function $R(n,t,e){n.forEach((s,o)=>{if(s<0||s>=e){const r=wo(o,t.length,lt(t)).join(",");throw new Error(`indices[${r}] = ${s} is not in [0, ${e})`)}})}function kR(n,t){for(let e=0;e<n.length;++e){const s=n[e],o=e===n.length-1?t:n[e+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>o)throw new Error("Ragged splits must not point past values");for(let r=1;r<s.length;++r)if(s[r-1]>s[r])throw new Error("Ragged splits must be sorted in ascending order")}}function vR(n,t,e,s){const o=[];let r=0;const i=t.length-1+e.length,a=new Array(i).fill(null).map(()=>[0]);kR(e,s);let l=1;for(let c=0;c<t.length-1;++c){l*=t[c];const u=t[c+1];for(let h=1;h<l+1;++h)a[c].push(h*u)}for(let c=0;c<n.length;++c){let u=n[c],h=n[c]+1;for(let d=0;d<e.length;++d){const p=e[d],f=d+t.length-1;if(f>=0){const m=a[f],g=m[m.length-1]-p[u];for(let x=u;x<h;++x)a[f].push(p[x+1]+g)}u=p[u],h=p[h]}h!==u&&(o.push([u,h]),r+=h-u)}return{outSplits:a,valueSlices:o,numValues:r}}function SR(n){const t=[];for(let e=0;e<n.length;++e){const s=n[e].length,o=Yt("int32",s);t.push(o),n[e].forEach((r,i)=>o[i]=r)}return t}function D0(n,t){const e=n.slice(0,t);for(;e.length<t;)e.push(1);for(let s=t;s<n.length;s++)e[t-1]*=n[s];return e}function NR(n,t,e,s,o,r){const i=D0(t,2)[1],a=D0(r,2)[1];let l=0;for(const c of e)for(let u=c[0];u<c[1];++u){for(let h=0;h<s;++h)o[l*a+h]=n[u*i+h];++l}}function TR(n,t,e,s,o){const r=t.slice();r[0]=o;const i=Yt(e,q(r)),a=n.length,l=a===0?0:a/t[0];return NR(n,t,s,l,i,r),[i,r]}function F0(n,t,e,s,o,r,i,a){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(t[0].length===0)throw new Error("Split tensors must not be scalars");const l=t[0][0]-1;if($R(r,i,l),s.length===0)throw new Error("params.rank must be nonzero");const c=s[0],{outSplits:u,valueSlices:h,numValues:d}=vR(r,i,n,c),p=SR(u),f=TR(e,s,o,h,d);return[p,f[0],f[1]]}const O0=2147483647;function _0(n,t,e,s,o,r,i){if(t.length>1)throw new Error("starts must be a scalar or vector");if(o.length>1)throw new Error("limits must be a scalar or vector");if(i.length>1)throw new Error("deltas must be a scalar or vector");const a=t.length===0,l=o.length===0,c=i.length===0,u=[];a||u.push(t[0]),l||u.push(o[0]),c||u.push(i[0]);for(let g=1;g<u.length;++g)if(u[g]!==u[g-1])throw new Error("starts, limits, and deltas must have the same shape");const h=u.length===0?1:u[0],d=Yt("int32",h+1);d[0]=0;for(let g=0;g<h;++g){const x=a?n[0]:n[g],b=l?s[0]:s[g],w=c?r[0]:r[g];if(w===0)throw new Error("Requires delta != 0");let y;if(w>0&&b<x||w<0&&b>x)y=0;else if(y=Math.ceil(Math.abs((b-x)/w)),y>O0)throw new Error(`Requires ((limit - start) / delta) <= ${O0}`);d[g+1]=d[g]+y}const p=d[h],f=Yt(e,p);let m=0;for(let g=0;g<h;++g){const x=d[g+1]-d[g];let b=a?n[0]:n[g];const w=c?r[0]:r[g];for(let y=0;y<x;++y)f[m++]=b,b+=w}return[d,f]}var un=xn;class tc{constructor(t,e,s,o,r,i,a,l,c,u){this.shape=t,this.shapeShape=e,this.values=s,this.valuesShape=o,this.valuesDType=r,this.defaultValue=i,this.defaultValueShape=a,this.rowPartitionValues=l,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=Wm(u),this.raggedRank=Um(this.rowPartitionTypes)}getRowPartitionTypeByDimension(t){return this.rowPartitionTypes[0]===un.FIRST_DIM_SIZE?this.rowPartitionTypes[t+1]:this.rowPartitionTypes[t]}getRowPartitionTensor(t){return this.rowPartitionTypes[0]===un.FIRST_DIM_SIZE?this.rowPartitionValues[t+1]:this.rowPartitionValues[t]}getMaxWidth(t){const e=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case un.VALUE_ROWIDS:return tc.getMaxWidthValueRowID(e);case un.ROW_SPLITS:return tc.getMaxWidthRowSplit(e);default:throw new Error(`Cannot handle partition type ${un[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(t){const e=t.length;if(e===0||e===1)return 0;let s=0;for(let o=0;o<e-1;++o){const r=t[o+1]-t[o];r>s&&(s=r)}return s}static getMaxWidthValueRowID(t){const e=t.length;if(e===0)return 0;let s=0,o=t[0],r=0;for(let i=1;i<e;++i){const a=t[i];a!==o&&(o=a,r=Math.max(i-s,r),s=i)}return Math.max(e-s,r)}tensorShapeFromTensor(t,e,s=!0){if(e.length===0){if(t[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return M0(t,s)}calculateOutputSize(t){const e=this.valuesShape,s=this.defaultValueShape;Gm(s,e);const o=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=Vm(this.raggedRank,o,e);i[0]<0&&(i[0]=t);for(let a=1;a<=this.raggedRank;++a)i[a]<0&&(i[a]=this.getMaxWidth(a));return i}calculateFirstParentOutputIndex(t,e,s){const o=Math.min(t,s),r=[];let i=0;for(let a=0;a<o;++a,i+=e)r.push(i);for(let a=o;a<t;++a)r.push(-1);return S(r.length===t,()=>"Final length of result must be equal to firstDimension."),r}calculateOutputIndexRowSplit(t,e,s,o){const r=t.length,i=[];for(let a=0;a<r-1;++a){const l=t[a+1]-t[a];let c=Math.min(o,l),u=e[a];u===-1&&(c=0);for(let h=0;h<c;++h)i.push(u),u+=s;for(let h=0;h<l-c;++h)i.push(-1)}if(r>0&&i.length!==t[r-1])throw new Error("Invalid row split size.");return i}calculateOutputIndexValueRowID(t,e,s,o){const r=t.length,i=[];if(r===0)return[];let a=0,l=t[0];if(l>=e.length)throw new Error(`Got currentValueRowId=${l}, which is not less than ${e.length}`);let c=e[l];i.push(c);for(let u=1;u<r;++u){const h=t[u];if(h===l)c>=0&&(++a,a<o?c+=s:c=-1);else{if(a=0,l=h,h>=e.length)throw new Error(`Got nextValueRowId=${h} which is not less than ${e.length}`);c=e[h]}i.push(c)}if(i.length!==t.length)throw new Error("Invalid row ids.");return i}calculateOutputIndex(t,e,s,o){const r=this.getRowPartitionTensor(t),i=this.getRowPartitionTypeByDimension(t);switch(i){case un.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(r,e,s,o);case un.ROW_SPLITS:if(r.length-1>e.length)throw new Error(`Row partition size is greater than output size: ${r.length-1} > ${e.length}`);return this.calculateOutputIndexRowSplit(r,e,s,o);default:throw new Error(`Unsupported partition type: ${un[i]}`)}}getFirstDimensionSize(){const t=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const e=this.rowPartitionTypes[0];switch(e){case un.FIRST_DIM_SIZE:return t[0];case un.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case un.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${un[e]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const e=this.getFirstDimensionSize(),s=this.calculateOutputSize(e),o=new Array(this.raggedRank+1);o[o.length-1]=1;for(let l=o.length-2;l>=0;--l)o[l]=o[l+1]*s[l+1];const r=M0(s,!1),i=Yt(this.valuesDType,q(r));if(o[0]*s[0]>0){let l=this.calculateFirstParentOutputIndex(e,o[0],s[0]);for(let c=1;c<=this.raggedRank;++c)l=this.calculateOutputIndex(c-1,l,o[c],s[c]);this.setOutput(this.raggedRank,l,i,r)}return[r,i]}setOutput(t,e,s,o){if(s.length===0)return;const r=this.values,i=s;let a=o.slice();a=a.slice(t+1);const l=q(a),c=e.length;let u=this.defaultValue;if(u.length!==l&&u.length!==1){const f=this.defaultValueShape;V(()=>{const m=L(u,f);u=ii(m,a).dataSync()})}let h=0,d=0,p=0;for(let f=0;f<=c;++f){let m=f<c?e[f]:-1;if(m===p){++p;continue}if(d<p){const g=r.subarray(h*l),x=i.subarray(d*l),b=(p-d)*l;L0(x,g,b)}if(f>=c){const g=s.length;m=Math.floor(g/l)}if(m>p)if(this.defaultValue.length===1)i.subarray(p*l,m*l).fill(this.defaultValue[0]),p=m;else for(;m>p;){const g=i.slice(p*l);L0(g,u,l),++p}m<0?(h=f+1,d=p):(h=f,d=p,p=d+1)}}}function L0(n,t,e){for(let s=0;s<e;s++)n[s]=t[s]}function M0(n,t){const e=[];for(let s of n){if(s<0){if(!t)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}e.push(s)}return e}function P0(n,t,e,s,o,r,i,a,l,c){return new tc(n,t,e,s,o,r,i,a,l,c).compute()}function B0(n,t,e,s){const o=n===t,r=n<t&&e<0,i=t<n&&e>1;if(o||r||i)return Ie(0,s);const a=Math.abs(Math.ceil((t-n)/e)),l=Ie(a,s);t<n&&e===1&&(e=-1),l[0]=n;for(let c=1;c<l.length;c++)l[c]=l[c-1]+e;return l}const z0=Vn(n=>1/Math.sqrt(n)),ER=$s(Mr,z0),RR={kernelName:Mr,backendName:"cpu",kernelFunc:ER};function ro(n,t,e,s,o,r,i,a,l,c){const u=[s/o,o],h=n.values,d=t.values;if(s===0)return wt(e,t.dtype);const p=l instanceof me?l:wt(u,t.dtype);typeof l=="string"||typeof l=="number"?p.values.fill(l):typeof l=="boolean"&&p.values.fill(+l);for(let f=0;f<r;f++){const m=[];let g=0;for(let x=0;x<i;x++){const b=h[f*i+x];m.push(b),g+=b*a[x]}if(g<0||g>=s/o)throw new Error(`Invalid indices: ${m} does not index into ${e}`);for(let x=0;x<o;x++)c?p.values[g*o+x]+=d[f*o+x]:p.values[g*o+x]=t.rank===0?d[0]:d[f*o+x]}return p}const AR=Vn(n=>1/(1+Math.exp(-n))),V0=Ft(Wr,n=>1/(1+Math.exp(-n))),DR={kernelName:Wr,backendName:"cpu",kernelFunc:V0};function W0(n,t,e,s,o){const r=kh(s,t,e),i=q(e),a=lt(s);if(r){const h=vh(t,a);return o==="string"?n.slice(h,h+i):n.subarray(h,h+i)}const l=o==="string"?Zn(n):n,c=wt(s,o,l),u=wt(e,o);for(let h=0;h<u.size;++h){const d=u.indexToLoc(h),p=d.map((f,m)=>f+t[m]);u.set(c.get(...p),...d)}return o==="string"?dg(u.values):u.values}function io(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,size:i}=s;rt(o,"slice");const[a,l]=$l(o,r,i);Ih(o,a,l);const c=e.data.get(o.dataId).values,u=W0(c,a,l,o.shape,o.dtype);return e.makeTensorInfo(l,o.dtype,u)}const FR={kernelName:Ua,backendName:"cpu",kernelFunc:io};function U0(n,t,e,s,o,r,i){const a=t[0],l=r[0],c=new Array(l),u=new Array(a),h=t[1];if(l===0){if(a!==0)throw new Error(Qm(a));const g=Yt(e,0),x=Yt(o,0);return[g,[0,h],x,c,u]}let d=!0,p=0;const f=new Array(l).fill(0);for(let g=0;g<a;++g){const x=n[g*h];if(x<0)throw new Error(tg(g,x));if(x>=l)throw new Error(eg(g,x,l));++f[x],d=d&&x>=p,p=x}let m=!0;for(let g=0;g<l;++g){const x=f[g]===0;c[g]=x,m=m&&!x,f[g]=Math.max(f[g],1),g>0&&(f[g]+=f[g-1])}if(m&&d){const g=n,x=s;for(let b=0;b<a;++b)u[b]=b;return[g,[a,h],x,c,u]}else{const g=f[l-1],x=Yt(e,g*h),b=Yt(o,g),w=new Array(l).fill(0);for(let y=0;y<a;++y){const C=n[y*h],$=w[C],N=(C===0?0:f[C-1])+$;w[C]++;for(let T=0;T<h;++T)x[N*h+T]=n[y*h+T];b[N]=s[y],u[y]=N}for(let y=0;y<l;++y)if(w[y]===0){const $=y===0?0:f[y-1];x[$*h+0]=y;for(let N=1;N<h;++N)x[$*h+N]=0;b[$]=i}return[x,[g,h],b,c,u]}}function G0(n,t,e,s,o){const r=q(s),i=t[0],a=o.length,l=[];let c=1,u=-1;for(let g=0;g<a;++g){const x=o[g];if(x===-1){if(u!==-1)throw new Error(ng(u,g));u=g,l.push(1)}else{if(x<0)throw new Error(sg(g,x));c*=x,l.push(x)}}if(u!==-1){if(c<=0)throw new Error(og());const g=Math.trunc(r/c);if(c*g!==r)throw new Error(rg(s,l));l[u]=g}if(q(l)!==r)throw new Error(ig(s,l));const d=s.length,p=[];if(d>0){p[d-1]=1;for(let g=d-2;g>=0;--g)p[g]=p[g+1]*s[g+1]}const f=[];if(a>0){f[a-1]=1;for(let g=a-2;g>=0;--g)f[g]=f[g+1]*l[g+1]}const m=Yt(e,i*a);for(let g=0;g<i;++g){let x=0;for(let b=0;b<d;++b)x+=n[g*d+b]*p[b];for(let b=0;b<a;++b)m[g*a+b]=Math.trunc(x/f[b]),x%=f[b]}return[m,[i,a],l]}function Vd(n,t,e,s,o,r=!1,i=0){const a=s.length,l=[t[0],n.length/t[0]],c=l[1],h=a>0?o[a-1]+1:0;if(h<0)throw new Error(qh());const d=t.slice();d[0]=h;const p=d.reduce((w,y)=>w*y,1),f=Yt(e,p);if(a===0)return h>0&&f.fill(i),[f,d];if(h<=0)throw new Error(qh());let m=0,g=1,x=0,b=o[m];for(;;){let w=0;if(g<a){if(w=o[g],b===w){++g;continue}if(b>=w)throw new Error(ag())}if(b<0||b>=h)throw new Error(lg(b,h));b>x&&f.fill(i,x*c,b*c);for(let y=m;y<g;++y){const C=s[y];if(C<0||C>=l[0])throw new Error(cg(y,s[y],l[0]));for(let $=0;$<c;$++)f[b*c+$]+=n[C*c+$]}if(r)for(let y=0;y<c;y++)f[b*c+y]/=g-m;if(m=g,++g,x=b+1,b=w,g>a)break}return x<h&&f.fill(i,x*c,h*c),[f,d]}const OR=Vn(n=>Math.sqrt(n)),_R=Ft(Gr,n=>Math.sqrt(n)),LR={kernelName:Gr,backendName:"cpu",kernelFunc:_R};const H0=ee(((n,t)=>{const e=n-t;return e*e})),MR=ue(Hr,H0),PR={kernelName:Hr,backendName:"cpu",kernelFunc:MR};const q0=Vn((n,t)=>{const{pattern:e,replaceGlobal:s,rewrite:o}=t;return n.replace(new RegExp(e,s?"g":""),o)}),BR=$s(bu,q0),zR={kernelName:bu,backendName:"cpu",kernelFunc:BR};function K0(n,t,e,s){const o=wt(n,t.dtype);for(let r=0;r<o.size;r++){const i=o.indexToLoc(r),a=new Array(i.length);for(let l=0;l<a.length;l++)a[l]=i[l]*e[l]+s[l];o.set(t.get(...a),...i)}return o}class VR{constructor(t,e,s,o,r,i){this.separator=is(t),this.nGramWidths=e,this.leftPad=is(s),this.rightPad=is(o),this.padWidth=r,this.preserveShort=i}getPadWidth(t){return Math.min(this.padWidth<0?t-1:this.padWidth,t-1)}getNumNGrams(t,e){const s=this.getPadWidth(e);return Math.max(0,t+2*s-e+1)}createNGrams(t,e,s,o,r,i){for(let a=0;a<r;++a){const l=this.getPadWidth(i),c=Math.max(0,l-a),u=Math.max(0,l-(r-(a+1))),h=i-(c+u),d=e+(c>0?0:a-l);let p=0;p+=c*this.leftPad.length;for(let b=0;b<h;++b)p+=t[d+b].length;p+=u*this.rightPad.length;const f=c+u+h-1;p+=f*this.separator.length,s[o+a]=new Uint8Array(p);const m=s[o+a];let g=0;const x=b=>b.forEach(w=>m[g++]=w);for(let b=0;b<c;++b)x(this.leftPad),x(this.separator);for(let b=0;b<h-1;++b)x(t[d+b]),x(this.separator);if(h>0){x(t[d+h-1]);for(let b=0;b<u;++b)x(this.separator),x(this.rightPad)}else{for(let b=0;b<u-1;++b)x(this.rightPad),x(this.separator);x(this.rightPad)}}}compute(t,e){const s=t.length,o=e.length;if(o>0){let l=e[0];if(l!==0)throw new Error(`First split value must be 0, got ${l}`);for(let c=1;c<o;++c){let u=e[c]>=l;if(u=u&&e[c]<=s,!u)throw new Error(`Invalid split value ${e[c]}, must be in [${l}, ${s}]`);l=e[c]}if(l!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${l}`)}const r=o-1,i=Yt("int32",o);if(s===0||o===0){const l=new Array(s);for(let c=0;c<=r;++c)i[c]=0;return[l,i]}i[0]=0;for(let l=1;l<=r;++l){const c=e[l]-e[l-1];let u=0;this.nGramWidths.forEach(h=>{u+=this.getNumNGrams(c,h)}),this.preserveShort&&c>0&&u===0&&(u=1),i[l]=i[l-1]+u}const a=new Array(i[r]);for(let l=0;l<r;++l){const c=e[l];let u=i[l];if(this.nGramWidths.forEach(h=>{const d=e[l+1]-e[l],p=this.getNumNGrams(d,h);this.createNGrams(t,c,a,u,p,h),u+=p}),this.preserveShort&&u===i[l]){const h=e[l+1]-e[l];if(h===0)continue;const d=h+2*this.padWidth;this.createNGrams(t,c,a,u,1,d)}}return[a,i]}}function j0(n,t,e,s,o,r,i,a){return new VR(e,s,o,r,i,a).compute(n,t)}function WR(n,t,e,s){if(!n.length)return;if(t.length===0){for(let r=0;r<n.length;++r)s.push(n.subarray(r,r+1));return}if(t.length===1){const r=t[0];let i=n.indexOf(r);for(;i!==-1;){const a=n.subarray(0,i);(!e||a.length!==0)&&s.push(a),n=n.subarray(i+1),i=n.indexOf(r)}(!e||n.length!==0)&&s.push(n);return}let o=0;for(let r=0;r<n.length+1;r++)if(r===n.length||t.indexOf(n[r])!==-1){const i=n.subarray(o,r);(!e||i.length!==0)&&s.push(i),o=r+1}}function X0(n,t,e){const s=n.length,o=[];let r=0,i=0;const a=new Array(s);for(let d=0;d<s;++d){const p=o.length;WR(n[d],t,e,o);const f=o.length-p;a[d]=f,r+=f,i=Math.max(i,f)}const l=Yt("int32",r*2),c=new Array(r),u=[s,i];let h=0;for(let d=0;d<s;++d)for(let p=0;p<a[d];++p)l[h*2]=d,l[h*2+1]=p,c[h]=o[h],++h;return[l,c,u]}function Y0(n,t){const e=Yt("int32",n.length);for(let s=0;s<n.length;++s)e[s]=bw(n[s]).modulo(t).getLowBitsUnsigned();return e}const Z0=ee(((n,t)=>n-t)),UR=Md(((n,t,e,s)=>({real:n-e,imag:t-s}))),Wd=ue(qr,Z0,UR),GR={kernelName:qr,backendName:"cpu",kernelFunc:Wd};function J0(n,t){const e=new Array(n.rank);for(let o=0;o<e.length;o++)e[o]=n.shape[o]*t[o];const s=wt(e,n.dtype);for(let o=0;o<s.values.length;++o){const r=s.indexToLoc(o),i=new Array(n.rank);for(let l=0;l<i.length;l++)i[l]=r[l]%n.shape[l];const a=n.locToIndex(i);s.values[o]=n.values[a]}return s}const Oi=(n,t)=>{const e=t.value-n.value;return e===0?n.index-t.index:e};function Q0(n,t,e=0,s=n.length-1){for(;s>e;){if(s-e>600){const a=s-e+1,l=t-e+1,c=Math.log(a),u=.5*Math.exp(2*c/3),h=.5*Math.sqrt(c*u*(a-u)/a)*Math.sign(l-a/2),d=Math.max(e,Math.floor(t-l*u/a+h)),p=Math.min(s,Math.floor(t+(a-l)*u/a+h));Q0(n,t,d,p)}const o=n[t];let r=e,i=s;for(go(n,e,t),Oi(n[s],o)>0&&go(n,e,s);r<i;){for(go(n,r,i),r++,i--;Oi(n[r],o)<0;)r=r+1;for(;Oi(n[i],o)>0;)i=i-1}Oi(n[e],o)===0?go(n,e,i):(i=i+1,go(n,i,s)),i<=t&&(e=i+1),t<=i&&(s=i-1)}}function t1(n,t,e,s,o){const r=t[t.length-1],[i,a]=[n.length/r,r],l=Ce(e,i*s),c=Ce("int32",i*s);for(let h=0;h<i;h++){const d=h*a,p=n.subarray(d,d+a);let f=new Array(p.length);p.forEach((b,w)=>f[w]={value:b,index:w}),s<f.length&&(Q0(f,s),f=f.slice(0,s)),o&&f.sort(Oi);const m=h*s,g=l.subarray(m,m+s),x=c.subarray(m,m+s);for(let b=0;b<s;b++)g[b]=f[b].value,x[b]=f[b].index}const u=t.slice();return u[u.length-1]=s,[wt(u,e,l),wt(u,"int32",c)]}function e1(n,t,e,s){const o=yt(t,e)[0],r=[1,e[0],1];for(let f=0;f<o;f++)r[0]*=e[f];r[1]=e[o];for(let f=o+1;f<e.length;f++)r[2]*=e[f];const i=new Map,a=new Int32Array(e[o]),l=new me(r,s,n),c=[],u=r[0]===1&&r[2]===1;for(let f=0;f<e[o];f++){let m;if(u)m=n[f].toString();else{const x=[];for(let b=0;b<r[0];b++)for(let w=0;w<r[2];w++)x.push(l.get(b,f,w));m=x.join(",")}const g=i.get(m);if(g!=null)a[f]=g;else{const x=i.size;i.set(m,x),a[f]=x,c.push(f)}}const h=r.slice();h[1]=i.size;const d=new me(h,s);c.forEach((f,m)=>{for(let g=0;g<r[0];g++)for(let x=0;x<r[2];x++)d.set(l.get(g,f,x),g,m,x)});const p=e.slice();return p[o]=h[1],{outputValues:d.values,outputShape:p,indices:a}}var HR=Object.freeze({__proto__:null,addImpl:i0,bincountImpl:Pd,bincountReduceImpl:a0,bitwiseAndImpl:l0,castImpl:r0,ceilImpl:c0,concatImpl:u0,equalImpl:h0,expImpl:p0,expm1Impl:m0,floorDivImpl:x0,floorImpl:g0,gatherNdImpl:b0,gatherV2Impl:y0,greaterEqualImpl:C0,greaterImpl:w0,lessEqualImpl:$0,lessImpl:I0,linSpaceImpl:k0,logImpl:v0,maxImpl:S0,maximumImpl:N0,minimumImpl:T0,multiplyImpl:Bd,negImpl:E0,notEqualImpl:R0,prodImpl:A0,raggedGatherImpl:F0,raggedRangeImpl:_0,raggedTensorToTensorImpl:P0,rangeImpl:B0,rsqrtImpl:z0,scatterImpl:ro,sigmoidImpl:AR,simpleAbsImpl:o0,sliceImpl:W0,sparseFillEmptyRowsImpl:U0,sparseReshapeImpl:G0,sparseSegmentReductionImpl:Vd,sqrtImpl:OR,squaredDifferenceImpl:H0,staticRegexReplaceImpl:q0,stridedSliceImpl:K0,stringNGramsImpl:j0,stringSplitImpl:X0,stringToHashBucketFastImpl:Y0,subImpl:Z0,tileImpl:J0,topKImpl:t1,transposeImpl:zd,uniqueImpl:e1});If("cpu",()=>new Zl,1);const n1=Ft(mr,n=>n>=0?n:Math.exp(n)-1),qR={kernelName:mr,backendName:"cpu",kernelFunc:n1};function s1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{alpha:r}=s;rt([o],"leakyRelu");const i=q(o.shape),a=e.data.get(o.dataId).values,l=Ce("float32",i);for(let c=0;c<a.length;c++)l[c]=a[c]<0?r*a[c]:a[c];return e.makeTensorInfo(o.shape,"float32",l)}const KR={kernelName:xa,backendName:"cpu",kernelFunc:s1};const jR=ee((n,t)=>n<0?t*n:n);function o1(n){const{inputs:t,backend:e}=n,{x:s,alpha:o}=t;rt([s,o],"prelu");const r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,[a,l]=jR(s.shape,o.shape,r,i,"float32");return e.makeTensorInfo(l,"float32",a)}const XR={kernelName:La,backendName:"cpu",kernelFunc:o1};const r1=Ft(Or,n=>Math.max(0,n)),YR={kernelName:Or,backendName:"cpu",kernelFunc:r1};const i1=Ft(_r,n=>Math.min(Math.max(0,n),6)),ZR={kernelName:_r,backendName:"cpu",kernelFunc:i1};function ec(n,t,e,s,o){if(e==="linear")return zn({inputs:{x:t},backend:n});if(e==="relu")return r1({inputs:{x:t},backend:n});if(e==="elu")return n1({inputs:{x:t},backend:n});if(e==="relu6")return i1({inputs:{x:t},backend:n});if(e==="prelu")return o1({inputs:{x:t,alpha:s},backend:n});if(e==="leakyrelu")return s1({inputs:{x:t},backend:n,attrs:{alpha:o}});if(e==="sigmoid")return V0({inputs:{x:t},backend:n});throw new Error(`Activation ${e} has not been implemented for the CPU backend.`)}function Bt(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{shape:r}=s,i=q(o.shape),a=bp(r,i),l=q(a);S(i===l,()=>`The new shape (${a}) has ${l} elements and the old shape (${o.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`),e.incRef(o.dataId);const c=e.data.get(o.dataId);if(c.complexTensorInfos!=null){const u=c.complexTensorInfos.real,h=c.complexTensorInfos.imag;u.shape=a,h.shape=a}return{dataId:o.dataId,shape:a,dtype:o.dtype}}const JR={kernelName:Pa,backendName:"cpu",kernelFunc:Bt};function a1(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r}=t,{transposeA:i,transposeB:a}=s;rt([o,r],"matMul");const l=o.shape.length,c=r.shape.length,u=i?o.shape[l-2]:o.shape[l-1],h=a?r.shape[c-1]:r.shape[c-2],d=i?o.shape[l-1]:o.shape[l-2],p=a?r.shape[c-2]:r.shape[c-1],f=o.shape.slice(0,-2),m=r.shape.slice(0,-2),g=q(f),x=q(m),w=mt(o.shape.slice(0,-2),r.shape.slice(0,-2)).concat([d,p]);S(u===h,()=>`Error in matMul: inner shapes (${u}) and (${h}) of Tensors with shapes ${o.shape} and ${r.shape} and transposeA=${i} and transposeB=${a} must match.`);const y=i?[g,u,d]:[g,d,u],C=a?[x,p,h]:[x,h,p],$=Bt({inputs:{x:o},backend:e,attrs:{shape:y}}),N=Bt({inputs:{x:r},backend:e,attrs:{shape:C}}),T=i?$.shape[1]:$.shape[2],k=i?$.shape[2]:$.shape[1],v=a?N.shape[1]:N.shape[2],I=Math.max(g,x),R=e.data.get($.dataId).values,A=e.data.get(N.dataId).values,P=lt($.shape),_=lt(N.shape),[B,z,W]=i?[P[0],1,P[1]]:[P[0],P[1],1],[H,K,j]=a?[1,_[1],_[0]]:[_[1],1,_[0]],X=k*v,Y=wt([I,k,v],$.dtype),tt=Y.values,Q=e.blockSize;for(let ot=0;ot<I;ot++){const at=ot%g,ct=ot%x;for(let ht=0;ht<k;ht+=Q){const gt=Math.min(ht+Q,k);for(let xt=0;xt<v;xt+=Q){const Tt=Math.min(xt+Q,v);for(let $t=0;$t<T;$t+=Q){const Xt=Math.min($t+Q,T);for(let zt=ht;zt<gt;zt++)for(let _t=xt;_t<Tt;_t++){let jt=0;for(let Ht=$t;Ht<Xt;Ht++){const es=R[at*B+zt*z+Ht*W],we=A[Ht*H+_t*K+ct*j];jt+=es*we}tt[ot*X+(zt*v+_t)]+=jt}}}}}return e.disposeIntermediateTensorInfo($),e.disposeIntermediateTensorInfo(N),e.makeTensorInfo(w,Y.dtype,Y.values)}const QR={kernelName:na,backendName:"cpu",kernelFunc:a1};function tA(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r,bias:i,preluActivationWeights:a}=t,{transposeA:l,transposeB:c,activation:u,leakyreluAlpha:h}=s;let d,p,f;const m=[];d=a1({inputs:{a:o,b:r},attrs:{transposeA:l,transposeB:c},backend:e}),i&&(p=zo({inputs:{a:d,b:i},backend:e}),m.push(d),d=p),u&&(f=ec(e,d,u,a,h),m.push(d),d=f);for(const x of m)e.disposeIntermediateTensorInfo(x);return d}const eA={kernelName:Za,backendName:"cpu",kernelFunc:tA};const nA=Ft(nr,n=>Math.acos(n)),sA={kernelName:nr,backendName:"cpu",kernelFunc:nA};const oA=Ft(sr,n=>Math.acosh(n)),rA={kernelName:sr,backendName:"cpu",kernelFunc:oA};function iA(n){const{inputs:t,backend:e}=n,s=t;rt(t,"addN");const o=s.map(a=>e.data.get(a.dataId).values),r=wt(s[0].shape,s[0].dtype),i=r.values;for(let a=0;a<s.length;a++){const l=o[a];for(let c=0;c<i.length;c++)i[c]+=l[c]}return e.makeTensorInfo(r.shape,r.dtype,r.values)}const aA={kernelName:_c,backendName:"cpu",kernelFunc:iA};function lA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;rt(o,"all");const a=yt(r,o.shape);let l=a;const c=qt(l,o.shape.length);let u=o;c!=null&&(u=ze({inputs:{x:o},backend:e,attrs:{perm:c}}),l=Jt(l.length,o.shape.length)),xe("all",l,u.shape.length);const[h,d]=de(u.shape,l),p=q(d),f=Ie(q(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const C=m[b+y];w=w&&C}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ne(h,a),b=Bt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const cA={kernelName:Lc,backendName:"cpu",kernelFunc:lA};function uA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;rt(o,"any");const a=yt(r,o.shape);let l=a;const c=qt(l,o.shape.length);let u=o;c!=null&&(u=ze({inputs:{x:o},backend:e,attrs:{perm:c}}),l=Jt(l.length,o.shape.length)),xe("any",l,u.shape.length);const[h,d]=de(u.shape,l),p=q(d),f=Ie(q(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const C=m[b+y];w=w||C}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ne(h,a),b=Bt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const hA={kernelName:Mc,backendName:"cpu",kernelFunc:uA};function dA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;rt(o,"argMax");let i=yt(r,o.shape);const a=qt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=ze({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=Jt(i.length,l.shape.length)),i=[i[0]],xe("argMax",i,l.shape.length);const[u,h]=de(l.shape,i),d=q(u),p=Ie(d,"int32"),f=q(h),m=e.data.get(l.dataId).values;for(let g=0;g<p.length;++g){const x=g*f;let b=m[x],w=0;for(let y=0;y<f;++y){const C=m[x+y];C>b&&(b=C,w=y)}p[g]=w}return c.forEach(g=>e.disposeIntermediateTensorInfo(g)),e.makeTensorInfo(u,"int32",p)}const pA={kernelName:Ji,backendName:"cpu",kernelFunc:dA};function fA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;rt(o,"argMin");let i=yt(r,o.shape);const a=qt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=ze({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=Jt(i.length,l.shape.length)),i=[i[0]],xe("argMin",i,l.shape.length);const[u,h]=de(l.shape,i),d=q(u),p=Ie(d,"int32"),f=q(h),m=e.data.get(l.dataId).values;for(let g=0;g<p.length;++g){const x=g*f;let b=m[x],w=0;for(let y=0;y<f;++y){const C=m[x+y];C<b&&(b=C,w=y)}p[g]=w}return c.forEach(g=>e.disposeIntermediateTensorInfo(g)),e.makeTensorInfo(u,"int32",p)}const mA={kernelName:Qi,backendName:"cpu",kernelFunc:fA};const gA=Ft(or,n=>Math.asin(n)),xA={kernelName:or,backendName:"cpu",kernelFunc:gA};const bA=Ft(rr,n=>Math.asinh(n)),yA={kernelName:rr,backendName:"cpu",kernelFunc:bA};const wA=Ft(ir,n=>Math.atan(n)),CA={kernelName:ir,backendName:"cpu",kernelFunc:wA};const IA=ee((n,t)=>Math.atan2(n,t)),$A=ue(lr,IA),kA={kernelName:lr,backendName:"cpu",kernelFunc:$A};const vA=Ft(ar,n=>Math.atanh(n)),SA={kernelName:ar,backendName:"cpu",kernelFunc:vA};function Ud(n,t,e,s,o,r){const i=o.strideHeight,a=o.strideWidth,l=o.dilationHeight,c=o.dilationWidth,u=o.effectiveFilterHeight,h=o.effectiveFilterWidth,d=o.padInfo.top,p=o.padInfo.left,f=r==="max"?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,m=wt(o.outShape,e),g=m.values,x=o.outShape[1]*o.outShape[2]*o.outShape[3],b=o.outShape[2]*o.outShape[3],w=o.outShape[3];for(let y=0;y<o.batchSize;++y){const C=y*x,$=y*s[0];for(let N=0;N<o.inChannels;++N)for(let T=0;T<o.outHeight;++T){const k=T*i-d,v=Math.max(0,k),I=Math.min(o.inHeight,u+k),R=C+T*b;for(let A=0;A<o.outWidth;++A){const P=A*a-p,_=Math.max(0,P),B=Math.min(o.inWidth,h+P);let z=f,W=0,H=0;for(let j=v;j<I;j+=l){const X=$+j*s[1];for(let Y=_;Y<B;Y+=c){const tt=X+Y*s[2],Q=n[tt+N];r==="max"&&Q>z?z=Q:r==="avg"&&(W+=Q,H++)}if(isNaN(z))break}const K=R+A*w+N;g[K]=r==="avg"?W/H:z}}}return m}function l1(n,t,e,s,o=!1,r=!1){const i=wt(s.outShape,"int32"),a=s.strideHeight,l=s.strideWidth,c=s.dilationHeight,u=s.dilationWidth,h=s.effectiveFilterHeight,d=s.effectiveFilterWidth,p=s.padInfo.top,f=s.padInfo.left,m=wt(t,e,n);for(let g=0;g<s.batchSize;++g)for(let x=0;x<s.inChannels;++x)for(let b=0;b<s.outHeight;++b){const w=b*a-p;let y=w;for(;y<0;)y+=c;const C=Math.min(s.inHeight,h+w);for(let $=0;$<s.outWidth;++$){const N=$*l-f;let T=N;for(;T<0;)T+=u;const k=Math.min(s.inWidth,d+N);let v=Number.NEGATIVE_INFINITY,I=-1;for(let R=y;R<C;R+=c){const A=R-w;for(let P=T;P<k;P+=u){const _=P-N,B=m.get(g,R,P,x);B>v&&(v=B,o?I=r?((g*s.inHeight+R)*s.inWidth+P)*s.inChannels+x:(R*s.inWidth+P)*s.inChannels+x:I=A*d+_)}}i.set(I,g,b,$,x)}}return i}function c1(n,t,e,s,o,r){const i=o.strideDepth,a=o.strideHeight,l=o.strideWidth,c=o.dilationDepth,u=o.dilationHeight,h=o.dilationWidth,d=o.effectiveFilterDepth,p=o.effectiveFilterHeight,f=o.effectiveFilterWidth,m=o.padInfo.front,g=o.padInfo.top,x=o.padInfo.left,b=r==="max"?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,w=wt(o.outShape,e),y=w.values,C=o.outShape[1]*o.outShape[2]*o.outShape[3]*o.outShape[4],$=o.outShape[2]*o.outShape[3]*o.outShape[4],N=o.outShape[3]*o.outShape[4],T=o.outShape[4];for(let k=0;k<o.batchSize;++k){const v=k*C,I=k*s[0];for(let R=0;R<o.inChannels;++R)for(let A=0;A<o.outDepth;++A){const P=A*i-m;let _=P;for(;_<0;)_+=c;const B=Math.min(o.inDepth,d+P),z=v+A*$;for(let W=0;W<o.outHeight;++W){const H=W*a-g;let K=H;for(;K<0;)K+=u;const j=Math.min(o.inHeight,p+H),X=z+W*N;for(let Y=0;Y<o.outWidth;++Y){const tt=Y*l-x;let Q=tt;for(;Q<0;)Q+=h;const ot=Math.min(o.inWidth,f+tt),at=X+Y*T;let ct=b,ht=0,gt=0;for(let Tt=_;Tt<B;Tt+=c){const $t=I+Tt*s[1];for(let Xt=K;Xt<j;Xt+=u){const zt=$t+Xt*s[2];for(let _t=Q;_t<ot;_t+=h){const jt=zt+_t*s[3],Ht=n[jt+R];if(r==="max"&&Ht>ct?ct=Ht:r==="avg"&&(ht+=Ht,gt++),isNaN(ct))break}if(isNaN(ct))break}if(isNaN(ct))break}const xt=at+R;y[xt]=r==="avg"?ht/Math.max(gt,1):ct}}}}return w}function NA(n,t){const e=wt(t.outShape,"int32"),s=t.strideDepth,o=t.strideHeight,r=t.strideWidth,i=t.dilationDepth,a=t.dilationHeight,l=t.dilationWidth,c=t.effectiveFilterDepth,u=t.effectiveFilterHeight,h=t.effectiveFilterWidth,d=t.padInfo.front,p=t.padInfo.top,f=t.padInfo.left;for(let m=0;m<t.batchSize;++m)for(let g=0;g<t.inChannels;++g)for(let x=0;x<t.outDepth;++x){const b=x*s-d;let w=b;for(;w<0;)w+=i;const y=Math.min(t.inDepth,c+b);for(let C=0;C<t.outHeight;++C){const $=C*o-p;let N=$;for(;N<0;)N+=a;const T=Math.min(t.inHeight,u+$);for(let k=0;k<t.outWidth;++k){const v=k*r-f;let I=v;for(;I<0;)I+=l;const R=Math.min(t.inWidth,h+v);let A=Number.NEGATIVE_INFINITY,P=-1;for(let _=w;_<y;_+=i){const B=_-b;for(let z=N;z<T;z+=a){const W=z-$;for(let H=I;H<R;H+=l){const K=H-v,j=n.get(m,_,z,H,g);j>=A&&(A=j,P=B*u*h+W*u+K)}}}e.set(P,m,x,C,k,g)}}}return e}function TA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;rt(o,"avgPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;S($e(i,c),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=nn(o.shape,r,i,c,a,l);let h;if(u.filterWidth===1&&u.filterHeight===1&&Et(u.inShape,u.outShape))h=zn({inputs:{x:o},backend:e});else{const d=e.data.get(o.dataId).values,p=lt(o.shape),f=Ud(d,o.shape,o.dtype,p,u,"avg");h=e.makeTensorInfo(u.outShape,o.dtype,f.values)}return h}const EA={kernelName:ta,backendName:"cpu",kernelFunc:TA};function RA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s;rt(o,"avgPool3d");const u=Hn(o.shape,r,i,1,a,l,c),h=e.data.get(o.dataId).values,d=c1(h,o.shape,o.dtype,lt(o.shape),u,"avg");return e.makeTensorInfo(d.shape,"float32",d.values)}const AA={kernelName:ea,backendName:"cpu",kernelFunc:RA};function DA(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,{filterSize:i,strides:a,pad:l,dimRoundingMode:c}=s;rt([o,r],"avgPool3DGrad");const u=Hn(r.shape,i,a,1,l,c),h=u.strideDepth,d=u.strideHeight,p=u.strideWidth,f=u.filterDepth,m=u.filterHeight,g=u.filterWidth,x=u.dilationDepth,b=u.dilationHeight,w=u.dilationWidth,y=u.effectiveFilterDepth,C=u.effectiveFilterHeight,$=u.effectiveFilterWidth,N=y-1-u.padInfo.front,T=$-1-u.padInfo.left,k=C-1-u.padInfo.top,v=wt(r.shape,"float32"),I=1/(f*m*g),R=e.bufferSync(o);for(let A=0;A<u.batchSize;++A)for(let P=0;P<u.inChannels;++P)for(let _=0;_<u.inDepth;++_)for(let B=0;B<u.inHeight;++B)for(let z=0;z<u.inWidth;++z){const W=_-N,H=B-k,K=z-T;let j=0;for(let X=0;X<y;X+=x){const Y=(W+X)/h;if(!(Y<0||Y>=u.outDepth||Math.floor(Y)!==Y))for(let tt=0;tt<C;tt+=b){const Q=(H+tt)/d;if(!(Q<0||Q>=u.outHeight||Math.floor(Q)!==Q))for(let ot=0;ot<$;ot+=w){const at=(K+ot)/p;if(at<0||at>=u.outWidth||Math.floor(at)!==at)continue;const ct=R.get(A,Y,Q,at,P);j+=ct}}}v.set(j*I,A,_,B,z,P)}return e.makeTensorInfo(v.shape,v.dtype,v.values)}const FA={kernelName:Bc,backendName:"cpu",kernelFunc:DA};function OA(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r;rt([o,r],"avgPoolGrad");const{filterSize:a,strides:l,pad:c}=s,u=nn(i.shape,a,l,1,c),h=u.strideHeight,d=u.strideWidth,p=u.filterHeight,f=u.filterWidth,m=u.dilationHeight,g=u.dilationWidth,x=u.effectiveFilterHeight,b=u.effectiveFilterWidth,w=b-1-u.padInfo.left,y=x-1-u.padInfo.top,C=wt(i.shape,"float32"),$=1/(p*f),N=e.data.get(o.dataId).values,T=wt(o.shape,"float32",N);for(let k=0;k<u.batchSize;++k)for(let v=0;v<u.inChannels;++v)for(let I=0;I<u.inHeight;++I)for(let R=0;R<u.inWidth;++R){const A=I-y,P=R-w;let _=0;for(let B=0;B<x;B+=m){const z=(A+B)/h;if(!(z<0||z>=u.outHeight||Math.floor(z)!==z))for(let W=0;W<b;W+=g){const H=(P+W)/d;if(H<0||H>=u.outWidth||Math.floor(H)!==H)continue;const K=T.get(k,z,H,v);_+=K}}C.set(_*$,k,I,R,v)}return e.makeTensorInfo(C.shape,C.dtype,C.values)}const _A={kernelName:Pc,backendName:"cpu",kernelFunc:OA};function LA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,scale:r,offset:i,mean:a,variance:l}=t;S(a.shape.length===l.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),S(i==null||a.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),S(r==null||a.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks."),rt([o,a,l,r,i],"batchNorm");let{varianceEpsilon:c}=s;c==null&&(c=.001);const u=e.data.get(o.dataId).values,h=e.data.get(a.dataId).values,d=e.data.get(l.dataId).values,p=r?e.data.get(r.dataId).values:new Float32Array([1]),f=i?e.data.get(i.dataId).values:new Float32Array([0]),m=new Float32Array(u.length),g=f.length,x=p.length,b=d.length,w=h.length;let y=0,C=0,$=0,N=0;for(let T=0;T<u.length;++T)m[T]=f[y++]+(u[T]-h[C++])*p[$++]/Math.sqrt(d[N++]+c),y>=g&&(y=0),C>=w&&(C=0),$>=x&&($=0),N>=b&&(N=0);return e.makeTensorInfo(o.shape,o.dtype,m)}const MA={kernelName:fa,backendName:"cpu",kernelFunc:LA};function PA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,crops:i}=s;rt([o],"batchToSpaceND");const a=r.reduce((x,b)=>x*b),l=fi(o.shape,r,a),c=mi(l.length,r.length),u=gi(o.shape,r,a),h=Rh(i,r.length),d=Ah(u,i,r.length),p=Bt({inputs:{x:o},backend:e,attrs:{shape:l}}),f=ze({inputs:{x:p},backend:e,attrs:{perm:c}}),m=Bt({inputs:{x:f},backend:e,attrs:{shape:u}}),g=io({inputs:{x:m},backend:e,attrs:{begin:h,size:d}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),g}const BA={kernelName:sa,backendName:"cpu",kernelFunc:PA};function zA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i}=s,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,c=Pd(a,l,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,c)}const VA={kernelName:zc,backendName:"cpu",kernelFunc:zA};function WA(n){const{inputs:t,backend:e}=n,{s0:s,s1:o}=t,r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,a=mt(Array.from(r),Array.from(i));return e.makeTensorInfo([a.length],"int32",Int32Array.from(a))}const UA={kernelName:vp,backendName:"cpu",kernelFunc:WA};const GA=Ft(hr,(n,t)=>{const e=t;return n>e.clipValueMax?e.clipValueMax:n<e.clipValueMin?e.clipValueMin:n}),HA={kernelName:hr,backendName:"cpu",kernelFunc:GA};const qA={kernelName:oa,backendName:"cpu",kernelFunc:n=>{const{x:t}=n.inputs,e=n.backend,s=new Float32Array(q(t.shape)),o=e.data.get(t.dataId),r=o.complexTensorInfos.real,i=o.complexTensorInfos.imag,a=e.data.get(r.dataId).values,l=e.data.get(i.dataId).values;for(let c=0;c<a.length;c++){const u=a[c],h=l[c];s[c]=Math.hypot(u,h)}return e.makeOutput(s,t.shape,"float32")}};function Vo(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.data.get(s.dataId).complexTensorInfos.imag,r=e.data.get(o.dataId).values;return e.makeTensorInfo(o.shape,o.dtype,r)}const KA={kernelName:iu,backendName:"cpu",kernelFunc:Vo};function Wo(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s,r=yt(o,t[0].shape)[0],i=t.map(m=>m.shape);Nh(i,r);let a=Fn(t.map(m=>m.shape),r);if(q(a)===0)return e.makeTensorInfo(a,t[0].dtype,[]);const l=t.filter(m=>q(m.shape)>0);if(l.length===1)return zn({inputs:{x:l[0]},backend:e});if(l[0].dtype==="complex64"){const m=l.map(y=>oo({inputs:{input:y},backend:e})),g=l.map(y=>Vo({inputs:{input:y},backend:e})),x=Wo({inputs:m,backend:e,attrs:{axis:r}}),b=Wo({inputs:g,backend:e,attrs:{axis:r}}),w=qe({inputs:{real:x,imag:b},backend:e});return m.forEach(y=>e.disposeIntermediateTensorInfo(y)),g.forEach(y=>e.disposeIntermediateTensorInfo(y)),e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(b),w}const c=l.map(m=>{const x=[-1,q(m.shape.slice(r))];return Bt({inputs:{x:m},backend:e,attrs:{shape:x}})}),u=c.map(m=>({vals:e.data.get(m.dataId).values,shape:m.shape}));a=Fn(c.map(m=>m.shape),1);const h=c[0].shape[0]===1,d=u0(u,a,t[0].dtype,h),p=Fn(l.map(m=>m.shape),r),f=e.makeTensorInfo(p,t[0].dtype,d);return c.forEach(m=>e.disposeIntermediateTensorInfo(m)),f}const jA={kernelName:ra,backendName:"cpu",kernelFunc:Wo};function u1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dataFormat:l,dilations:c,dimRoundingMode:u}=s;rt([o,r],"conv2d");const h=qn(l),d=ge(o.shape,r.shape,i,c,a,u,!1,h),p=d.filterHeight,f=d.filterWidth,m=d.dilationHeight,g=d.dilationWidth,x=d.padInfo.left,b=d.padInfo.top,w=d.dataFormat==="channelsLast",y=new me(d.outShape,o.dtype),C=lt(o.shape),$=lt(r.shape),N=C[0],T=w?C[1]:C[2],k=w?C[2]:1,v=w?1:C[1],I=y.strides[0],R=w?y.strides[1]:y.strides[2],A=w?y.strides[2]:1,P=w?1:y.strides[1],_=e.data.get(o.dataId).values,B=e.data.get(r.dataId).values,z=y.values;for(let W=0;W<d.batchSize;++W){const H=W*N,K=W*I;for(let j=0;j<d.outHeight;++j){const X=K+j*R,Y=j*d.strideHeight-b;for(let tt=0;tt<p;++tt){const Q=Y+tt*m;if(Q<0||Q>=d.inHeight)continue;const ot=tt*$[0],at=H+Q*T;for(let ct=0;ct<d.outWidth;++ct){const ht=X+ct*A,gt=ct*d.strideWidth-x;for(let xt=0;xt<f;++xt){const Tt=gt+xt*g;if(Tt<0||Tt>=d.inWidth)continue;const $t=ot+xt*$[1],Xt=at+Tt*k;let zt=$t;for(let _t=0;_t<d.inChannels;++_t){const jt=_[Xt+_t*v];for(let Ht=0;Ht<d.outChannels;++Ht)z[ht+Ht*P]+=jt*B[zt+Ht];zt+=d.outChannels}}}}}}return e.makeTensorInfo(y.shape,y.dtype,z)}const XA={kernelName:ia,backendName:"cpu",kernelFunc:u1};function YA(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,dataFormat:l,dimRoundingMode:c,filterShape:u}=s;rt([o,r],"conv2dBackpropFilter");const h=qn(l),d=ge(o.shape,u,i,1,a,c,!1,h),{strideHeight:p,strideWidth:f,filterHeight:m,filterWidth:g}=d,x=d.dataFormat==="channelsLast",b=new me(d.filterShape,"float32"),w=d.padInfo.left,y=d.padInfo.top,C=e.data.get(o.dataId).values,$=e.data.get(r.dataId).values,N=new me(o.shape,o.dtype,C),T=new me(r.shape,r.dtype,$);for(let k=0;k<m;++k){const v=Math.max(0,Math.ceil((y-k)/p)),I=Math.min(d.outHeight,(d.inHeight+y-k)/p);for(let R=0;R<g;++R){const A=Math.max(0,Math.ceil((w-R)/f)),P=Math.min(d.outWidth,(d.inWidth+w-R)/f);for(let _=0;_<d.inChannels;++_)for(let B=0;B<d.outChannels;++B){let z=0;for(let W=0;W<d.batchSize;++W)for(let H=v;H<I;++H){const K=k+H*p-y;for(let j=A;j<P;++j){const X=R+j*f-w;x?z+=N.get(W,K,X,_)*T.get(W,H,j,B):z+=N.get(W,_,K,X)*T.get(W,B,H,j)}}b.set(z,k,R,_,B)}}}return e.makeTensorInfo(b.shape,b.dtype,b.values)}const ZA={kernelName:Uc,backendName:"cpu",kernelFunc:YA};function JA(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{inputShape:i,strides:a,pad:l,dataFormat:c,dimRoundingMode:u}=s;rt([o,r],"conv2dBackpropInput");const h=lt(r.shape),d=lt(o.shape);let p=qn(c);const f=ge(i,r.shape,a,1,l,u,!1,p),m=new me(f.inShape,"float32"),g=m.values,x=e.data.get(o.dataId).values,b=e.data.get(r.dataId).values,[w,y,C]=h,{batchSize:$,filterHeight:N,filterWidth:T,inChannels:k,inHeight:v,inWidth:I,outChannels:R,outHeight:A,outWidth:P,strideHeight:_,strideWidth:B}=f;p=f.dataFormat;const z=N-1-f.padInfo.top,W=T-1-f.padInfo.left,H=p==="channelsLast",K=m.strides[0],j=H?m.strides[1]:m.strides[2],X=H?m.strides[2]:1,Y=H?1:m.strides[1],tt=d[0],Q=H?d[1]:d[2],ot=H?d[2]:1,at=H?1:d[1];for(let ct=0;ct<$;++ct)for(let ht=0;ht<k;++ht)for(let gt=0;gt<v;++gt){const xt=gt-z,Tt=Math.max(0,Math.ceil(xt/_)),$t=Math.min(A,(N+xt)/_);for(let Xt=0;Xt<I;++Xt){const zt=Xt-W,_t=Math.max(0,Math.ceil(zt/B)),jt=Math.min(P,(T+zt)/B);let Ht=0;for(let we=Tt;we<$t;++we){const Ns=we*_-xt;for(let tn=_t;tn<jt;++tn){const mo=tn*B-zt,vn=tt*ct+Q*we+ot*tn,ns=w*(N-1-Ns)+y*(T-1-mo)+C*ht;for(let Ts=0;Ts<R;++Ts){const Es=x[vn+at*Ts],Rs=b[ns+Ts];Ht+=Es*Rs}}}const es=K*ct+j*gt+X*Xt+Y*ht;g[es]=Ht}}return e.makeTensorInfo(m.shape,m.dtype,m.values)}const QA={kernelName:aa,backendName:"cpu",kernelFunc:JA};function tD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s;rt([o,r],"conv3d");const c=cs(o.shape,r.shape,i,l,a),{filterDepth:u,filterHeight:h,filterWidth:d,dilationDepth:p,dilationHeight:f,dilationWidth:m,padInfo:g}=c,x=g.front,b=g.left,w=g.top,y=new me(c.outShape,o.dtype),C=e.data.get(o.dataId).values,$=e.data.get(r.dataId).values,N=y.values,T=lt(o.shape),k=lt(r.shape);for(let v=0;v<c.batchSize;++v){const I=v*T[0],R=v*y.strides[0];for(let A=0;A<c.outDepth;++A){const P=R+A*y.strides[1],_=A*c.strideDepth-x;for(let B=0;B<u;++B){const z=_+B*p;if(z<0||z>=c.inDepth)continue;const W=B*k[0],H=I+z*T[1];for(let K=0;K<c.outHeight;++K){const j=P+K*y.strides[2],X=K*c.strideHeight-w;for(let Y=0;Y<h;++Y){const tt=X+Y*f;if(tt<0||tt>=c.inHeight)continue;const Q=W+Y*k[1],ot=H+tt*T[2];for(let at=0;at<c.outWidth;++at){const ct=j+at*c.outChannels,ht=at*c.strideWidth-b;for(let gt=0;gt<d;++gt){const xt=ht+gt*m;if(xt<0||xt>=c.inWidth)continue;const Tt=Q+gt*k[2],$t=ot+xt*c.inChannels;let Xt=Tt;for(let zt=0;zt<c.inChannels;++zt){const _t=C[$t+zt];for(let jt=0;jt<c.outChannels;++jt)N[ct+jt]+=_t*$[Xt+jt];Xt+=c.outChannels}}}}}}}}return e.makeTensorInfo(y.shape,y.dtype,y.values)}const eD={kernelName:la,backendName:"cpu",kernelFunc:tD};function nD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,filterShape:l}=s;rt([o,r],"conv3dBackpropFilterV2");const c=lt(o.shape),u=lt(r.shape),h=cs(o.shape,l,i,1,a),d=h.strideDepth,p=h.strideHeight,f=h.strideWidth,m=h.filterDepth,g=h.filterHeight,x=h.filterWidth,b=new me(h.filterShape,"float32"),w=b.values,[y,C,$,N]=b.strides,T=e.data.get(r.dataId).values,[k,v,I,R]=u,A=e.data.get(o.dataId).values,[P,_,B,z]=c,W=h.padInfo.front,H=h.padInfo.left,K=h.padInfo.top;for(let j=0;j<m;++j){const X=Math.max(0,Math.ceil((W-j)/d)),Y=Math.min(h.outDepth,(h.inDepth+W-j)/d),tt=j*y;for(let Q=0;Q<g;++Q){const ot=Math.max(0,Math.ceil((K-Q)/p)),at=Math.min(h.outHeight,(h.inHeight+K-Q)/p),ct=Q*C+tt;for(let ht=0;ht<x;++ht){const gt=Math.max(0,Math.ceil((H-ht)/f)),xt=Math.min(h.outWidth,(h.inWidth+H-ht)/f),Tt=ht*$+ct;for(let $t=0;$t<h.inChannels;++$t){const Xt=$t*N+Tt;for(let zt=0;zt<h.outChannels;++zt){let _t=0;for(let jt=0;jt<h.batchSize;++jt){const Ht=jt*P,es=jt*k;for(let we=X;we<Y;++we){const tn=(j+we*d-W)*_+Ht,mo=we*v+es;for(let vn=ot;vn<at;++vn){const Ts=(Q+vn*p-K)*B+tn,Es=vn*I+mo;for(let Rs=gt;Rs<xt;++Rs){const fp=(ht+Rs*f-H)*z+Ts,mp=Rs*R+Es;_t+=A[fp+$t]*T[mp+zt]}}}}w[Xt+zt]=_t}}}}}return e.makeTensorInfo(b.shape,b.dtype,b.values)}const sD={kernelName:Gc,backendName:"cpu",kernelFunc:nD};function oD(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{pad:i,strides:a,inputShape:l}=s;rt([o],"conv3dBackpropInputV2");const c=lt(o.shape),u=lt(r.shape),h=cs(l,r.shape,a,1,i),d=new me(h.inShape,"float32"),p=d.values,[f,m,g,x]=d.strides,b=e.data.get(o.dataId).values,[w,y,C,$]=c,N=e.data.get(r.dataId).values,[T,k,v,I]=u,{batchSize:R,filterDepth:A,filterHeight:P,filterWidth:_,inChannels:B,inDepth:z,inHeight:W,inWidth:H,outChannels:K,outDepth:j,outHeight:X,outWidth:Y,strideDepth:tt,strideHeight:Q,strideWidth:ot}=h,at=A-1-h.padInfo.front,ct=P-1-h.padInfo.top,ht=_-1-h.padInfo.left;for(let gt=0;gt<R;++gt)for(let xt=0;xt<B;++xt)for(let Tt=0;Tt<z;++Tt){const $t=Tt-at,Xt=Math.max(0,Math.ceil($t/tt)),zt=Math.min(j,(A+$t)/tt);for(let _t=0;_t<W;++_t){const jt=_t-ct,Ht=Math.max(0,Math.ceil(jt/Q)),es=Math.min(X,(P+jt)/Q);for(let we=0;we<H;++we){const Ns=we-ht,tn=Math.max(0,Math.ceil(Ns/ot)),mo=Math.min(Y,(_+Ns)/ot);let vn=0;for(let ns=Xt;ns<zt;++ns){const Ts=ns*tt-$t;for(let Es=Ht;Es<es;++Es){const Rs=Es*Q-jt;for(let Xi=tn;Xi<mo;++Xi){const fp=Xi*ot-Ns,mp=w*gt+y*ns+C*Es+$*Xi,pH=T*(A-1-Ts)+k*(P-1-Rs)+v*(_-1-fp)+I*xt;for(let Ic=0;Ic<K;++Ic){const fH=b[mp+Ic],mH=N[pH+Ic];vn+=fH*mH}}}}p[f*gt+m*Tt+g*_t+x*we+xt]=vn}}}return e.makeTensorInfo(d.shape,d.dtype,d.values)}const rD={kernelName:Hc,backendName:"cpu",kernelFunc:oD};const iD=Ft(dr,n=>Math.cos(n)),aD={kernelName:dr,backendName:"cpu",kernelFunc:iD};const lD=Ft(pr,n=>Math.cosh(n)),cD={kernelName:pr,backendName:"cpu",kernelFunc:lD};function uD(n){const{inputs:t,backend:e,attrs:s}=n,{image:o,boxes:r,boxInd:i}=t,{cropSize:a,method:l,extrapolationValue:c}=s,[u,h,d,p]=o.shape,f=r.shape[0],[m,g]=a,x=wt([f,m,g,p],"float32"),b=e.data.get(r.dataId).values,w=e.data.get(i.dataId).values,y=e.data.get(o.dataId).values,C=lt(o.shape),$=lt(x.shape);for(let N=0;N<f;N++){const T=N*4,k=b[T],v=b[T+1],I=b[T+2],R=b[T+3],A=w[N];if(A>=u)continue;const P=m>1?(I-k)*(h-1)/(m-1):0,_=g>1?(R-v)*(d-1)/(g-1):0;for(let B=0;B<m;B++){const z=m>1?k*(h-1)+B*P:.5*(k+I)*(h-1);if(z<0||z>h-1){for(let W=0;W<g;W++)for(let H=0;H<p;H++){const K=H+W*$[2]+B*$[1]+N*$[0];x.values[K]=c}continue}if(l==="bilinear"){const W=Math.floor(z),H=Math.ceil(z),K=z-W;for(let j=0;j<g;j++){const X=g>1?v*(d-1)+j*_:.5*(v+R)*(d-1);if(X<0||X>d-1){for(let ot=0;ot<p;ot++){const at=ot+j*$[2]+B*$[1]+N*$[0];x.values[at]=c}continue}const Y=Math.floor(X),tt=Math.ceil(X),Q=X-Y;for(let ot=0;ot<p;ot++){let at=ot+Y*C[2]+W*C[1]+A*C[0];const ct=y[at];at=ot+tt*C[2]+W*C[1]+A*C[0];const ht=y[at];at=ot+Y*C[2]+H*C[1]+A*C[0];const gt=y[at];at=ot+tt*C[2]+H*C[1]+A*C[0];const xt=y[at],Tt=ct+(ht-ct)*Q,$t=gt+(xt-gt)*Q;at=ot+j*$[2]+B*$[1]+N*$[0],x.values[at]=Tt+($t-Tt)*K}}}else for(let W=0;W<g;++W){const H=g>1?v*(d-1)+W*_:.5*(v+R)*(d-1);if(H<0||H>d-1){for(let X=0;X<p;X++){const Y=X+W*$[2]+B*$[1]+N*$[0];x.values[Y]=c}continue}const K=Math.round(H),j=Math.round(z);for(let X=0;X<p;X++){const Y=X+K*C[2]+j*C[1]+A*C[0],tt=X+W*$[2]+B*$[1]+N*$[0];x.values[tt]=y[Y]}}}}return e.makeTensorInfo(x.shape,x.dtype,x.values)}const hD={kernelName:Kc,backendName:"cpu",kernelFunc:uD};function dD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;rt(o,"cumprod");const l=qt([r],o.shape.length);let c=o;l!=null&&(c=ze({inputs:{x:o},backend:e,attrs:{perm:l}}));const u=Jt(1,o.shape.length)[0];if(u!==c.shape.length-1)throw new Error(`backend.cumprod in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${u}`);const h=Ue(c.dtype,"int32"),d=Ac(q(c.shape),h),p=e.data.get(c.dataId).values,f=c.shape[c.shape.length-1],m=a?(x,b)=>x+f-b-1:(x,b)=>x+b;for(let x=0;x<p.length;x+=f)for(let b=0;b<f;b++){const w=m(x,b);if(b===0)d[w]=i?1:p[w];else{const y=m(x,b-1);d[w]=i?p[y]*d[y]:p[w]*d[y]}}const g=e.makeTensorInfo(c.shape,h,d);if(l!=null){const x=us(l),b=ze({inputs:{x:g},backend:e,attrs:{perm:x}});return e.disposeIntermediateTensorInfo(g),e.disposeIntermediateTensorInfo(c),b}return g}const pD={kernelName:qc,backendName:"cpu",kernelFunc:dD};function fD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;rt(o,"cumsum");const l=qt([r],o.shape.length);let c=o;l!=null&&(c=ze({inputs:{x:o},backend:e,attrs:{perm:l}}));const u=Jt(1,o.shape.length)[0];if(u!==c.shape.length-1)throw new Error(`backend.cumsum in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${u}`);const h=Ue(c.dtype,"int32"),d=Ie(q(c.shape),h),p=e.data.get(c.dataId).values,f=c.shape[c.shape.length-1],m=a?(x,b)=>x+f-b-1:(x,b)=>x+b;for(let x=0;x<p.length;x+=f)for(let b=0;b<f;b++){const w=m(x,b);if(b===0)d[w]=i?0:p[w];else{const y=m(x,b-1);d[w]=i?p[y]+d[y]:p[w]+d[y]}}const g=e.makeTensorInfo(c.shape,h,d);if(l!=null){const x=us(l),b=ze({inputs:{x:g},backend:e,attrs:{perm:x}});return e.disposeIntermediateTensorInfo(g),e.disposeIntermediateTensorInfo(c),b}return g}const mD={kernelName:ca,backendName:"cpu",kernelFunc:fD};function gD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i,binaryOutput:a}=s;if(o.shape.length===1){const l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=Pd(l,c,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,u)}else if(o.shape.length===2){const l=e.bufferSync(o),c=e.bufferSync(r),u=a0(l,c,i,a);return e.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const xD={kernelName:jc,backendName:"cpu",kernelFunc:gD};function bD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockSize:r,dataFormat:i}=s;S(i==="NHWC",()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${i}`);const a=o.shape[0],l=o.shape[1],c=o.shape[2],u=o.shape[3],h=l*r,d=c*r,p=u/(r*r),f=e.data.get(o.dataId).values,m=new Float32Array(a*h*d*p);let g=0;for(let x=0;x<a;++x)for(let b=0;b<h;++b){const w=Math.floor(b/r),y=b%r;for(let C=0;C<d;++C){const $=Math.floor(C/r),N=C%r,T=(y*r+N)*p;for(let k=0;k<p;++k){const I=k+T+u*($+c*(w+l*x));m[g++]=f[I]}}}return e.makeTensorInfo([a,h,d,p],o.dtype,m)}const yD={kernelName:Xc,backendName:"cpu",kernelFunc:bD};function h1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l,dimRoundingMode:c}=s;rt([o,r],"depthwiseConv2DNative");const u=lt(o.shape),h=lt(r.shape);let d=l;d==null&&(d=[1,1]),S($e(i,d),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${d}'`);const p=ge(o.shape,r.shape,i,d,a,c,!0),{filterHeight:f,filterWidth:m,dilationHeight:g,dilationWidth:x,padInfo:b}=p,w=b.left,y=b.top,C=p.outChannels/p.inChannels,$=new me(p.outShape,o.dtype),N=e.data.get(o.dataId).values,T=e.data.get(r.dataId).values,k=$.values;for(let v=0;v<p.batchSize;++v){const I=v*u[0],R=v*$.strides[0];for(let A=0;A<p.outHeight;++A){const P=R+A*$.strides[1],_=A*p.strideHeight-y;for(let B=0;B<f;++B){const z=_+B*g;if(z<0||z>=p.inHeight)continue;const W=B*h[0],H=I+z*u[1];for(let K=0;K<p.outWidth;++K){const j=P+K*$.strides[2],X=K*p.strideWidth-w;for(let Y=0;Y<m;++Y){const tt=X+Y*x;if(tt<0||tt>=p.inWidth)continue;const Q=W+Y*h[1],ot=H+tt*p.inChannels;let at=j,ct=Q;for(let ht=0;ht<p.inChannels;++ht){const gt=N[ot+ht];for(let xt=0;xt<C;++xt)k[at+xt]+=gt*T[ct+xt];at+=C,ct+=C}}}}}}return e.makeTensorInfo($.shape,$.dtype,$.values)}const wD={kernelName:ua,backendName:"cpu",kernelFunc:h1};function CD(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,filterShape:u}=s;rt([o,r],"depthwiseConv2dNativeBackpropFilter");const h=ge(o.shape,u,i,a,l,c,!0),{strideHeight:d,strideWidth:p,filterHeight:f,filterWidth:m}=h,g=new me(h.filterShape,"float32"),x=h.padInfo.left,b=h.padInfo.top,w=h.outChannels/h.inChannels,y=e.data.get(o.dataId).values,C=new me(o.shape,o.dtype,y),$=e.data.get(r.dataId).values,N=new me(r.shape,r.dtype,$);for(let T=0;T<f;++T){const k=Math.max(0,Math.ceil((b-T)/d)),v=Math.min(h.outHeight,(h.inHeight+b-T)/d);for(let I=0;I<m;++I){const R=Math.max(0,Math.ceil((x-I)/p)),A=Math.min(h.outWidth,(h.inWidth+x-I)/p);for(let P=0;P<h.outChannels;++P){const _=Math.trunc(P/w),B=P%w;let z=0;for(let W=0;W<h.batchSize;++W)for(let H=k;H<v;++H){const K=T+H*d-b;for(let j=R;j<A;++j){const X=I+j*p-x;z+=C.get(W,K,X,_)*N.get(W,H,j,P)}}g.set(z,T,I,_,B)}}}return e.makeTensorInfo(g.shape,g.dtype,g.values)}const ID={kernelName:Yc,backendName:"cpu",kernelFunc:CD};function $D(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,inputShape:u}=s;rt([o,r],"depthwiseConv2DNativeBackpropInput");const h=lt(o.shape),d=lt(r.shape),p=ge(u,r.shape,i,a,l,c,!0),f=new me(p.inShape,"float32"),m=f.values,[g,x,b]=f.strides,w=e.data.get(o.dataId).values,[y,C,$]=h,N=e.data.get(r.dataId).values,[T,k,v]=d,{batchSize:I,filterHeight:R,filterWidth:A,inChannels:P,inHeight:_,inWidth:B,outChannels:z,outHeight:W,outWidth:H,strideHeight:K,strideWidth:j}=p,X=R-1-p.padInfo.top,Y=A-1-p.padInfo.left,tt=z/P;for(let Q=0;Q<I;++Q)for(let ot=0;ot<P;++ot)for(let at=0;at<_;++at){const ct=at-X,ht=Math.max(0,Math.ceil(ct/K)),gt=Math.min(W,(R+ct)/K);for(let xt=0;xt<B;++xt){const Tt=xt-Y,$t=Math.max(0,Math.ceil(Tt/j)),Xt=Math.min(H,(A+Tt)/j);let zt=0;for(let _t=ht;_t<gt;++_t){const jt=_t*K-ct;for(let Ht=$t;Ht<Xt;++Ht){const es=Ht*j-Tt,we=y*Q+C*_t+$*Ht,Ns=T*(R-1-jt)+k*(A-1-es)+v*ot;for(let tn=0;tn<tt;++tn){const mo=ot*tt+tn,vn=w[we+mo],ns=N[Ns+tn];zt+=vn*ns}}}m[g*Q+x*at+b*xt+ot]=zt}}return e.makeTensorInfo(f.shape,f.dtype,f.values)}const kD={kernelName:Zc,backendName:"cpu",kernelFunc:$D};function vD(n){const{inputs:t,backend:e}=n,{x:s}=t,o=q(s.shape),r=e.data.get(s.dataId).values,i=wt([o,o],s.dtype),a=i.values;for(let c=0;c<r.length;c++)a[c*o+c]=r[c];const l=[...s.shape,...s.shape];return e.makeTensorInfo(l,i.dtype,i.values)}const SD={kernelName:Sp,backendName:"cpu",kernelFunc:vD};const ND={kernelName:ha,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o}=n,{strides:r,pad:i,dilations:a}=e,l=t,c=l.data.get(s.dataId).values,u=s.shape.length,h=l.data.get(o.dataId).values,d=o.shape.length,{batchSize:p,inHeight:f,inWidth:m,inChannels:g,outHeight:x,outWidth:b,padInfo:w,strideHeight:y,strideWidth:C,filterHeight:$,filterWidth:N,dilationHeight:T,dilationWidth:k,outShape:v}=si(s.shape,o.shape,r,i,"NHWC",a),I=q(v),R=v.length,A=Yt(s.dtype,I);for(let _=0;_<p;++_)for(let B=0;B<x;++B){const z=B*y-w.top;for(let W=0;W<b;++W){const H=W*C-w.left;for(let K=0;K<g;++K){let j=Number.MIN_SAFE_INTEGER;for(let Y=0;Y<$;++Y){const tt=z+Y*T;if(tt>=0&&tt<f)for(let Q=0;Q<N;++Q){const ot=H+Q*k;if(ot>=0&&ot<m){const at=Sn([_,tt,ot,K],u,lt(s.shape)),ct=Sn([Y,Q,K],d,lt(o.shape)),ht=c[at]+h[ct];ht>j&&(j=ht)}}}const X=Sn([_,B,W,K],R,lt(v));A[X]=j}}}return{dataId:l.write(Os(A,s.dtype),v,s.dtype),shape:v,dtype:s.dtype}}};const TD={kernelName:Qc,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o,dy:r}=n,{strides:i,pad:a,dilations:l}=e,c=t,u=pn(s.shape,c.data.get(s.dataId).values),h=pn(o.shape,c.data.get(o.dataId).values),{batchSize:d,inHeight:p,inWidth:f,inChannels:m,outHeight:g,outWidth:x,padInfo:b,strideHeight:w,strideWidth:y,filterHeight:C,filterWidth:$,dilationHeight:N,dilationWidth:T,outShape:k}=si(s.shape,o.shape,i,a,"NHWC",l);S(r.rank===k.length,()=>`Error in ${Qc}, dy must have the same rank as output ${k.length}, but got ${r.rank}`);const v=pn(k,c.data.get(r.dataId).values),I=Cp(o.shape,o.dtype);for(let A=0;A<d;++A)for(let P=0;P<g;++P){const _=P*w-b.top;for(let B=0;B<x;++B){const z=B*y-b.left;for(let W=0;W<m;++W){let H=Number.MIN_SAFE_INTEGER,K=0,j=0;for(let X=0;X<C;++X){const Y=_+X*N;if(Y>=0&&Y<p)for(let tt=0;tt<$;++tt){const Q=z+tt*T;if(Q>=0&&Q<f){const ot=u[A][Y][Q][W]+h[X][tt][W];ot>H&&(H=ot,K=X,j=tt)}}}I[K][j][W]+=v[A][P][B][W]}}}return{dataId:c.write(Os(I,s.dtype),o.shape,o.dtype),shape:o.shape,dtype:o.dtype}}};const ED={kernelName:Jc,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o,dy:r}=n,{strides:i,pad:a,dilations:l}=e,c=t,u=pn(s.shape,c.data.get(s.dataId).values),h=pn(o.shape,c.data.get(o.dataId).values),{batchSize:d,inHeight:p,inWidth:f,inChannels:m,outHeight:g,outWidth:x,padInfo:b,strideHeight:w,strideWidth:y,filterHeight:C,filterWidth:$,dilationHeight:N,dilationWidth:T,outShape:k}=si(s.shape,o.shape,i,a,"NHWC",l);S(r.rank===k.length,()=>`Error in ${Jc}, dy must have the same rank as output ${k.length}, but got ${r.rank}`);const v=pn(k,c.data.get(r.dataId).values),I=Cp(s.shape,s.dtype);for(let A=0;A<d;++A)for(let P=0;P<g;++P){const _=P*w-b.top;for(let B=0;B<x;++B){const z=B*y-b.left;for(let W=0;W<m;++W){let H=Number.MIN_SAFE_INTEGER,K=_<0?0:_,j=z<0?0:z;for(let X=0;X<C;++X){const Y=_+X*N;if(Y>=0&&Y<p)for(let tt=0;tt<$;++tt){const Q=z+tt*T;if(Q>=0&&Q<f){const ot=u[A][Y][Q][W]+h[X][tt][W];ot>H&&(H=ot,K=Y,j=Q)}}}I[A][K][j][W]+=v[A][P][B][W]}}}return{dataId:c.write(Os(I,s.dtype),s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};function RD(n){const{inputs:t,backend:e,attrs:s}=n,{image:o}=t,{canvas:r,options:i}=s,{contextOptions:a,imageOptions:l}=i||{},c=l?.alpha||1,u=a?.contextType||"2d";if(u!=="2d")throw new Error(`Context type ${a.contextType} is not supported by the CPU backend.`);const h=r.getContext(u,a?.contextAttributes||{});if(h==null)throw new Error(`Could not get the context with ${u} type.`);const[d,p]=o.shape.slice(0,2),f=o.shape.length===2?1:o.shape[2],m=e.data.get(o.dataId).values,g=o.dtype==="float32"?255:1,x=new Uint8ClampedArray(p*d*4);for(let w=0;w<d*p;++w){const y=[0,0,0,255*c];for(let $=0;$<f;$++){const N=m[w*f+$];if(o.dtype==="float32"){if(N<0||N>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${N}.`)}else if(o.dtype==="int32"&&(N<0||N>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${N}.`);f===1?(y[0]=N*g,y[1]=N*g,y[2]=N*g):y[$]=N*g}const C=w*4;x[C+0]=Math.round(y[0]),x[C+1]=Math.round(y[1]),x[C+2]=Math.round(y[2]),x[C+3]=Math.round(y[3])}r.width=p,r.height=d;const b=new ImageData(x,p,d);return h.putImageData(b,0,0),o}const AD={kernelName:rw,backendName:"cpu",kernelFunc:RD};function _i(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;rt(o,"sum");let a;o.dtype==="bool"?a=Is({inputs:{x:o},backend:e,attrs:{dtype:"int32"}}):a=zn({inputs:{x:o},backend:e});const l=a.shape.length,c=yt(r,a.shape),u=qt(c,l);let h=c,d=a;u!=null&&(d=ze({inputs:{x:a},backend:e,attrs:{perm:u}}),h=Jt(h.length,l)),xe("sum",h,d.shape.length);const[p,f]=de(d.shape,h),m=Ue(d.dtype,"int32");let g=Jl(e,p,m);const x=q(f),b=e.data.get(g.dataId).values,w=e.data.get(d.dataId).values;for(let y=0;y<b.length;++y){const C=y*x;let $=0;for(let N=0;N<x;++N)$+=w[C+N];b[y]=$}if(i){const y=ne(g.shape,c),C=g;g=Bt({inputs:{x:g},backend:e,attrs:{shape:y}}),e.disposeIntermediateTensorInfo(C)}return e.disposeIntermediateTensorInfo(a),u!=null&&e.disposeIntermediateTensorInfo(d),g}const DD={kernelName:Ga,backendName:"cpu",kernelFunc:_i};function FD(n){const{inputs:t,backend:e,attrs:s}=n,{equation:o}=s,r=t,{allDims:i,summedDims:a,idDims:l}=zh(o,r.length);Wh(i.length,l,r);const{path:c,steps:u}=Uh(a,l),h=u.length;let d=null,p=i.length;const f=[];for(let m=0;m<h;++m){for(const g of u[m]){const{permutationIndices:x,expandDims:b}=Vh(p,l[g]);let w;Gh(x)?w=r[g]:(w=ze({inputs:{x:r[g]},backend:e,attrs:{perm:x}}),f.push(w));const y=w.shape.slice();for(let C=0;C<b.length;++C)y.splice(b[C],0,1);Et(w.shape,y)||(w=Bt({inputs:{x:w},backend:e,attrs:{shape:y}}),f.push(w)),d===null?d=w:(d=Ql({inputs:{a:w,b:d},backend:e}),f.push(d))}m<h-1&&(c[m]>=0&&(d=_i({inputs:{x:d},backend:e,attrs:{axis:c[m]-(i.length-p),keepDims:!1}}),f.push(d)),p--)}for(const m of f)m!==d&&e.disposeIntermediateTensorInfo(m);return d}const OD={kernelName:tu,backendName:"cpu",kernelFunc:FD};function _D(n){const{inputs:t,backend:e}=n,{dy:s,y:o}=t;rt([s,o],"eluGrad");const r=new Float32Array(q(o.shape)),i=e.data.get(o.dataId).values,a=e.data.get(s.dataId).values;for(let l=0;l<i.length;++l){const c=i[l];c>=0?r[l]=a[l]:r[l]=a[l]*(c+1)}return e.makeTensorInfo(o.shape,"float32",r)}const LD={kernelName:eu,backendName:"cpu",kernelFunc:_D};const MD=Dh,PD=Fh,BD=Oh,zD=_h,VD=Lh,WD=Mh,UD=Ft(gr,n=>{const t=Math.sign(n),e=Math.abs(n),s=1/(1+MD*e);return t*(1-((((WD*s+VD)*s+zD)*s+BD)*s+PD)*s*Math.exp(-e*e))}),GD={kernelName:gr,backendName:"cpu",kernelFunc:UD};function nc(n){const{inputs:t,backend:e,attrs:s}=n,{input:o}=t,{dim:r}=s,i=o.shape.length,a=o.shape.slice();let l=r;return r<0&&(S(-(i+1)<=r,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+r+1),a.splice(l,0,1),Bt({inputs:{x:o},backend:e,attrs:{shape:a}})}const HD={kernelName:pa,backendName:"cpu",kernelFunc:nc};const qD=ee((n,t)=>n/t),Gd=ue(fr,qD),Hd={kernelName:fr,backendName:"cpu",kernelFunc:Gd};function d1(n,t,e){const s=n.shape,o=s[0],r=s[1],i=e.data.get(n.dataId),a=i.complexTensorInfos.real,l=i.complexTensorInfos.imag,c=[o,r],u=q(c),h=Ce("float32",u),d=Ce("float32",u);for(let g=0;g<o;g++){const x=io({inputs:{x:a},backend:e,attrs:{begin:[g,0],size:[1,r]}}),b=io({inputs:{x:l},backend:e,attrs:{begin:[g,0],size:[1,r]}}),w=qe({inputs:{real:x,imag:b},backend:e}),{real:y,imag:C}=KD(w,t,e),$=Yn(y,C);for(let N=0;N<r;N++){const T=Ph($,N);h[g*r+N]=T.real,d[g*r+N]=T.imag}e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(b),e.disposeIntermediateTensorInfo(w)}const p=e.makeTensorInfo(c,"float32",h),f=e.makeTensorInfo(c,"float32",d),m=qe({inputs:{real:p,imag:f},backend:e});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),m}function KD(n,t,e){const s=q(n.shape),o=e.data.get(n.dataId),r=e.data.get(o.complexTensorInfos.real.dataId).values,i=e.data.get(o.complexTensorInfos.imag.dataId).values;if(jD(s)){const a=qd(r,i,s,t,e),l=[n.shape[0],n.shape[1]];if(t){const c=e.makeTensorInfo(l,"float32",a.real),u=e.makeTensorInfo(l,"float32",a.imag),h=e.makeTensorInfo([],"float32",rs(s,"float32")),d=zn({inputs:{x:h},backend:e}),p=Hd.kernelFunc({inputs:{a:c,b:h},backend:e}),f=Hd.kernelFunc({inputs:{a:u,b:d},backend:e}),m=e.data.get(p.dataId).values,g=e.data.get(f.dataId).values;return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),{real:m,imag:g}}return a}else{const a=Yn(r,i),l=XD(a,s,t);return Hm(l)}}function jD(n){return(n&n-1)===0}function qd(n,t,e,s,o){if(e===1)return{real:n,imag:t};const r=Yn(n,t),i=e/2,a=qm(r),l=a.real,c=a.imag,u=[l.length],h=o.makeTensorInfo(u,"float32",l),d=o.makeTensorInfo(u,"float32",c),p=qe({inputs:{real:h,imag:d},backend:o}),f=Km(r),m=f.real,g=f.imag,x=[m.length],b=o.makeTensorInfo(x,"float32",m),w=o.makeTensorInfo(x,"float32",g),y=qe({inputs:{real:b,imag:w},backend:o}),C=qd(l,c,i,s,o),$=C.real,N=C.imag,T=[$.length],k=o.makeTensorInfo(T,"float32",$),v=o.makeTensorInfo(T,"float32",N),I=qe({inputs:{real:k,imag:v},backend:o}),R=qd(m,g,i,s,o),A=R.real,P=R.imag,_=[A.length],B=o.makeTensorInfo(_,"float32",A),z=o.makeTensorInfo(_,"float32",P),W=qe({inputs:{real:B,imag:z},backend:o}),H=Xm(e,s),K=[H.real.length],j=o.makeTensorInfo(K,"float32",H.real),X=o.makeTensorInfo(K,"float32",H.imag),Y=qe({inputs:{real:j,imag:X},backend:o}),tt=Ql({inputs:{a:Y,b:W},backend:o}),Q=zo({inputs:{a:I,b:tt},backend:o}),ot=Wd({inputs:{a:I,b:tt},backend:o}),at=oo({inputs:{input:Q},backend:o}),ct=oo({inputs:{input:ot},backend:o}),ht=Vo({inputs:{input:Q},backend:o}),gt=Vo({inputs:{input:ot},backend:o}),xt=Wo({inputs:[at,ct],backend:o,attrs:{axis:0}}),Tt=Wo({inputs:[ht,gt],backend:o,attrs:{axis:0}}),$t=o.data.get(xt.dataId).values,Xt=o.data.get(Tt.dataId).values;return o.disposeIntermediateTensorInfo(h),o.disposeIntermediateTensorInfo(d),o.disposeIntermediateTensorInfo(p),o.disposeIntermediateTensorInfo(b),o.disposeIntermediateTensorInfo(w),o.disposeIntermediateTensorInfo(y),o.disposeIntermediateTensorInfo(k),o.disposeIntermediateTensorInfo(v),o.disposeIntermediateTensorInfo(I),o.disposeIntermediateTensorInfo(B),o.disposeIntermediateTensorInfo(z),o.disposeIntermediateTensorInfo(W),o.disposeIntermediateTensorInfo(j),o.disposeIntermediateTensorInfo(X),o.disposeIntermediateTensorInfo(Y),o.disposeIntermediateTensorInfo(tt),o.disposeIntermediateTensorInfo(Q),o.disposeIntermediateTensorInfo(ot),o.disposeIntermediateTensorInfo(at),o.disposeIntermediateTensorInfo(ht),o.disposeIntermediateTensorInfo(ct),o.disposeIntermediateTensorInfo(gt),o.disposeIntermediateTensorInfo(xt),o.disposeIntermediateTensorInfo(Tt),{real:$t,imag:Xt}}function XD(n,t,e){const s=new Float32Array(t*2);for(let o=0;o<t;o++){let r=0,i=0;for(let a=0;a<t;a++){const l=Ym(o*a,t,e),c=Ph(n,a);r+=c.real*l.real-c.imag*l.imag,i+=c.real*l.imag+c.imag*l.real}e&&(r/=t,i/=t),jm(s,r,i,o)}return s}function YD(n){const{inputs:t,backend:e}=n,{input:s}=t,o=q(s.shape),r=s.shape[s.shape.length-1],i=o/r,a=Bt({inputs:{x:s},backend:e,attrs:{shape:[i,r]}}),l=d1(a,!1,e),c=Bt({inputs:{x:l},backend:e,attrs:{shape:s.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(l),c}const ZD={kernelName:nu,backendName:"cpu",kernelFunc:YD};function Kd(n){const{backend:t,attrs:e}=n,{shape:s,value:o,dtype:r}=e,i=r||yo(o),a=Yt(i,q(s));return QD(a,o,i),t.makeTensorInfo(s,i,a)}const JD={kernelName:su,backendName:"cpu",kernelFunc:Kd};function QD(n,t,e){n.fill(t)}const tF={kernelName:ou,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,o=e,r=Ce(s.dtype,q(s.shape)),[i,a,l,c]=s.shape,u=o.data.get(s.dataId).values;for(let d=0;d<i;d++){const p=d*l*a*c;for(let f=0;f<a;f++){const m=f*(l*c);for(let g=0;g<l;g++){const x=g*c;for(let b=0;b<c;b++){const w=Math.round(l-g-1),y=p+m+x+b;let C=u[y];if(w>=0&&w<l){const $=w*c,N=p+m+$+b;C=u[N]}r[y]=C}}}}return{dataId:o.write(r,s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};function eF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s;let m=u1({inputs:{x:o,filter:r},backend:e,attrs:{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d}});if(i){const g=m;if(u==="NCHW"&&i.shape.length===1&&i.shape[0]!==1){const x=Bt({inputs:{x:i},backend:e,attrs:{shape:[i.shape[0],1,1]}});m=zo({inputs:{a:m,b:x},backend:e}),e.disposeIntermediateTensorInfo(x)}else m=zo({inputs:{a:m,b:i},backend:e});e.disposeIntermediateTensorInfo(g)}if(p){const g=m;if(u==="NCHW"&&p==="prelu"&&a.shape.length===1&&a.shape[0]!==1){const x=Bt({inputs:{x:a},backend:e,attrs:{shape:[a.shape[0],1,1]}});m=ec(e,m,p,x,f),e.disposeIntermediateTensorInfo(x)}else m=ec(e,m,p,a,f);e.disposeIntermediateTensorInfo(g)}return m}const nF={kernelName:Ja,backendName:"cpu",kernelFunc:eF};function sF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s;let m=h1({inputs:{x:o,filter:r},backend:e,attrs:{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d}});if(i){const g=m;m=zo({inputs:{a:m,b:i},backend:e}),e.disposeIntermediateTensorInfo(g)}if(p){const g=m;m=ec(e,m,p,a,f),e.disposeIntermediateTensorInfo(g)}return m}const oF={kernelName:Hp,backendName:"cpu",kernelFunc:sF};function rF(n){const{inputs:t,backend:e}=n,{params:s,indices:o}=t,r=q(s.shape),i=o.shape,a=i[i.length-1],[l,c,u,h]=wh(s,o);if(c===0)return e.makeTensorInfo(l,s.dtype,[]);const d=e.data.get(o.dataId).values,p=e.bufferSync(s),f=b0(d,p,s.dtype,c,a,u,h,s.shape,r);return e.makeTensorInfo(l,s.dtype,f.values)}const iF={kernelName:Np,backendName:"cpu",kernelFunc:rF};function aF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,indices:r}=t,{axis:i,batchDims:a}=s;rt([o,r],"gatherV2");const l=yt(i,o.shape)[0],c=e.data.get(r.dataId).values,u=o.shape[l];for(let y=0;y<c.length;++y){const C=c[y];S(C<=u-1&&C>=0,()=>`GatherV2: the index value ${C} is not in [0, ${u-1}]`)}let h=a;a==null&&(h=0);const d=q(r.shape),p=Kh(o,r,l,h),f=Bt({inputs:{x:o},backend:e,attrs:{shape:[p.batchSize,p.outerSize,p.dimSize,p.sliceSize]}}),m=Bt({inputs:{x:r},backend:e,attrs:{shape:[p.batchSize,d/p.batchSize]}}),g=[p.batchSize,p.outerSize,d/p.batchSize,p.sliceSize],x=e.bufferSync(m),b=e.bufferSync(f),w=y0(b,x,g);return e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),e.makeTensorInfo(p.outputShape,w.dtype,w.values)}const lF={kernelName:ma,backendName:"cpu",kernelFunc:aF};function cF(n){const{inputs:t,backend:e}=n,{input:s}=t,o=q(s.shape),r=s.shape[s.shape.length-1],i=o/r,a=Bt({inputs:{x:s},backend:e,attrs:{shape:[i,r]}}),l=d1(a,!0,e),c=Bt({inputs:{x:l},backend:e,attrs:{shape:s.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(l),c}const uF={kernelName:ru,backendName:"cpu",kernelFunc:cF};const hF=Ft($r,n=>Number.isFinite(n)?1:0,"bool"),dF={kernelName:$r,backendName:"cpu",kernelFunc:hF};const pF=Ft(kr,n=>Math.abs(n)===1/0?1:0,"bool"),fF={kernelName:kr,backendName:"cpu",kernelFunc:pF};const mF=Ft(vr,n=>Number.isNaN(n)?1:0,"bool"),gF={kernelName:vr,backendName:"cpu",kernelFunc:mF};function xF(n){const{backend:t,attrs:e}=n,{start:s,stop:o,num:r}=e,i=k0(s,o,r);return t.makeTensorInfo([i.length],"float32",i)}const bF={kernelName:Tp,backendName:"cpu",kernelFunc:xF};const yF=Ft(Nr,n=>Math.log1p(n)),wF={kernelName:Nr,backendName:"cpu",kernelFunc:yF};const CF=ee((n,t)=>n&&t),IF=ue(wa,CF,null,"bool"),$F={kernelName:wa,backendName:"cpu",kernelFunc:IF};const kF=Ft(Ca,n=>n?0:1,"bool"),vF={kernelName:Ca,backendName:"cpu",kernelFunc:kF};const SF=ee((n,t)=>n||t),NF=ue(Ia,SF,null,"bool"),TF={kernelName:Ia,backendName:"cpu",kernelFunc:NF};function EF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{depthRadius:r,bias:i,alpha:a,beta:l}=s;rt(o,"LRN");const c=o.shape[3],u=c-1,h=e.data.get(o.dataId).values,d=q(o.shape),p=new Float32Array(d);function f(m){const g=m%c;let x=m-g+Math.max(0,g-r);const b=m-g+Math.min(g+r,u);let w=0;for(;x<=b;x++){const y=h[x];w+=y*y}return w}for(let m=0;m<d;m++){const g=f(m),x=h[m]*Math.pow(i+a*g,-l);p[m]=x}return e.makeTensorInfo(o.shape,o.dtype,p)}const RF={kernelName:$a,backendName:"cpu",kernelFunc:EF};function AF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,y:r,dy:i}=t,{depthRadius:a,bias:l,alpha:c,beta:u}=s;rt(i,"LRNGrad");const h=q(i.shape),d=i.shape[3],p=e.data.get(i.dataId).values,f=e.data.get(o.dataId).values,m=e.data.get(r.dataId).values,g=new Float32Array(h),x=h;for(let b=0;b<x;b++){const w=b%d,y=b-w+Math.max(0,w-a),C=b-w+Math.min(d,w+a+1);let $=0;for(let N=y;N<C;N++)$+=Math.pow(f[N],2);$=c*$+l;for(let N=y;N<C;N++){let T=-2*c*u*f[N]*m[b]/$;b===N&&(T+=Math.pow($,-u)),T*=p[b],g[N]+=T}}return e.makeTensorInfo(i.shape,o.dtype,g)}const DF={kernelName:au,backendName:"cpu",kernelFunc:AF};function p1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reductionIndices:r,keepDims:i}=s,a=e;let l=o.shape;const c=l.length,u=yt(r,l);let h=u;const d=qt(h,c);let p=a.data.get(o.dataId).values;if(d!=null){const y=new Array(c);for(let C=0;C<y.length;C++)y[C]=l[d[C]];p=zd(p,l,o.dtype,d,y),h=Jt(h.length,c),l=y}rt(o,"max"),xe("max",h,c);const[f,m]=de(l,h),g=q(m),x=S0(p,g,f,o.dtype),b=a.write(x,f,o.dtype);let w=f;return i&&(w=ne(f,u)),{dataId:b,shape:w,dtype:o.dtype}}const FF={kernelName:ka,backendName:"cpu",kernelFunc:p1};function OF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;rt(o,"maxPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;S($e(i,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=nn(o.shape,r,i,c,a,l);let h;if(u.filterWidth===1&&u.filterHeight===1&&Et(u.inShape,u.outShape))h=zn({inputs:{x:o},backend:e});else{const d=e.data.get(o.dataId).values,p=lt(o.shape),f=Ud(d,o.shape,o.dtype,p,u,"max");h=e.makeTensorInfo(u.outShape,o.dtype,f.values)}return h}const _F={kernelName:va,backendName:"cpu",kernelFunc:OF};function LF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s;rt(o,"maxPool3d");const u=Hn(o.shape,r,i,1,a,l,c),h=e.data.get(o.dataId).values,d=c1(h,o.shape,o.dtype,lt(o.shape),u,"max");return e.makeTensorInfo(d.shape,"float32",d.values)}const MF={kernelName:Sa,backendName:"cpu",kernelFunc:LF};function PF(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,{filterSize:i,strides:a,pad:l,dimRoundingMode:c}=s;rt([o,r],"maxPool3DGrad");const u=Hn(r.shape,i,a,1,l,c),h=e.bufferSync(r),d=NA(h,u),p=u.strideDepth,f=u.strideHeight,m=u.strideWidth,g=u.dilationDepth,x=u.dilationHeight,b=u.dilationWidth,w=u.effectiveFilterDepth,y=u.effectiveFilterHeight,C=u.effectiveFilterWidth,$=w-1-u.padInfo.front,N=C-1-u.padInfo.left,T=y-1-u.padInfo.top,k=wt(r.shape,"float32"),v=e.bufferSync(o);for(let I=0;I<u.batchSize;++I)for(let R=0;R<u.inChannels;++R)for(let A=0;A<u.inDepth;++A)for(let P=0;P<u.inHeight;++P)for(let _=0;_<u.inWidth;++_){const B=A-$,z=P-T,W=_-N;let H=0;for(let K=0;K<w;K+=g){const j=(B+K)/p;if(!(j<0||j>=u.outDepth||Math.floor(j)!==j))for(let X=0;X<y;X+=x){const Y=(z+X)/f;if(!(Y<0||Y>=u.outHeight||Math.floor(Y)!==Y))for(let tt=0;tt<C;tt+=b){const Q=(W+tt)/m;if(Q<0||Q>=u.outWidth||Math.floor(Q)!==Q)continue;const ot=w*y*C-1-d.get(I,j,Y,Q,R),at=K*y*C+X*C+tt,ct=ot===at?1:0;if(ct===0)continue;const ht=v.get(I,j,Y,Q,R);H+=ht*ct}}}k.set(H,I,A,P,_,R)}return e.makeTensorInfo(k.shape,k.dtype,k.values)}const BF={kernelName:cu,backendName:"cpu",kernelFunc:PF};function zF(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r,output:i}=t,a=r;rt([r,i],"maxPoolGrad");const{filterSize:l,strides:c,pad:u,dimRoundingMode:h}=s,d=nn(a.shape,l,c,1,u,h),p=e.data.get(a.dataId).values,f=wt(d.outShape,a.dtype,l1(p,a.shape,a.dtype,d).values),m=d.strideHeight,g=d.strideWidth,x=d.dilationHeight,b=d.dilationWidth,w=d.effectiveFilterHeight,y=d.effectiveFilterWidth,C=y-1-d.padInfo.left,$=w-1-d.padInfo.top,N=wt(a.shape,"float32"),T=e.data.get(o.dataId).values,k=wt(o.shape,"float32",T);for(let v=0;v<d.batchSize;++v)for(let I=0;I<d.inChannels;++I)for(let R=0;R<d.inHeight;++R)for(let A=0;A<d.inWidth;++A){const P=R-$,_=A-C;let B=0;for(let z=0;z<w;z+=x){const W=(P+z)/m;if(!(W<0||W>=d.outHeight||Math.floor(W)!==W))for(let H=0;H<y;H+=b){const K=(_+H)/g;if(K<0||K>=d.outWidth||Math.floor(K)!==K)continue;const j=w*y-1-f.get(v,W,K,I),X=z*y+H,Y=j===X?1:0;if(Y===0)continue;const tt=k.get(v,W,K,I);B+=tt*Y}}N.set(B,v,R,A,I)}return e.makeTensorInfo(N.shape,N.dtype,N.values)}const VF={kernelName:lu,backendName:"cpu",kernelFunc:zF};function WF(n,t,e,s,o){const r=lt(t),i=Ud(n,t,e,r,o,"max"),a=l1(n,t,e,o,!0,s);return[i.values,a.values]}const UF={kernelName:Ep,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{filterSize:o,strides:r,pad:i,includeBatchInIndex:a}=t,l=e;rt(s,"MaxPoolWithArgmax");const c=l.data.get(s.dataId).values,u=nn(s.shape,o,r,[1,1],i),[h,d]=WF(c,s.shape,s.dtype,a,u),p=l.write(h,u.outShape,s.dtype),f=l.write(d,u.outShape,s.dtype);return[{dataId:p,shape:u.outShape,dtype:s.dtype},{dataId:f,shape:u.outShape,dtype:"int32"}]}};function GF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=yt(r,o.shape),c=de(o.shape,a)[1],u=q(c),h=[],d=e.makeTensorInfo([],"float32",new Float32Array([u]));h.push(d);const p=Is({inputs:{x:o},backend:e,attrs:{dtype:"float32"}});h.push(p);const f=Gd({inputs:{a:p,b:d},backend:e});h.push(f);const m=_i({inputs:{x:f},backend:e,attrs:{axis:r,keepDims:i}});return h.forEach(g=>e.disposeIntermediateTensorInfo(g)),m}const HF={kernelName:Na,backendName:"cpu",kernelFunc:GF};function qF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;rt(o,"min");const a=yt(r,o.shape);let l=a;const c=qt(l,o.shape.length);let u=o;c!=null&&(u=ze({inputs:{x:o},backend:e,attrs:{perm:c}}),l=Jt(l.length,o.shape.length)),xe("min",l,u.shape.length);const[h,d]=de(u.shape,l),p=q(d),f=Ie(q(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const C=m[b+y];(Number.isNaN(C)||C<w)&&(w=C)}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ne(h,a),b=Bt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const KF={kernelName:Ta,backendName:"cpu",kernelFunc:qF};function jF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,mode:i}=s;rt(o,"mirrorPad");const a=r.map((w,y)=>w[0]+o.shape[y]+w[1]),l=r.map(w=>w[0]),c=r.map((w,y)=>w[0]+o.shape[y]),u=i==="reflect"?0:1,h=e.data.get(o.dataId).values,d=o.shape.length,p=lt(o.shape),f=q(a),m=a.length,g=lt(a),x=Ce(o.dtype,f);for(let w=0;w<f;w++){let y=wo(w,m,g);for(let $=0;$<m;$++)y[$]<l[$]?y[$]=l[$]*2-y[$]-u:y[$]>=c[$]&&(y[$]=(c[$]-1)*2-y[$]+u);y=y.map(($,N)=>$-l[N]);const C=Sn(y,d,p);x[w]=h[C]}return{dataId:e.write(x,a,o.dtype),shape:a,dtype:o.dtype}}const XF={kernelName:Ea,backendName:"cpu",kernelFunc:jF};const YF=ee(((n,t)=>{const e=n%t;return n<0&&t<0||n>=0&&t>=0?e:(e+t)%t})),ZF=ue(Rr,YF),JF={kernelName:Rr,backendName:"cpu",kernelFunc:ZF};function f1(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{dim:r}=s,i=o.shape.length;let a=r;if(a===-1&&(a=i-1),a!==i-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${i} and dim was ${a}`);const l=yt([a],o.shape),c=p1({inputs:{x:o},backend:e,attrs:{reductionIndices:l,keepDims:!1}}),u=ne(c.shape,l),h=Bt({inputs:{x:c},backend:e,attrs:{shape:u}}),d=Wd({inputs:{a:o,b:h},backend:e}),p=f0({inputs:{x:d},backend:e}),f=_i({inputs:{x:p},backend:e,attrs:{axis:l,keepDims:!1}}),m=Bt({inputs:{x:f},backend:e,attrs:{shape:u}}),g=Gd({inputs:{a:p,b:m},backend:e});return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),g}const QF={kernelName:Ka,backendName:"cpu",kernelFunc:f1};function tO(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{numSamples:r,seed:i,normalized:a}=s;rt(o,"multinomial");const l=a?o:f1({inputs:{logits:o},backend:e,attrs:{dim:-1}}),c=l.shape[0],u=l.shape[1],h=e.data.get(l.dataId).values,d=[c,r],p=Ie(q(d),"int32");for(let f=0;f<c;++f){const m=f*u,g=new Float32Array(u-1);g[0]=h[m];for(let w=1;w<g.length;++w)g[w]=g[w-1]+h[m+w];const x=rh.alea(i.toString()),b=f*r;for(let w=0;w<r;++w){const y=x();p[b+w]=g.length;for(let C=0;C<g.length;C++)if(y<g[C]){p[b+w]=C;break}}}return a||e.disposeIntermediateTensorInfo(l),e.makeTensorInfo(d,"int32",p)}const eO={kernelName:Rp,backendName:"cpu",kernelFunc:tO};const nO=mh;function sO(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l}=s;rt(o,"NonMaxSuppression");const c=e.data.get(o.dataId).values,u=e.data.get(r.dataId).values,{selectedIndices:h}=nO(c,u,i,a,l);return e.makeTensorInfo([h.length],"int32",new Int32Array(h))}const oO={kernelName:uu,backendName:"cpu",kernelFunc:sO};const rO=gh;function iO(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,padToMaxOutputSize:c}=s;rt(o,"NonMaxSuppressionPadded");const u=e.data.get(o.dataId).values,h=e.data.get(r.dataId).values,{selectedIndices:d,validOutputs:p}=rO(u,h,i,a,l,c);return[e.makeTensorInfo([d.length],"int32",new Int32Array(d)),e.makeTensorInfo([],"int32",new Int32Array([p]))]}const aO={kernelName:hu,backendName:"cpu",kernelFunc:iO};const lO=xh;function cO(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,softNmsSigma:c}=s;rt(o,"NonMaxSuppressionWithScore");const u=e.data.get(o.dataId).values,h=e.data.get(r.dataId).values,d=i,p=a,f=l,m=c,{selectedIndices:g,selectedScores:x}=lO(u,h,d,p,f,m);return[e.makeTensorInfo([g.length],"int32",new Int32Array(g)),e.makeTensorInfo([x.length],"float32",new Float32Array(x))]}const uO={kernelName:du,backendName:"cpu",kernelFunc:cO};function hO(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o}=t,{dtype:r,depth:i,onValue:a,offValue:l}=s;rt(o,"oneHot");const c=q(o.shape),u=new Float32Array(c*i);u.fill(l);const h=e.data.get(o.dataId).values;for(let d=0;d<c;++d)h[d]>=0&&h[d]<i&&(u[d*i+h[d]]=a);return e.makeTensorInfo([...o.shape,i],r,u)}const dO={kernelName:Fa,backendName:"cpu",kernelFunc:hO};function sc(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("zerosLike is not supported for string tensors");if(s.dtype==="complex64"){const o=oo({inputs:{input:s},backend:e}),r=sc({inputs:{x:o},backend:e}),i=Vo({inputs:{input:s},backend:e}),a=sc({inputs:{x:i},backend:e}),l=qe({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return Kd({backend:e,attrs:{shape:s.shape,value:0,dtype:s.dtype}})}const pO={kernelName:Ya,backendName:"cpu",kernelFunc:sc};function m1(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("onesLike is not supported for string tensors");if(s.dtype==="complex64"){const o=oo({inputs:{input:s},backend:e}),r=m1({inputs:{x:o},backend:e}),i=Vo({inputs:{input:s},backend:e}),a=sc({inputs:{x:i},backend:e}),l=qe({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return Kd({backend:e,attrs:{shape:s.shape,value:1,dtype:s.dtype}})}const fO={kernelName:Da,backendName:"cpu",kernelFunc:m1};function g1(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s;if(t.length===1)return nc({inputs:{input:t[0]},backend:e,attrs:{dim:o}});const r=t[0].shape,i=t[0].dtype;t.forEach(u=>{vc(r,u.shape,"All tensors passed to stack must have matching shapes"),S(i===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=[],l=t.map(u=>{const h=nc({inputs:{input:u},backend:e,attrs:{dim:o}});return a.push(h),h}),c=Wo({inputs:l,backend:e,attrs:{axis:o}});return a.forEach(u=>e.disposeIntermediateTensorInfo(u)),c}const mO={kernelName:Oa,backendName:"cpu",kernelFunc:g1};function gO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,constantValue:i}=s;rt(o,"pad");const a=r.map((b,w)=>b[0]+o.shape[w]+b[1]),l=r.map(b=>b[0]),c=e.data.get(o.dataId).values,u=q(o.shape),h=o.shape.length,d=lt(o.shape),p=q(a),f=a.length,m=lt(a),g=Ce(o.dtype,p);i!==0&&g.fill(i);for(let b=0;b<u;b++){const y=wo(b,h,d).map(($,N)=>$+l[N]),C=Sn(y,f,m);g[C]=c[b]}return{dataId:e.write(g,a,o.dtype),shape:a,dtype:o.dtype}}const x1={kernelName:_a,backendName:"cpu",kernelFunc:gO};const xO=ee((n,t)=>Math.pow(n,t)),bO=ue(Dr,xO),yO={kernelName:Dr,backendName:"cpu",kernelFunc:bO};function wO(n){const{inputs:t,backend:e,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:i}=t,{outputRaggedRank:a}=s,l=o.map(x=>e.data.get(x.dataId).values),c=o.map(x=>x.shape),u=e.data.get(r.dataId).values,h=e.data.get(i.dataId).values,[d,p,f]=F0(l,c,u,r.shape,r.dtype,h,i.shape),m=d.map(x=>e.makeTensorInfo([x.length],"int32",x)),g=e.makeTensorInfo(f,r.dtype,p);return m.concat([g])}const CO={kernelName:Ap,backendName:"cpu",kernelFunc:wO};function IO(n){const{inputs:t,backend:e}=n,{starts:s,limits:o,deltas:r}=t,i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=_0(i,s.shape,s.dtype,a,o.shape,l,r.shape),h=e.makeTensorInfo([c.length],"int32",c),d=e.makeTensorInfo([u.length],s.dtype,u);return[h,d]}const $O={kernelName:Dp,backendName:"cpu",kernelFunc:IO};function kO(n){const{inputs:t,backend:e,attrs:s}=n,{shape:o,values:r,defaultValue:i,rowPartitionTensors:a}=t,{rowPartitionTypes:l}=s,c=e.data.get(o.dataId).values,u=e.data.get(r.dataId).values,h=e.data.get(i.dataId).values,d=a.map(g=>e.data.get(g.dataId).values),p=a.map(g=>g.shape),[f,m]=P0(c,o.shape,u,r.shape,r.dtype,h,i.shape,d,p,l);return e.makeTensorInfo(f,r.dtype,m)}const vO={kernelName:Fp,backendName:"cpu",kernelFunc:kO};function SO(n){const{backend:t,attrs:e}=n,{start:s,stop:o,dtype:r,step:i}=e,a=B0(s,o,i,r);return t.makeTensorInfo([a.length],r,a)}const NO={kernelName:pu,backendName:"cpu",kernelFunc:SO};const TO=Ft(Fr,n=>1/n),EO={kernelName:Fr,backendName:"cpu",kernelFunc:TO};function RO(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s;rt(o,"resizeBilinear");const l=lt(o.shape),[c,u]=a,[h,d,p,f]=o.shape,m=e.data.get(o.dataId).values,g=new Float32Array(q([h,c,u,f])),x=[r&&c>1?d-1:d,r&&u>1?p-1:p],b=[r&&c>1?c-1:c,r&&u>1?u-1:u];let w=0;const y=x[0]/b[0],C=x[1]/b[1];for(let $=0;$<h;$++)for(let N=0;N<c;N++){let T;i?T=y*(N+.5)-.5:T=y*N;const k=Math.max(0,Math.floor(T)),v=T-k,I=Math.min(d-1,Math.ceil(T)),R=$*l[0]+k*l[1],A=$*l[0]+I*l[1];for(let P=0;P<u;P++){let _;i?_=C*(P+.5)-.5:_=C*P;const B=Math.max(0,Math.floor(_)),z=_-B,W=Math.min(p-1,Math.ceil(_)),H=R+B*l[2],K=A+B*l[2],j=R+W*l[2],X=A+W*l[2];for(let Y=0;Y<f;Y++){const tt=m[H+Y],Q=m[K+Y],ot=m[j+Y],at=m[X+Y],ct=tt+(ot-tt)*z,ht=Q+(at-Q)*z,gt=ct+(ht-ct)*v;g[w++]=gt}}}return e.makeTensorInfo([h,c,u,f],"float32",g)}const AO={kernelName:za,backendName:"cpu",kernelFunc:RO};function DO(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s;rt([r,o],"resizeBilinearGrad");const a=lt(o.shape),[l,c,u,h]=o.shape,[,d,p]=r.shape,f=new Float32Array(l*c*u*h),m=[i&&d>1?c-1:c,i&&p>1?u-1:u],g=[i&&d>1?d-1:d,i&&p>1?p-1:p],x=m[0]/g[0],b=m[1]/g[1],w=e.data.get(r.dataId).values;let y=0;for(let C=0;C<l;C++){const $=C*a[0];for(let N=0;N<d;N++){const T=N*x,k=Math.floor(T),v=Math.min(Math.ceil(T),c-1),I=$+k*a[1],R=$+v*a[1],A=T-k,P=1-A;for(let _=0;_<p;_++){const B=_*b,z=Math.floor(B),W=Math.min(Math.ceil(B),u-1),H=B-z,K=1-H,j=I+z*a[2],X=I+W*a[2],Y=R+z*a[2],tt=R+W*a[2],Q=P*K,ot=P*H,at=A*K,ct=A*H;for(let ht=0;ht<h;ht++){const gt=w[y++];f[j+ht]+=gt*Q,f[X+ht]+=gt*ot,f[Y+ht]+=gt*at,f[tt+ht]+=gt*ct}}}}return e.makeTensorInfo([l,u,c,h],"float32",f)}const FO={kernelName:gu,backendName:"cpu",kernelFunc:DO};function OO(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s;rt(o,"resizeNearestNeighbor");const l=lt(o.shape),[c,u]=a,[h,d,p,f]=o.shape,m=e.data.get(o.dataId).values,g=new Float32Array(h*c*u*f),x=[r&&c>1?d-1:d,r&&u>1?p-1:p],b=[r&&c>1?c-1:c,r&&u>1?u-1:u],w=x[0]/b[0],y=x[1]/b[1];let C=0;for(let $=0;$<h;$++){const N=$*l[0];for(let T=0;T<c;T++){const k=i?w*(T+.5):w*T;let v=Math.min(d-1,r?Math.round(k):Math.floor(k));i&&(v=Math.max(0,v));const I=N+v*l[1];for(let R=0;R<u;R++){const A=i?y*(R+.5):y*R;let P=Math.min(p-1,r?Math.round(A):Math.floor(A));i&&(P=Math.max(0,P));const _=I+P*l[2];for(let B=0;B<f;B++){const z=m[_+B];g[C++]=z}}}}return e.makeTensorInfo([h,c,u,f],o.dtype,g)}const _O={kernelName:Ba,backendName:"cpu",kernelFunc:OO};function LO(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s;rt([r,o],"resizeNearestNeighborGrad");const a=lt(o.shape),l=lt(r.shape),[c,u,h,d]=o.shape,[,p,f]=r.shape,m=new Float32Array(c*u*h*d),g=e.data.get(r.dataId).values,x=[i&&p>1?u-1:u,i&&f>1?h-1:h],b=[i&&p>1?p-1:p,i&&f>1?f-1:f],w=x[0]/b[0],y=x[1]/b[1],C=1/w,$=1/y,N=Math.ceil(C)*2+2,T=Math.ceil($)*2+2;for(let k=0;k<c;k++){const v=k*a[0];for(let I=0;I<u;I++){const R=v+I*a[1],A=Math.floor(I*C),P=Math.floor(A-N/2);for(let _=0;_<h;_++){const B=R+_*a[2],z=Math.floor(_*$),W=Math.floor(z-T/2);for(let H=0;H<d;H++){let K=0;for(let j=0;j<N;j++){const X=j+P;if(X<0||X>=p)continue;const Y=v+X*l[1],tt=X*w,Q=Math.min(u-1,i?Math.round(tt):Math.floor(tt));if(I===Q)for(let ot=0;ot<T;ot++){const at=ot+W;if(at<0||at>=f)continue;const ct=Y+at*l[2],ht=at*y,gt=Math.min(h-1,i?Math.round(ht):Math.floor(ht));_===gt&&(K+=g[ct+H])}}m[B+H]=K}}}}return e.makeTensorInfo(o.shape,o.dtype,m)}const MO={kernelName:mu,backendName:"cpu",kernelFunc:LO};function PO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dims:r}=s;rt(o,"reverse");const i=o.shape.length,a=yt(r,o.shape);if(i===0)return zn({inputs:{x:o},backend:e});const l=new me(o.shape,o.dtype),c=e.bufferSync(o);for(let u=0;u<l.size;u++){const h=l.indexToLoc(u),d=h.slice();a.forEach(p=>d[p]=o.shape[p]-1-d[p]),l.set(c.get(...d),...h)}return e.makeTensorInfo(l.shape,l.dtype,l.values)}const BO={kernelName:Va,backendName:"cpu",kernelFunc:PO};const zO={kernelName:$u,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,{radians:o,fillValue:r,center:i}=t,a=e,l=Ce(s.dtype,q(s.shape)),[c,u,h,d]=s.shape,[p,f]=Eh(i,u,h),m=255,g=Math.sin(o),x=Math.cos(o),b=a.data.get(s.dataId).values;for(let y=0;y<c;y++){const C=y*h*u*d;for(let $=0;$<u;$++){const N=$*(h*d);for(let T=0;T<h;T++){const k=T*d;for(let v=0;v<d;v++){const I=[c,$,T,v],R=I[2],A=I[1];let P=(R-p)*x-(A-f)*g,_=(R-p)*g+(A-f)*x;P=Math.round(P+p),_=Math.round(_+f);let B=r;if(typeof r!="number"&&(v===3?B=m:B=r[v]),P>=0&&P<h&&_>=0&&_<u){const W=_*(h*d),H=P*d,K=C+W+H+v;B=b[K]}const z=C+N+k+v;l[z]=B}}}}return{dataId:a.write(l,s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};const VO=Ft(Lr,n=>{const t=Math.floor(n);return n-t<.5?Math.floor(n):n-t>.5?Math.ceil(n):t%2===0?t:t+1}),WO={kernelName:Lr,backendName:"cpu",kernelFunc:VO};function UO(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o,updates:r}=t,{shape:i}=s,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Ks(r,o,i),d=!0,p=e.bufferSync(o),f=e.bufferSync(r),m=ro(p,f,i,h,c,l,a,u,0,d);return e.makeTensorInfo(i,m.dtype,m.values)}const GO={kernelName:Op,backendName:"cpu",kernelFunc:UO};function HO(n,t){let e=0,s=n.length,o=0;for(;e<s;)o=Math.floor((e+s)/2),n[o]<t?e=o+1:s=o;return s}function qO(n,t){let e=0,s=n.length,o=0;for(;e<s;)o=Math.floor((e+s)/2),n[o]<=t?e=o+1:s=o;return s}function KO(n,t,e,s,o,r){const i=Yt("int32",e*o);for(let a=0;a<e;++a){const l=n.slice(a*s,(a+1)*s),c=a*o;for(let u=0;u<o;++u)i[c+u]=r==="left"?HO(l,t[u+c]):qO(l,t[u+c])}return i}function jO(n){const{inputs:t,backend:e,attrs:s}=n,{sortedSequence:o,values:r}=t,{side:i}=s,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,c=KO(a,l,o.shape[0],o.shape[1],r.shape[1],i);return e.makeTensorInfo(r.shape,"int32",c)}const XO={kernelName:Lp,backendName:"cpu",kernelFunc:jO};function YO(n){const{inputs:t,backend:e}=n,{condition:s,t:o,e:r}=t;rt([s,o,r],"select");const i=s.shape.length,a=e.data.get(s.dataId).values,l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=Ue(o.dtype,r.dtype),h=Ie(q(o.shape),u);let d=0;const p=i===0||i>1||o.shape.length===1?1:q(o.shape.slice(1));for(let f=0;f<a.length;f++)for(let m=0;m<p;m++)a[f]===1?h[d++]=l[f]:h[d++]=c[f];return e.makeTensorInfo(o.shape,u,h)}const ZO={kernelName:Wa,backendName:"cpu",kernelFunc:YO};const JO=vl,QO=Sl,t_=Ft(Pr,n=>n>=0?QO*n:JO*(Math.exp(n)-1)),e_={kernelName:Pr,backendName:"cpu",kernelFunc:t_};const n_=Ft(Vr,n=>n<0?-1:n>0?1:0),s_={kernelName:Vr,backendName:"cpu",kernelFunc:n_};const o_=Ft(Br,n=>Math.sin(n)),r_={kernelName:Br,backendName:"cpu",kernelFunc:o_};const i_=Ft(zr,n=>Math.sinh(n)),a_={kernelName:zr,backendName:"cpu",kernelFunc:i_};const b1=Math.log(11920928955078125e-23)+2,l_=Ft(Ur,n=>{const t=n>-b1,e=n<b1,s=Math.exp(n);let o;return e?o=s:t?o=n:o=Math.log(1+s),o}),c_={kernelName:Ur,backendName:"cpu",kernelFunc:l_};function u_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,paddings:i}=s;rt([o],"spaceToBatchND");const a=q(r),l=[[0,0]];l.push(...i);for(let $=1+r.length;$<o.shape.length;++$)l.push([0,0]);const c=x1.kernelFunc({inputs:{x:o},backend:e,attrs:{paddings:l,constantValue:0}}),u=fi(c.shape,r,a,!1),h=mi(u.length,r.length,!1),d=gi(c.shape,r,a,!1),m=Bt({inputs:{x:c},backend:e,attrs:{shape:u}}),b=ze({inputs:{x:m},backend:e,attrs:{perm:h}}),C=Bt({inputs:{x:b},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(b),C}const h_={kernelName:Ha,backendName:"cpu",kernelFunc:u_};function d_(n){const{inputs:t,backend:e}=n,{indices:s,values:o,denseShape:r,defaultValue:i}=t;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
        ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
        ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
        ${o.shape}`);if(i.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${i.shape}`);const a=e.data.get(s.dataId).values,l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=e.data.get(i.dataId).values[0],[h,d,p,f,m]=U0(a,s.shape,s.dtype,l,o.dtype,c,u);return[e.makeTensorInfo(d,s.dtype,h),e.makeTensorInfo([d[0]],o.dtype,p),e.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(g=>Number(g)))),e.makeTensorInfo([m.length],s.dtype,new Int32Array(m))]}const p_={kernelName:Mp,backendName:"cpu",kernelFunc:d_};function f_(n){const{inputs:t,backend:e}=n,{inputIndices:s,inputShape:o,newShape:r}=t;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape
        ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape
        ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const i=Array.from(e.data.get(o.dataId).values),a=e.data.get(s.dataId).values,l=Array.from(e.data.get(r.dataId).values),[c,u,h]=G0(a,s.shape,s.dtype,i,l);return[e.makeTensorInfo(u,s.dtype,c),e.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}const m_={kernelName:Pp,backendName:"cpu",kernelFunc:f_};function g_(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
          ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
          ${r.shape}`);if(o.shape[0]!==r.shape[0])throw new Error("segmentIds and indices should have same size.");const i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=Vd(i,s.shape,s.dtype,a,l,!0);return e.makeTensorInfo(u,s.dtype,c)}const x_={kernelName:Bp,backendName:"cpu",kernelFunc:g_};function b_(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
         ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
         ${r.shape}`);if(o.shape[0]!==r.shape[0])throw new Error("segmentIds and indices should have same size.");const i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=Vd(i,s.shape,s.dtype,a,l);return e.makeTensorInfo(u,s.dtype,c)}const y_={kernelName:zp,backendName:"cpu",kernelFunc:b_};function w_(n){const{inputs:t,backend:e,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:i}=t,{outputShape:a}=s,{sliceRank:l,numUpdates:c,sliceSize:u,strides:h,outputSize:d}=Ks(r,o,a),p=!1,f=e.bufferSync(o);let m;switch(r.dtype){case"bool":{const g=e.bufferSync(r),x=!!e.data.get(i.dataId).values[0];m=ro(f,g,a,d,u,c,l,h,x,p);break}case"float32":{const g=e.bufferSync(r),x=e.data.get(i.dataId).values[0];m=ro(f,g,a,d,u,c,l,h,x,p);break}case"int32":{const g=e.bufferSync(r),x=e.data.get(i.dataId).values[0];m=ro(f,g,a,d,u,c,l,h,x,p);break}case"string":{const g=e.bufferSync(r),x=as(e.data.get(i.dataId).values[0]);m=ro(f,g,a,d,u,c,l,h,x,p);break}default:throw new Error(`Unsupported type ${r.dtype}`)}return e.makeTensorInfo(a,m.dtype,m.values)}const C_={kernelName:Vp,backendName:"cpu",kernelFunc:w_};function I_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{numOrSizeSplits:r,axis:i}=s,a=yt(i,o.shape)[0],l=Hh(o,r,a),c=new Array(o.shape.length).fill(0),u=o.shape.slice();return l.map(h=>{const d=[...u];d[a]=h;const p=io({inputs:{x:o},backend:e,attrs:{begin:c,size:d}});return c[a]+=h,p})}const $_={kernelName:qa,backendName:"cpu",kernelFunc:I_};const k_={kernelName:xu,backendName:"cpu",kernelFunc:({inputs:n,backend:t})=>{const{x:e}=n,s=t;rt(e,"square");const o=s.data.get(e.dataId).values,r=new Float32Array(o.length);for(let a=0;a<o.length;++a){const l=o[a];r[a]=l*l}return{dataId:s.write(r,e.shape,e.dtype),shape:e.shape,dtype:e.dtype}}};const v_=Ft(Yr,(n,t)=>{const e=t;return isNaN(n)?NaN:n>0?1:e.alpha}),S_={kernelName:Yr,backendName:"cpu",kernelFunc:v_};function N_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,end:i,strides:a,beginMask:l,endMask:c,ellipsisMask:u,newAxisMask:h,shrinkAxisMask:d}=s;rt(o,"stridedSlice");const{finalShapeSparse:p,finalShape:f,isIdentity:m,sliceDim0:g,isSimpleSlice:x,begin:b,end:w,strides:y}=Sh(o.shape,r,i,a,l,c,u,h,d);let C;if(m)C=Bt({inputs:{x:o},backend:e,attrs:{shape:f}});else if(g||x){S(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const $=$h(b,w,y),N=io({inputs:{x:o},backend:e,attrs:{begin:b,size:$}});C=Bt({inputs:{x:N},backend:e,attrs:{shape:f}}),e.disposeIntermediateTensorInfo(N)}else{const $=e.bufferSync(o),N=K0(p,$,y,b);C=e.makeTensorInfo(f,N.dtype,N.values)}return C}const T_={kernelName:yu,backendName:"cpu",kernelFunc:N_};function E_(n){const{inputs:t,backend:e,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:i,rightPad:a,padWidth:l,preserveShortSequences:c}=s,{data:u,dataSplits:h}=t,d=e.data.get(u.dataId).values,p=e.data.get(h.dataId).values,[f,m]=j0(d,p,o,r,i,a,l,c);return[e.makeTensorInfo([f.length],"string",f),e.makeTensorInfo(h.shape,"int32",m)]}const R_={kernelName:Wp,backendName:"cpu",kernelFunc:E_};function A_(n){const{inputs:t,backend:e,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:i}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(i.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${i.shape}`);const a=e.data.get(r.dataId).values,l=e.data.get(i.dataId).values[0],[c,u,h]=X0(a,l,o),d=u.length;return[e.makeTensorInfo([d,2],"int32",c),e.makeTensorInfo([d],"string",u),e.makeTensorInfo([2],"int32",new Int32Array(h))]}const D_={kernelName:Up,backendName:"cpu",kernelFunc:A_};function F_(n){const{inputs:t,backend:e,attrs:s}=n,{numBuckets:o}=s,{input:r}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const i=e.data.get(r.dataId).values,a=Y0(i,o);return e.makeTensorInfo(r.shape,"int32",a)}const O_={kernelName:Gp,backendName:"cpu",kernelFunc:F_};const __=Ft(Kr,n=>Math.tan(n)),L_={kernelName:Kr,backendName:"cpu",kernelFunc:__};const M_=Ft(jr,n=>Math.tanh(n)),P_={kernelName:jr,backendName:"cpu",kernelFunc:M_};function B_(n){const{inputs:t,backend:e}=n,{tensor:s,indices:o,updates:r}=t,{sliceRank:i,numUpdates:a,sliceSize:l,strides:c,outputSize:u}=Ks(r,o,s.shape),h=!1,d=e.bufferSync(o),p=e.bufferSync(r),f=e.bufferSync(s),m=ro(d,p,s.shape,u,l,a,i,c,f,h);return e.makeTensorInfo(s.shape,m.dtype,m.values)}const z_={kernelName:_p,backendName:"cpu",kernelFunc:B_};function V_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reps:r}=s;rt(o,"tile");const i=J0(e.bufferSync(o),r);return e.makeTensorInfo(i.shape,i.dtype,i.values)}const W_={kernelName:Xr,backendName:"cpu",kernelFunc:V_};function U_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{k:r,sorted:i}=s;rt(o,"topk");const a=e.data.get(o.dataId).values,[l,c]=t1(a,o.shape,o.dtype,r,i);return[e.makeTensorInfo(l.shape,l.dtype,l.values),e.makeTensorInfo(c.shape,c.dtype,c.values)]}const G_={kernelName:wu,backendName:"cpu",kernelFunc:U_};function H_(n){const{inputs:t,attrs:e,backend:s}=n,{image:o,transforms:r}=t,{interpolation:i,fillMode:a,fillValue:l,outputShape:c}=e,[u,h,d,p]=o.shape,[f,m]=c??[h,d],g=[u,f,m,p],x=lt(o.shape),b=x[0],w=x[1],y=x[2],C=lt(g),$=C[0],N=C[1],T=C[2],k=Ce(o.dtype,q(g));k.fill(l);const v=s.data.get(o.dataId).values,I=s.data.get(r.dataId).values;for(let A=0;A<u;++A){const P=r.shape[0]===1?I:I.subarray(A*8,A*8+8);for(let _=0;_<f;++_)for(let B=0;B<m;++B)for(let z=0;z<p;++z){let W;const H=P[6]*B+P[7]*_+1;if(H===0)continue;const K=(P[0]*B+P[1]*_+P[2])/H,j=(P[3]*B+P[4]*_+P[5])/H,X=y1(K,d,a),Y=y1(j,h,a);switch(i){case"nearest":W=Z_(v,h,d,b,w,y,A,Y,X,z,l);break;case"bilinear":W=J_(v,h,d,b,w,y,A,Y,X,z,l);break;default:throw new Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${i}`)}const tt=A*$+_*N+B*T+z;k[tt]=W}return s.makeTensorInfo(g,o.dtype,k)}return{dataId:s.write(k,g,o.dtype),shape:o.shape,dtype:o.dtype}}const q_={kernelName:Cu,backendName:"cpu",kernelFunc:H_};function y1(n,t,e){switch(e){case"reflect":return K_(n,t);case"wrap":return j_(n,t);case"nearest":return Y_(n,t);default:return X_(n)}}function K_(n,t){let e=n;if(e<0)if(t<=1)e=0;else{const s=2*t;e<s&&(e=s*Math.trunc(-e/s)+e),e=e<-t?e+s:-e-1}else if(e>t-1)if(t<=1)e=0;else{const s=2*t;e-=s*Math.trunc(e/s),e>=t&&(e=s-e-1)}return As(0,e,t-1)}function j_(n,t){let e=n;if(e<0)if(t<=1)e=0;else{const s=t-1;e+=t*(Math.trunc(-e/s)+1)}else if(e>t-1)if(t<=1)e=0;else{const s=t-1;e-=t*Math.trunc(e/s)}return As(0,e,t-1)}function X_(n,t){return n}function Y_(n,t){return As(0,n,t-1)}function Li(n,t,e,s,o,r,i,a,l,c,u){const h=i*s+a*o+l*r+c;return 0<=a&&a<t&&0<=l&&l<e?n[h]:u}function Z_(n,t,e,s,o,r,i,a,l,c,u){const h=Math.round(a),d=Math.round(l);return Li(n,t,e,s,o,r,i,h,d,c,u)}function J_(n,t,e,s,o,r,i,a,l,c,u){const h=Math.floor(a),d=Math.floor(l),p=h+1,f=d+1,m=(f-l)*Li(n,t,e,s,o,r,i,h,d,c,u)+(l-d)*Li(n,t,e,s,o,r,i,h,f,c,u),g=(f-l)*Li(n,t,e,s,o,r,i,p,d,c,u)+(l-d)*Li(n,t,e,s,o,r,i,p,f,c,u);return(p-a)*m+(a-h)*g}function Q_(n){const{inputs:t,attrs:e,backend:s}=n,{axis:o}=e,{x:r}=t;rt(r,"unique");const i=s.data.get(r.dataId).values,{outputValues:a,outputShape:l,indices:c}=e1(i,o,r.shape,r.dtype);return[s.makeTensorInfo(l,r.dtype,a),s.makeTensorInfo([c.length],"int32",c)]}const tL={kernelName:Iu,backendName:"cpu",kernelFunc:Q_};function eL(n){const{inputs:t,backend:e,attrs:s}=n,{value:o}=t;let{axis:r}=s;r<0&&(r+=o.shape.length);const i=o.shape.length,a=o.shape[r],l=new Array(i-1);let c=0;for(let p=0;p<i;p++)p!==r&&(l[c++]=o.shape[p]);const u=new Array(i).fill(0),h=o.shape.slice();h[r]=1;const d=new Array(a);for(let p=0;p<d.length;p++){u[r]=p;const f=io({inputs:{x:o},backend:e,attrs:{begin:u,size:h}});d[p]=Bt({inputs:{x:f},backend:e,attrs:{shape:l}}),e.disposeIntermediateTensorInfo(f)}return d}const nL={kernelName:ja,backendName:"cpu",kernelFunc:eL};function sL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,segmentIds:r}=t,{numSegments:i}=s;rt(o,"unsortedSegmentSum");const a=o.shape.length,l=r.shape.length,c=[],u=[],h=a-l;let d=r;for(let f=0;f<h;++f){const m=nc({inputs:{input:d},backend:e,attrs:{dim:f+1}});d=m,u.push(m)}for(let f=0;f<i;++f){const m=rs(f,"int32"),g=e.makeTensorInfo([],"int32",m),x=d0({inputs:{a:g,b:d},backend:e}),b=Is({inputs:{x},backend:e,attrs:{dtype:"float32"}}),w=Ql({inputs:{a:b,b:o},backend:e}),y=_i({inputs:{x:w},backend:e,attrs:{axis:0,keepDims:!1}});c.push(y),u.push(g),u.push(x),u.push(b),u.push(w),u.push(y)}const p=g1({inputs:c,backend:e,attrs:{axis:0}});return u.forEach(f=>e.disposeIntermediateTensorInfo(f)),p}const oL={kernelName:Xa,backendName:"cpu",kernelFunc:sL};const rL=[eA,_E,sA,rA,VE,aA,cA,hA,pA,mA,xA,yA,CA,kA,SA,EA,AA,FA,_A,QR,MA,BA,VA,UE,UA,BE,HE,HA,LE,qA,jA,XA,ZA,QA,eD,sD,rD,aD,cD,hD,pD,mD,xD,yD,wD,ID,kD,SD,ND,TD,ED,AD,OD,qR,LD,qE,GD,KE,HD,XE,ZD,JD,tF,ZE,QE,nF,oF,iF,lF,eR,sR,ME,uF,KA,dF,fF,gF,KR,rR,aR,bF,cR,wF,$F,vF,TF,RF,DF,FF,hR,_F,MF,BF,VF,UF,HF,KF,pR,XF,JF,eO,mR,xR,oO,aO,uO,yR,dO,fO,mO,x1,yO,XR,IR,CO,$O,vO,NO,PE,Hd,EO,YR,ZR,JR,AO,FO,_O,MO,BO,zO,WO,RR,GO,XO,ZO,e_,DR,s_,r_,a_,FR,QF,c_,h_,p_,m_,x_,y_,C_,$_,LR,k_,PR,zR,S_,T_,R_,D_,O_,GR,DD,L_,P_,z_,W_,G_,q_,wR,tL,nL,oL,pO];for(const n of rL)Xp(n);const ao={},oc={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function iL(n,t){ao[n]=t}function kn(n,t){if(!(n in ao)||t!=null){const s=lL(n,t);if(s!==null)ao[n]=s;else return console.log("Could not get context for WebGL version",n),null}const e=ao[n];return e==null||e.isContextLost()?(delete ao[n],kn(n)):(e.disable(e.DEPTH_TEST),e.disable(e.STENCIL_TEST),e.disable(e.BLEND),e.disable(e.DITHER),e.disable(e.POLYGON_OFFSET_FILL),e.disable(e.SAMPLE_COVERAGE),e.enable(e.SCISSOR_TEST),e.enable(e.CULL_FACE),e.cullFace(e.BACK),ao[n])}function aL(n){if(!U().getBool("IS_SAFARI")&&typeof OffscreenCanvas<"u"&&n===2)return new OffscreenCanvas(300,150);if(typeof document<"u")return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}function lL(n,t){if(n!==1&&n!==2)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const e=t??aL(n);return e.addEventListener("webglcontextlost",s=>{s.preventDefault(),delete ao[n]},!1),U().getBool("SOFTWARE_WEBGL_ENABLED")&&(oc.failIfMajorPerformanceCaveat=!1),n===1?e.getContext("webgl",oc)||e.getContext("experimental-webgl",oc):e.getContext("webgl2",oc)}var Mi;(function(n){n[n.DENSE=0]="DENSE",n[n.SHARED_BATCH=1]="SHARED_BATCH"})(Mi||(Mi={}));var Qe;(function(n){n[n.RENDER=0]="RENDER",n[n.UPLOAD=1]="UPLOAD",n[n.PIXELS=2]="PIXELS",n[n.DOWNLOAD=3]="DOWNLOAD"})(Qe||(Qe={}));var be;(function(n){n[n.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",n[n.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",n[n.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",n[n.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",n[n.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"})(be||(be={}));function Pi(n,t){return[t,n]}function cL(n,t){return n*t}function rc(n){const t=q(n),e=Math.ceil(t/4);return Nc(e)}function Uo(n,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(n/2))]}function uL(n,t){const[e,s]=Uo(n,t);return e*s*4}function jd(n,t){const e=n;let s,o,r,i,a,l,c,u,h,d;return U().getNumber("WEBGL_VERSION")===2?(s=e.R32F,o=e.R16F,r=e.RGBA16F,i=e.RGBA32F,a=e.RED,c=4,u=1,h=e.HALF_FLOAT,d=e.FLOAT,l=e.RGBA8):(s=n.RGBA,o=n.RGBA,r=n.RGBA,i=e.RGBA,a=n.RGBA,c=4,u=4,h=t!=null?t.HALF_FLOAT_OES:null,d=n.FLOAT,l=n.RGBA),{internalFormatFloat:s,internalFormatHalfFloat:o,internalFormatPackedHalfFloat:r,internalFormatPackedFloat:i,textureFormatFloat:a,downloadTextureFormat:l,downloadUnpackNumChannels:c,defaultNumChannels:u,textureTypeHalfFloat:h,textureTypeFloat:d}}function st(n,t){const e=t();return U().getBool("DEBUG")&&hL(n),e}function hL(n){const t=n.getError();if(t!==n.NO_ERROR)throw new Error("WebGL Error: "+mL(n,t))}const dL=596e-10,pL=65504;function fL(n){return!!(U().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||n===0||dL<Math.abs(n)&&Math.abs(n)<pL)}function mL(n,t){switch(t){case n.NO_ERROR:return"NO_ERROR";case n.INVALID_ENUM:return"INVALID_ENUM";case n.INVALID_VALUE:return"INVALID_VALUE";case n.INVALID_OPERATION:return"INVALID_OPERATION";case n.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case n.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case n.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${t}`}}function ic(n,t){return ts(n,()=>n.getExtension(t),'Extension "'+t+'" not supported on this browser.')}function gL(n,t){const e=ts(n,()=>n.createShader(n.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(st(n,()=>n.shaderSource(e,t)),st(n,()=>n.compileShader(e)),n.getShaderParameter(e,n.COMPILE_STATUS)===!1)throw console.log(n.getShaderInfoLog(e)),new Error("Failed to compile vertex shader.");return e}function xL(n,t){const e=ts(n,()=>n.createShader(n.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(st(n,()=>n.shaderSource(e,t)),st(n,()=>n.compileShader(e)),U().get("ENGINE_COMPILE_ONLY"))return e;if(n.getShaderParameter(e,n.COMPILE_STATUS)===!1)throw w1(t,n.getShaderInfoLog(e)),new Error("Failed to compile fragment shader.");return e}const bL=/ERROR: [0-9]+:([0-9]+):/g;function w1(n,t){const e=bL.exec(t);if(e==null){console.log(`Couldn't parse line number in error: ${t}`),console.log(n);return}const s=+e[1],o=n.split(`
`),r=o.length.toString().length+2,i=o.map((h,d)=>bo((d+1).toString(),r)+h);let a=0;for(let h=0;h<i.length;h++)a=Math.max(i[h].length,a);const l=i.slice(0,s-1),c=i.slice(s-1,s),u=i.slice(s);console.log(l.join(`
`)),console.log(t.split(`
`)[0]),console.log(`%c ${bo(c[0],a)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(u.join(`
`))}function yL(n){return ts(n,()=>n.createProgram(),"Unable to create WebGLProgram.")}function wL(n,t){if(st(n,()=>n.linkProgram(t)),!U().get("ENGINE_COMPILE_ONLY")&&n.getProgramParameter(t,n.LINK_STATUS)===!1)throw console.log(n.getProgramInfoLog(t)),new Error("Failed to link vertex and fragment shaders.")}function Xd(n,t){if(st(n,()=>n.validateProgram(t)),n.getProgramParameter(t,n.VALIDATE_STATUS)===!1)throw console.log(n.getProgramInfoLog(t)),new Error("Shader program validation failed.")}function CL(n,t){const e=ts(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return st(n,()=>n.bindBuffer(n.ARRAY_BUFFER,e)),st(n,()=>n.bufferData(n.ARRAY_BUFFER,t,n.STATIC_DRAW)),e}function IL(n,t){const e=ts(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return st(n,()=>n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,e)),st(n,()=>n.bufferData(n.ELEMENT_ARRAY_BUFFER,t,n.STATIC_DRAW)),e}function $L(n){return ts(n,()=>n.createTexture(),"Unable to create WebGLTexture.")}function kL(n,t){const e=U().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(n<=0||t<=0){const s=`[${n}x${t}]`;throw new Error("Requested texture size "+s+" is invalid.")}if(n>e||t>e){const s=`[${n}x${t}]`,o=`[${e}x${e}]`;throw new Error("Requested texture size "+s+" greater than WebGL maximum on this browser / GPU "+o+".")}}function vL(n){return ts(n,()=>n.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function C1(n,t,e,s,o,r,i){const a=n.getAttribLocation(t,e);return a===-1?!1:(st(n,()=>n.bindBuffer(n.ARRAY_BUFFER,s)),st(n,()=>n.vertexAttribPointer(a,o,n.FLOAT,!1,r,i)),st(n,()=>n.enableVertexAttribArray(a)),!0)}function SL(n,t,e){AL(n,e),st(n,()=>n.activeTexture(n.TEXTURE0+e)),st(n,()=>n.bindTexture(n.TEXTURE_2D,t))}function NL(n,t,e){return ts(n,()=>n.getUniformLocation(t,e),'uniform "'+e+'" not present in program.')}function TL(n,t,e){return n.getUniformLocation(t,e)}function EL(n,t,e,s){st(n,()=>SL(n,t,s)),st(n,()=>n.uniform1i(e,s))}function Yd(n,t,e){st(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,e)),st(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t,0))}function I1(n,t){st(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,t)),st(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,null,0))}function ac(n){const t=n.checkFramebufferStatus(n.FRAMEBUFFER);if(t!==n.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+RL(n,t))}function RL(n,t){switch(t){case n.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case n.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${t}`}}function ts(n,t,e){const s=st(n,()=>t());if(s==null)throw new Error(e);return s}function AL(n,t){const e=n.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,s=t+n.TEXTURE0;if(s<n.TEXTURE0||s>e){const o=`[gl.TEXTURE0, gl.TEXTURE${e}]`;throw new Error(`textureUnit must be in ${o}.`)}}function Go(n,t=2){return q(n.slice(0,n.length-t))}function Ho(n){if(n.length===0)throw Error("Cannot get rows and columns of an empty shape array.");return[n.length>1?n[n.length-2]:1,n[n.length-1]]}function lc(n){let t=[1,1,1];return n.length===0||n.length===1&&n[0]===1||(t=[Go(n),...Ho(n)]),t}function DL(n,t=!1){let e=U().getNumber("WEBGL_MAX_TEXTURE_SIZE"),s=U().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");s===1/0&&U().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(s=e/2),t&&(e=e*2,s=s*2,n=n.map((a,l)=>l>=n.length-2?kc(n[l]):n[l]),n.length===1&&(n=[2,n[0]])),n.length!==2&&(n=ss(n).newShape);let o=q(n),r=null;n.length<=1&&o<=e?r=[1,o]:n.length===2&&n[0]<=e&&n[1]<=e?r=n:n.length===3&&n[0]*n[1]<=e&&n[2]<=e?r=[n[0]*n[1],n[2]]:n.length===3&&n[0]<=e&&n[1]*n[2]<=e?r=[n[0],n[1]*n[2]]:n.length===4&&n[0]*n[1]*n[2]<=e&&n[3]<=e?r=[n[0]*n[1]*n[2],n[3]]:n.length===4&&n[0]<=e&&n[1]*n[2]*n[3]<=e&&(r=[n[0],n[1]*n[2]*n[3]]);const i=r!=null&&Math.max(...r)>s&&Math.min(...r)<=(t?2:1)&&Math.min(...r)>0;if(r==null||i)if(t){const a=Go(n);let l=2,c=2;n.length&&([l,c]=Ho(n)),o=a*(l/2)*(c/2),r=Nc(o).map(u=>u*2)}else r=Nc(o);return r}function cc(n){return n%2===0}function uc(n,t){if(n=n.slice(-2),t=t.slice(-2),Et(n,t)||!n.length||!t.length||n[0]===0||n[1]===0||t[0]===0||t[1]===0)return!0;if(n.length!==t.length){const e=n[n.length-1],s=t[t.length-1];if(e===s||cc(e)&&cc(s)&&(n[0]===1||t[0]===1))return!0}return n[1]===t[1]&&cc(n[0])&&cc(t[0])}let Zd,Jd;function FL(n){if(Zd==null){const t=kn(n);Zd=t.getParameter(t.MAX_TEXTURE_SIZE)}return Zd}function OL(n){if(Jd==null){const t=kn(n);Jd=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,Jd)}function _L(n){if(n===0)return 0;let t;const e=kn(n);return hn(e,"EXT_disjoint_timer_query_webgl2")&&n===2?t=2:hn(e,"EXT_disjoint_timer_query")?t=1:t=0,t}function hn(n,t){return n.getExtension(t)!=null}function $1(n){try{if(kn(n)!=null)return!0}catch(t){return console.log("Error when getting WebGL context: ",t),!1}return!1}function LL(n){if(n===0)return!1;const t=kn(n);if(n===1){if(!hn(t,"OES_texture_float"))return!1}else if(!hn(t,"EXT_color_buffer_float"))return!1;return Qd(t)}function ML(n){if(n===0)return!1;const t=kn(n);if(n===1){if(!hn(t,"OES_texture_float")||!hn(t,"WEBGL_color_buffer_float"))return!1}else{if(hn(t,"EXT_color_buffer_float"))return Qd(t);const s="EXT_color_buffer_half_float";if(hn(t,s)){const o=t.getExtension(s);return PL(t,o)}return!1}return Qd(t)}function Qd(n){const t=jd(n),e=n.createTexture();n.bindTexture(n.TEXTURE_2D,e),n.texImage2D(n.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);const r=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,r),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,e,0);const i=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(e),n.deleteFramebuffer(r),i}function PL(n,t){const e=jd(n,t),s=n.createTexture();n.bindTexture(n.TEXTURE_2D,s),n.texImage2D(n.TEXTURE_2D,0,e.internalFormatHalfFloat,1,1,0,e.textureFormatFloat,e.textureTypeHalfFloat,null);const i=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,i),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,s,0);const a=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(s),n.deleteFramebuffer(i),a}function BL(n){return n!==2?!1:kn(n).fenceSync!=null}function Bi(n,t){Array.isArray(n)||(n=[n]),n.forEach(e=>{e!=null&&S(e.dtype!=="complex64",()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}const it=U();it.registerFlag("HAS_WEBGL",()=>it.getNumber("WEBGL_VERSION")>0),it.registerFlag("WEBGL_VERSION",()=>$1(2)?2:$1(1)?1:0),it.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1),it.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>it.get("WEBGL_VERSION")===2),it.registerFlag("WEBGL_CPU_FORWARD",()=>!0),it.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1),it.registerFlag("WEBGL_PACK",()=>it.getBool("HAS_WEBGL")),it.registerFlag("WEBGL_PACK_NORMALIZATION",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_CLIP",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_REDUCE",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_LAZILY_UNPACK",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_CONV_IM2COL",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>it.getBool("WEBGL_PACK")),it.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>FL(it.getNumber("WEBGL_VERSION"))),it.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>OL(it.getNumber("WEBGL_VERSION"))),it.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const n=it.getNumber("WEBGL_VERSION");return n===0?0:_L(n)}),it.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>it.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!ff()),it.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>LL(it.getNumber("WEBGL_VERSION"))),it.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>it.getBool("WEBGL_FORCE_F16_TEXTURES")?!1:it.getBool("WEBGL_RENDER_FLOAT32_CAPABLE")),it.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>ML(it.getNumber("WEBGL_VERSION"))),it.registerFlag("WEBGL_FENCE_API_ENABLED",()=>BL(it.getNumber("WEBGL_VERSION"))),it.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>it.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0),it.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${n}.`)}),it.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>ff()?1:-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${n}.`)}),it.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128),it.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1),it.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5),it.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128),it.registerFlag("WEBGL_EXP_CONV",()=>!1),it.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>it.getBool("IS_TEST")),it.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0),it.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1),it.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1),it.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);function Ae(){let n,t,e,s,o,r,i,a,l,c;return U().getNumber("WEBGL_VERSION")===2?(n="#version 300 es",t="in",e="out",s="in",o="texture",r="outputColor",i="out vec4 outputColor;",a=U().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",l="",c=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(n="",t="attribute",e="varying",s="varying",o="texture2D",r="gl_FragColor",i="",a=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,l=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,c=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:n,attribute:t,varyingVs:e,varyingFs:s,texture2D:o,output:r,defineOutput:i,defineSpecialNaN:a,defineSpecialInf:l,defineRound:c}}function lo(n,t,e="index"){const s=lt(t);return s.map((o,r)=>{const i=`int ${n[r]} = ${e} / ${o}`,a=r===s.length-1?`int ${n[r+1]} = ${e} - ${n[r]} * ${o}`:`index -= ${n[r]} * ${o}`;return`${i}; ${a};`}).join("")}function hc(n,t,e="index"){const s=lt(t);return s.map((o,r)=>{const i=`int ${n[r]} = ${e} / outShapeStrides[${r}]`,a=r===s.length-1?`int ${n[r+1]} = ${e} - ${n[r]} * outShapeStrides[${r}]`:`index -= ${n[r]} * outShapeStrides[${r}]`;return`${i}; ${a};`}).join("")}function zL(n,t){const e=n.length,s=n.map(r=>`${t}[${r}]`),o=new Array(e-1);o[e-2]=s[e-1];for(let r=e-3;r>=0;--r)o[r]=`(${o[r+1]} * ${s[r+1]})`;return o}function VL(n,t,e="index"){const s=n.map((r,i)=>i),o=zL(s,t);return o.map((r,i)=>{const a=`int ${n[i]} = ${e} / ${o[i]}`,l=i===o.length-1?`int ${n[i+1]} = ${e} - ${n[i]} * ${o[i]}`:`index -= ${n[i]} * ${o[i]}`;return`${a}; ${l};`}).join("")}function tp(n){const t=lt(n).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function ep(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}const k1=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;const{getBroadcastDims:v1}=RS;function WL(n,t,e){const s=[];if(n.forEach(p=>{const f=q(p.shapeInfo.logicalShape);if(p.shapeInfo.isUniform?s.push(`uniform float ${p.name}${f>1?`[${f}]`:""};`):(s.push(`uniform sampler2D ${p.name};`),s.push(`uniform int offset${p.name};`)),e.enableShapeUniforms){const{uniformShape:m}=np(e.packedInputs,p.shapeInfo.logicalShape,p.shapeInfo.texShape);switch(m.length){case 1:s.push(`uniform int ${p.name}Shape;`);break;case 2:s.push(`uniform ivec2 ${p.name}Shape;`);break;case 3:s.push(`uniform ivec3 ${p.name}Shape;`);break;case 4:s.push(`uniform ivec4 ${p.name}Shape;`);break}s.push(`uniform ivec2 ${p.name}TexShape;`)}}),e.enableShapeUniforms){switch(t.logicalShape.length){case 1:s.push("uniform int outShape;");break;case 2:s.push("uniform ivec2 outShape;"),s.push("uniform int outShapeStrides;");break;case 3:s.push("uniform ivec3 outShape;"),s.push("uniform ivec2 outShapeStrides;");break;case 4:s.push("uniform ivec4 outShape;"),s.push("uniform ivec3 outShapeStrides;");break}s.push("uniform ivec2 outTexShape;")}e.customUniforms&&e.customUniforms.forEach(p=>{s.push(`uniform ${p.type} ${p.name}${p.arrayIndex?`[${p.arrayIndex}]`:""};`)});const o=s.join(`
`),r=n.map(p=>UL(p,t,e.packedInputs,e.enableShapeUniforms)).join(`
`),i=t.texShape,a=Ae(),l=qL(a);let c,u,h=XL(a);return t.isPacked?(c=GL(t.logicalShape,i,e.enableShapeUniforms),u=jL(a)):(c=HL(t.logicalShape,i,e.enableShapeUniforms),u=KL(a)),e.packedInputs&&(h+=QL),[h,l,u,o,c,r,e.userCode].join(`
`)}function qo(n,t=!1){const e=n.shapeInfo.logicalShape;switch(e.length){case 0:return hM(n,t);case 1:return pM(n,t);case 2:return mM(n,t);case 3:return xM(n,t);case 4:return yM(n,t);case 5:return wM(n);case 6:return CM(n);default:throw new Error(`${e.length}-D input sampling is not yet supported`)}}function S1(n,t){switch(n.shapeInfo.logicalShape.length){case 0:return uM(n);case 1:return dM(n,t);case 2:return fM(n,t);case 3:return gM(n,t);default:return bM(n,t)}}function UL(n,t,e=!1,s){let o="";e?o+=S1(n,s):o+=qo(n,s);const r=n.shapeInfo.logicalShape,i=t.logicalShape;return r.length<=i.length&&(e?o+=IM(n,t):o+=$M(n,t)),o}function GL(n,t,e){switch(n.length){case 0:return N1();case 1:return tM(n,t,e);case 2:return lM(n,t,e);case 3:return nM(n,t,e);default:return oM(n,t,e)}}function HL(n,t,e){switch(n.length){case 0:return N1();case 1:return eM(n,t,e);case 2:return cM(n,t,e);case 3:return sM(n,t,e);case 4:return rM(n,t,e);case 5:return iM(n,t);case 6:return aM(n,t);default:throw new Error(`${n.length}-D output sampling is not yet supported`)}}function qL(n){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `}function KL(n){return`
    void setOutput(float val) {
      ${n.output} = vec4(val, 0, 0, 0);
    }
  `}function jL(n){return`
    void setOutput(vec4 val) {
      ${n.output} = val;
    }
  `}function XL(n){return`${n.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${n.varyingFs} vec2 resultUV;
    ${n.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${n.defineSpecialNaN}
    ${n.defineSpecialInf}
    ${n.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${YL}
    ${ZL}
    ${JL}
  `}const YL=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,ZL=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,JL=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,QL=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function N1(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function tM(n,t,e){const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];return s[0]===1?e?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:s[1]===1?e?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:e?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `}function eM(n,t,e){return t[0]===1?e?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${t[1]}.0);
      }
    `:t[1]===1?e?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${t[0]}.0);
      }
    `:e?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      return resTexRC.x * ${t[1]} + resTexRC.y;
    }
  `}function nM(n,t,e){if(e)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],o=Math.ceil(n[2]/2),r=o*Math.ceil(n[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec3(b, r, c);
    }
  `}function sM(n,t,e){if(e)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${hc(["r","c","d"],n)}
    return ivec3(r, c, d);
  }
`;const s=lo(["r","c","d"],n);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${s}
      return ivec3(r, c, d);
    }
  `}function oM(n,t,e){if(e)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],o=Math.ceil(n[n.length-1]/2),r=o*Math.ceil(n[n.length-2]/2);let i=r,a="",l="b, r, c";for(let c=2;c<n.length-1;c++)i*=n[n.length-c-1],a=`
      int b${c} = index / ${i};
      index -= b${c} * ${i};
    `+a,l=`b${c}, `+l;return`
    ivec${n.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      ${a}

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec${n.length}(${l});
    }
  `}function rM(n,t,e){if(e)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${hc(["r","c","d","d2"],n)}
      return ivec4(r, c, d, d2);
    }
  `;const s=lo(["r","c","d","d2"],n);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${s}
      return ivec4(r, c, d, d2);
    }
  `}function iM(n,t){const e=lo(["r","c","d","d2","d3"],n);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${t[0]},
                             ${t[1]}));

      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${e}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function aM(n,t){const e=lo(["r","c","d","d2","d3","d4"],n);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${e}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function lM(n,t,e){const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];if(Et(n,t))return e?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${s[0]}, ${s[1]}));
      }
    `;const o=Math.ceil(n[1]/2);return e?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));

      int index = resTexRC.x * ${s[1]} + resTexRC.y;
      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec2(r, c);
    }
  `}function cM(n,t,e){return Et(n,t)?e?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${t[0]}, ${t[1]}));
      }
    `:n[1]===1?e?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:n[0]===1?e?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:e?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      int r = index / ${n[1]};
      int c = index - r * ${n[1]};
      return ivec2(r, c);
    }
  `}function co(n){return`offset${n}`}function uM(n){const t=n.name,e="get"+t.charAt(0).toUpperCase()+t.slice(1),s=Ae();return`
    vec4 ${e}() {
      return ${s.texture2D}(${t}, halfCR);
    }
  `}function hM(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1);if(n.shapeInfo.isUniform)return`float ${s}() {return ${e};}`;const[o,r]=n.shapeInfo.texShape;if(o===1&&r===1)return`
      float ${s}() {
        return sampleTexture(${e}, halfCR);
      }
    `;const i=co(e);if(t)return`
    float ${s}() {
      vec2 uv = uvFromFlat(${e}TexShape[0], ${e}TexShape[1], ${i});
      return sampleTexture(${e}, uv);
    }
  `;const[a,l]=n.shapeInfo.texShape;return`
    float ${s}() {
      vec2 uv = uvFromFlat(${a}, ${l}, ${i});
      return sampleTexture(${e}, uv);
    }
  `}function dM(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=n.shapeInfo.texShape,r=Ae();if(t)return`
    vec4 ${s}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${e}TexShape[0]) / 2.0), ceil(float(${e}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${r.texture2D}(${e}, uv);
    }
  `;const i=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];return`
    vec4 ${s}(int index) {
      vec2 uv = packedUVfrom1D(
        ${i[0]}, ${i[1]}, index);
      return ${r.texture2D}(${e}, uv);
    }
  `}function pM(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1);if(n.shapeInfo.isUniform)return`
      float ${s}(int index) {
        ${Ko(n)}
      }
    `;const o=n.shapeInfo.texShape,r=o[0],i=o[1];if(i===1&&r===1)return`
      float ${s}(int index) {
        return sampleTexture(${e}, halfCR);
      }
    `;const a=co(e);return i===1?t?`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / float(${e}TexShape[0]));
        return sampleTexture(${e}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / ${r}.0);
        return sampleTexture(${e}, uv);
      }
    `:r===1?t?`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / float(${e}TexShape[1]), 0.5);
        return sampleTexture(${e}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / ${i}.0, 0.5);
        return sampleTexture(${e}, uv);
      }
    `:t?`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${e}TexShape[0], ${e}TexShape[1], index + ${a});
      return sampleTexture(${e}, uv);
    }
  `:`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${r}, ${i}, index + ${a});
      return sampleTexture(${e}, uv);
    }
  `}function fM(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,i=r[0],a=r[1],l=Ae();if(r!=null&&Et(e,r))return t?`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);

        return ${l.texture2D}(${s}, uv);
      }
    `:`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${a}.0, ${i}.0);

        return ${l.texture2D}(${s}, uv);
      }
    `;if(t)return`
    vec4 ${o}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${l.texture2D}(${s}, uv);
    }
  `;const c=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)],u=Math.ceil(e[1]/2);return`
    vec4 ${o}(int row, int col) {
      vec2 uv = packedUVfrom2D(${u}, ${c[0]}, ${c[1]}, row, col);
      return ${l.texture2D}(${s}, uv);
    }
  `}function mM(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape;if(r!=null&&Et(e,r)){if(t)return`
      float ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `;const d=r[0],p=r[1];return`
    float ${o}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${p}.0, ${d}.0);
      return sampleTexture(${s}, uv);
    }
  `}const{newShape:i,keptDims:a}=ss(e),l=i;if(l.length<e.length){const d=jo(n,l),p=["row","col"];return`
      ${qo(d,t)}
      float ${o}(int row, int col) {
        return ${o}(${Xo(p,a)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${e[1]}, 1)));
        ${Ko(n)}
      }
    `;const c=r[0],u=r[1],h=co(s);return u===1?t?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${h}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${s}TexShape[0]));
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${h}), vec3(${e[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${c}.0);
      return sampleTexture(${s}, uv);
    }
  `:c===1?t?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${h}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${s}TexShape[1]), 0.5);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${h}), vec3(${e[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${s}, uv);
    }
  `:t?`
      float ${o}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${s}Shape[1] + col + ${h};
        vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
        return sampleTexture(${s}, uv);
      }
    `:`
  float ${o}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${e[1]} + col + ${h};
    vec2 uv = uvFromFlat(${c}, ${u}, index);
    return sampleTexture(${s}, uv);
  }
`}function gM(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,i=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)];if(e[0]===1){const d=e.slice(1),p=[1,2],f=jo(n,d),m=["b","row","col"];return`
        ${S1(f,t)}
        vec4 ${o}(int b, int row, int col) {
          return ${o}(${Xo(m,p)});
        }
      `}const a=Ae();if(t)return`
    vec4 ${o}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `;const l=i[0],c=i[1],u=Math.ceil(e[2]/2),h=u*Math.ceil(e[1]/2);return`
    vec4 ${o}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${l}, ${c}, ${h}, ${u}, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `}function xM(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=e[1]*e[2],i=e[2],{newShape:a,keptDims:l}=ss(e),c=a;if(c.length<e.length){const m=jo(n,c),g=["row","col","depth"];return`
        ${qo(m,t)}
        float ${o}(int row, int col, int depth) {
          return ${o}(${Xo(g,l)});
        }
      `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${r}, ${i}, 1)));
        ${Ko(n)}
      }
    `;const u=n.shapeInfo.texShape,h=u[0],d=u[1],p=n.shapeInfo.flatOffset;if(d===r&&p==null)return t?`
      float ${o}(int row, int col, int depth) {
        int stride1 = ${s}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
        float ${o}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${i}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${d}.0, ${h}.0);
          return sampleTexture(${s}, uv);
        }
      `;if(d===i&&p==null)return t?`
      float ${o}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${s}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${e[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${d}.0, ${h}.0);
      return sampleTexture(${s}, uv);
    }
  `;const f=co(s);return t?`
    float ${o}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${s}Shape[1] * ${s}Shape[2];
      int stride1 = ${s}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${f};
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
      return sampleTexture(${s}, uv);
    }
    `:`
      float ${o}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r} + col * ${i} + depth + ${f};
        vec2 uv = uvFromFlat(${h}, ${d}, index);
        return sampleTexture(${s}, uv);
      }
  `}function bM(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=Ae();if(t)return`
    vec4 ${s}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${e}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${e}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${e}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${e}TexShape[0]) / 2.0), ceil(float(${e}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${o.texture2D}(${e}, uv);
    }
  `;const r=n.shapeInfo.logicalShape,i=r.length,a=n.shapeInfo.texShape,l=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)],c=l[0],u=l[1],h=Math.ceil(r[i-1]/2);let d=h*Math.ceil(r[i-2]/2),p="int b, int row, int col",f=`b * ${d} + (row / 2) * ${h} + (col / 2)`;for(let m=2;m<i-1;m++)p=`int b${m}, `+p,d*=r[i-m-1],f=`b${m} * ${d} + `+f;return`
    vec4 ${s}(${p}) {
      int index = ${f};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${c});
      return ${o.texture2D}(${e}, uv);
    }
  `}function yM(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=e[3],i=e[2]*r,a=e[1]*i,{newShape:l,keptDims:c}=ss(e);if(l.length<e.length){const b=jo(n,l),w=["row","col","depth","depth2"];return`
      ${qo(b,t)}
      float ${o}(int row, int col, int depth, int depth2) {
        return ${o}(${Xo(w,c)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${a}, ${i}, ${r}, 1)));
        ${Ko(n)}
      }
    `;const u=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,d=h[0],p=h[1],f=`int stride2 = ${s}Shape[3];`,m=`int stride1 = ${s}Shape[2] * stride2;`,g=`int stride0 = ${s}Shape[1] * stride1;`;if(p===a&&u==null)return t?`
      float ${o}(int row, int col, int depth, int depth2) {
        ${f}
        ${m}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${i}, ${r}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${d}.0);
        return sampleTexture(${s}, uv);
      }
    `;if(p===r&&u==null)return t?`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${s}Shape[1] * ${s}Shape[2], ${s}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${e[1]*e[2]}, ${e[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${d}.0);
        return sampleTexture(${s}, uv);
      }
    `;const x=co(s);return t?`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${f}
      ${m}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index + ${x});
      return sampleTexture(${s}, uv);
    }
  `:`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} +
          depth * ${r} + depth2;
      vec2 uv = uvFromFlat(${d}, ${p}, index + ${x});
      return sampleTexture(${s}, uv);
    }
  `}function wM(n){const t=n.shapeInfo.logicalShape,e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=t[4],r=t[3]*o,i=t[2]*r,a=t[1]*i,{newShape:l,keptDims:c}=ss(t);if(l.length<t.length){const m=jo(n,l),g=["row","col","depth","depth2","depth3"];return`
      ${qo(m)}
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        return ${s}(${Xo(g,c)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${a}, ${i}, ${r}, ${o})) +
          depth3;
        ${Ko(n)}
      }
    `;const u=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,d=h[0],p=h[1];if(p===a&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${i}, ${r}, ${o}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${d}.0);
        return sampleTexture(${e}, uv);
      }
    `;if(p===o&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]},
               ${t[2]*t[3]}, ${t[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${d}.0);
        return sampleTexture(${e}, uv);
      }
    `;const f=co(e);return`
    float ${s}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} + depth * ${r} +
          depth2 * ${o} + depth3 + ${f};
      vec2 uv = uvFromFlat(${d}, ${p}, index);
      return sampleTexture(${e}, uv);
    }
  `}function CM(n){const t=n.shapeInfo.logicalShape,e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),{newShape:o,keptDims:r}=ss(t);if(o.length<t.length){const g=jo(n,o),x=["row","col","depth","depth2","depth3","depth4"];return`
      ${qo(g)}
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${s}(${Xo(x,r)});
      }
    `}const i=t[5],a=t[4]*i,l=t[3]*a,c=t[2]*l,u=t[1]*c;if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${c}, ${l}, ${a})) +
          dot(
            vec2(depth3, depth4),
            vec2(${i}, 1)));
        ${Ko(n)}
      }
    `;const h=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,p=d[0],f=d[1];if(f===u&&h==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${c}, ${l}, ${a}, ${i})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${p}.0);
        return sampleTexture(${e}, uv);
      }
    `;if(f===i&&h==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]*t[4]},
               ${t[2]*t[3]*t[4]},
               ${t[3]*t[4]},
               ${t[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${p}.0);
        return sampleTexture(${e}, uv);
      }
    `;const m=co(e);return`
    float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${c} + depth * ${l} +
          depth2 * ${a} + depth3 * ${i} + depth4 + ${m};
      vec2 uv = uvFromFlat(${p}, ${f}, index);
      return sampleTexture(${e}, uv);
    }
  `}function Ko(n){const t=n.name,e=q(n.shapeInfo.logicalShape);return e<2?`return ${t};`:`
    for (int i = 0; i < ${e}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function IM(n,t){const e=n.name,s=e.charAt(0).toUpperCase()+e.slice(1),o="get"+s+"AtOutCoords",r=n.shapeInfo.logicalShape.length,i=t.logicalShape.length,a=v1(n.shapeInfo.logicalShape,t.logicalShape),l=Ot(i),c=i-r;let u;const h=["x","y","z","w","u","v"];r===0?u="":i<2&&a.length>=1?u="coords = 0;":u=a.map(b=>`coords.${h[b+c]} = 0;`).join(`
`);let d="";i<2&&r>0?d="coords":d=n.shapeInfo.logicalShape.map((b,w)=>`coords.${h[w+c]}`).join(", ");let p="return outputValue;";const m=q(n.shapeInfo.logicalShape)===1,x=q(t.logicalShape)===1;if(r===1&&!m&&!x)p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(m&&!x)i===1?p=`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:p=`
        return vec4(outputValue.x);
      `;else if(a.length){const b=r-2,w=r-1;a.indexOf(b)>-1&&a.indexOf(w)>-1?p="return vec4(outputValue.x);":a.indexOf(b)>-1?p="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":a.indexOf(w)>-1&&(p="return vec4(outputValue.xx, outputValue.zz);")}return`
    vec4 ${o}() {
      ${l} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${s}(${d});
      ${p}
    }
  `}function $M(n,t){const e=n.name,s=e.charAt(0).toUpperCase()+e.slice(1),o="get"+s+"AtOutCoords",r=t.texShape,i=n.shapeInfo.texShape,a=n.shapeInfo.logicalShape.length,l=t.logicalShape.length;if(!n.shapeInfo.isUniform&&a===l&&n.shapeInfo.flatOffset==null&&Et(i,r))return`
      float ${o}() {
        return sampleTexture(${e}, resultUV);
      }
    `;const c=Ot(l),u=v1(n.shapeInfo.logicalShape,t.logicalShape),h=l-a;let d;const p=["x","y","z","w","u","v"];a===0?d="":l<2&&u.length>=1?d="coords = 0;":d=u.map(m=>`coords.${p[m+h]} = 0;`).join(`
`);let f="";return l<2&&a>0?f="coords":f=n.shapeInfo.logicalShape.map((m,g)=>`coords.${p[g+h]}`).join(", "),`
    float ${o}() {
      ${c} coords = getOutputCoords();
      ${d}
      return get${s}(${f});
    }
  `}function Ot(n){if(n<=1)return"int";if(n===2)return"ivec2";if(n===3)return"ivec3";if(n===4)return"ivec4";if(n===5)return"ivec5";if(n===6)return"ivec6";throw Error(`GPU for rank ${n} is not yet supported`)}function np(n,t,e){const{newShape:s,keptDims:o}=ss(t),r=t.length,i=n&&r===3&&t[0]===1,a=i?t.slice(1):s,l=!n&&r>1&&!Et(t,e)&&s.length<r||i;return{useSqueezeShape:l,uniformShape:l?a:t,keptDims:o}}function jo(n,t){const e=JSON.parse(JSON.stringify(n));return e.shapeInfo.logicalShape=t,e}function Xo(n,t){return t.map(e=>n[e]).join(", ")}function kM(n,t,e,s){const o=e.map((u,h)=>{const d={logicalShape:u.shape,texShape:u.isUniform?null:u.texData.texShape,isUniform:u.isUniform,isPacked:u.isUniform?!1:u.texData.isPacked,flatOffset:null};return u.texData!=null&&u.texData.slice!=null&&u.texData.slice.flatOffset>0&&(d.flatOffset=u.texData.slice.flatOffset),{name:t.variableNames[h],shapeInfo:d}}),r=o.map(u=>u.shapeInfo),i={logicalShape:s.shape,texShape:s.texData.texShape,isUniform:!1,isPacked:s.texData.isPacked,flatOffset:null},a=WL(o,i,t),l=xL(n.gl,a),c=n.createProgram(l);return U().get("ENGINE_COMPILE_ONLY")?{program:t,fragmentShader:l,source:a,webGLProgram:c,inShapeInfos:r,outShapeInfo:i,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(n.buildVao(c),Object.assign({program:t,fragmentShader:l,source:a,webGLProgram:c,inShapeInfos:r,outShapeInfo:i},T1(n,t,c)))}function T1(n,t,e){const s=[],o=[];let r,i,a,l=null,c=null;c=n.getUniformLocation(e,"NAN",!1),U().getNumber("WEBGL_VERSION")===1&&(l=n.getUniformLocation(e,"INFINITY",!1));const u=!1;for(const h of t.variableNames){const d={name:h,uniform:n.getUniformLocation(e,h,u),offset:n.getUniformLocation(e,`offset${h}`,u)};t.enableShapeUniforms&&(d.shape=n.getUniformLocation(e,`${h}Shape`,u),d.texShape=n.getUniformLocation(e,`${h}TexShape`,u)),s.push(d)}if(t.enableShapeUniforms&&(r=n.getUniformLocation(e,"outShape",u),a=n.getUniformLocation(e,"outShapeStrides",u),i=n.getUniformLocation(e,"outTexShape",u)),t.customUniforms)for(const h of t.customUniforms)o.push(n.getUniformLocation(e,h.name,u));return{variablesLocations:s,customUniformLocations:o,infLoc:l,nanLoc:c,outShapeLocation:r,outShapeStridesLocation:a,outTexShapeLocation:i}}function E1(n,t){if(n.length!==t.length)throw Error(`Binary was compiled with ${n.length} inputs, but was executed with ${t.length} inputs`);n.forEach((e,s)=>{const o=e.logicalShape,r=t[s],i=r.shape;if(!Et(o,i))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${o} and ${i} must match`);if(e.isUniform&&r.isUniform)return;const a=e.texShape,l=r.isUniform?null:r.texData.texShape;if(!Et(a,l))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${a} and ${l} must match`)})}function vM(n,t,e,s,o){t.program.enableShapeUniforms||(E1(t.inShapeInfos,e),E1([t.outShapeInfo],[s]));const r=s.texData.texture,i=s.texData.texShape;s.texData.isPacked?n.setOutputPackedMatrixTexture(r.texture,i[0],i[1]):n.setOutputMatrixTexture(r.texture,i[0],i[1]),n.setProgram(t.webGLProgram),n.bindVertexArray(t.webGLProgram.vao),U().getNumber("WEBGL_VERSION")===1&&t.infLoc!==null&&n.gl.uniform1f(t.infLoc,1/0),t.nanLoc!==null&&n.gl.uniform1f(t.nanLoc,NaN);for(let l=0;l<e.length;++l){const c=e[l],{uniform:u,offset:h,shape:d,texShape:p}=t.variablesLocations[l];if(d){const{uniformShape:f}=np(t.program.packedInputs,c.shape,c.texData.texShape);switch(f.length){case 1:n.gl.uniform1iv(d,new Int32Array(f));break;case 2:n.gl.uniform2iv(d,new Int32Array(f));break;case 3:n.gl.uniform3iv(d,new Int32Array(f));break;case 4:n.gl.uniform4iv(d,new Int32Array(f));break}}if(p&&n.gl.uniform2i(p,c.texData.texShape[0],c.texData.texShape[1]),u!=null){if(c.isUniform){if(q(c.shape)<2)n.gl.uniform1f(u,c.uniformValues[0]);else{let f=c.uniformValues;f instanceof Float32Array||(f=new Float32Array(f)),n.gl.uniform1fv(u,f)}continue}c.texData.slice!=null&&h!=null&&n.gl.uniform1i(h,c.texData.slice.flatOffset),n.setInputMatrixTexture(c.texData.texture.texture,u,l)}}const a=t.outShapeLocation;if(a)switch(s.shape.length){case 1:n.gl.uniform1iv(a,new Int32Array(s.shape));break;case 2:n.gl.uniform2iv(a,new Int32Array(s.shape));break;case 3:n.gl.uniform3iv(a,new Int32Array(s.shape));break;case 4:n.gl.uniform4iv(a,new Int32Array(s.shape));break}if(t.outShapeStridesLocation){const l=lt(s.shape);switch(s.shape.length){case 2:n.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(l));break;case 3:n.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(l));break;case 4:n.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(l));break}}if(t.outTexShapeLocation&&n.gl.uniform2i(t.outTexShapeLocation,s.texData.texShape[0],s.texData.texShape[1]),t.program.customUniforms&&o)for(let l=0;l<t.program.customUniforms.length;++l){const c=t.program.customUniforms[l],u=t.customUniformLocations[l],h=o[l];if(c.type==="float")n.gl.uniform1fv(u,h);else if(c.type==="vec2")n.gl.uniform2fv(u,h);else if(c.type==="vec3")n.gl.uniform3fv(u,h);else if(c.type==="vec4")n.gl.uniform4fv(u,h);else if(c.type==="int")n.gl.uniform1iv(u,h);else if(c.type==="ivec2")n.gl.uniform2iv(u,h);else if(c.type==="ivec3")n.gl.uniform3iv(u,h);else if(c.type==="ivec4")n.gl.uniform4iv(u,h);else throw Error(`uniform type ${c.type} is not supported yet.`)}n.executeProgram()}function SM(n,t,e){let s="";t.concat(e).forEach(i=>{const a=i.texData!=null&&i.texData.slice!=null&&i.texData.slice.flatOffset>0;if(n.enableShapeUniforms&&!i.isUniform){const l=i.texData.texShape,{useSqueezeShape:c,uniformShape:u,keptDims:h}=np(n.packedInputs,i.shape,l);let d="",p="",f="";if(u.length===1&&n.packedInputs){const C=[Math.ceil(l[0]/2),Math.ceil(l[1]/2)];d=`${C[0]>1}_${C[1]>1}`}else if(u.length===2&&!n.packedInputs)p=`${u[0]>1}_${u[1]>1}`;else if(u.length>2&&!n.packedInputs){const C=lt(u);f=`${C[0]===l[1]}_${C[C.length-1]===l[1]}`}const m=i.shape.length,g=u.length===2&&Et(i.shape,l),x=q(i.shape)===1,b=Ro(i.shape,e.shape),w=!n.packedInputs&&m===e.shape.length&&Et(l,e.texData.texShape),y=n.packedInputs||u.length>2?"":`${l[0]>1}_${l[1]>1}`;s+=`${m}_${w}_${c?h:""}_${u.length}_${x}_${b}_${g}_${d}_${p}_${f}_${y}_${a}`}else{const l=i.isUniform?"uniform":i.texData.texShape;s+=`${i.shape}_${l}_${a}`}});const o=n.userCode;let r=n.constructor.name;return r+="_"+s+"_"+o+`${U().getNumber("WEBGL_VERSION")}`,r}function Se(n){return U().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&n<=4}class NM{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=Mi.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const e=Ae();this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?hc(["r","c","d"],t):lo(["r","c","d"],t)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${e.output} = result;
      }
    `}}class TM{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=Mi.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const e=Ae();this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?hc(["r","c","d"],t):lo(["r","c","d"],t)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${e.output} = result;
      }
    `}}class EM{constructor(t){this.variableNames=["A"],this.outTexUsage=Qe.DOWNLOAD;const e=Ae();this.outputShape=t,this.userCode=`
      ${k1}

      void main() {
        float x = getAAtOutCoords();
        ${e.output} = encode_float(x);
      }
    `}}class RM{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=Qe.DOWNLOAD;const e=Ae();this.outputShape=t,this.userCode=`
      ${k1}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${e.output} = encode_float(x);
      }
    `}}const AM={R:0,G:1,B:2,A:3};class R1{constructor(t,e=!1,s="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const o=Ae();this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length);let r="result";e&&(r="floor(result * 255. + 0.5)");let i="";for(let a=0;a<s.length;a++){const l=s[a];i+=`
          if(offset == ${a}) {
            result = values[${AM[l]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?ep():tp(t)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${s.length});

        flatIndex = idiv(flatIndex, ${s.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${o.texture2D}(A, uv);
          ${i}
        }
        ${o.output} = vec4(${r}, 0., 0., 0.);
      }
    `}}class DM{constructor(t,e=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const s=Ae();this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length);let o="",r="result";e&&(r="floor(result * 255. + 0.5)");for(let i=0;i<=1;i++)for(let a=0;a<=1;a++){const l=i*2+a;o+=`
          localCoords = coords;
          if(localCoords[2] + ${a} < ${this.enableShapeUniforms?"outShape[2]":`${t[2]}`}) {
          localCoords[2] += ${a};
          if (localCoords[1] + ${i} < ${this.enableShapeUniforms?"outShape[1]":`${t[1]}`}) {
            localCoords[1] += ${i};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${s.texture2D}(A, uv);

            if (offset == 0) {
              result[${l}] = values[0];
            } else if (offset == 1) {
              result[${l}] = values[1];
            } else if (offset == 2) {
              result[${l}] = values[2];
            } else {
              result[${l}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?ep():tp(t)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${o}

          ${s.output} = ${r};
        }
    `}}function FM(n){const t=Ae(),e=`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;return gL(n,e)}function OM(n){const t=new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]);return CL(n,t)}function _M(n){const t=new Uint16Array([0,1,2,2,1,3]);return IL(n,t)}function zi(n,t,e,s,o,r){kL(t,e);const i=$L(n),a=n.TEXTURE_2D;return st(n,()=>n.bindTexture(a,i)),st(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE)),st(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)),st(n,()=>n.texParameteri(a,n.TEXTURE_MIN_FILTER,n.NEAREST)),st(n,()=>n.texParameteri(a,n.TEXTURE_MAG_FILTER,n.NEAREST)),U().getNumber("WEBGL_VERSION")===1?st(n,()=>n.texImage2D(a,0,s,t,e,0,o,r,null)):st(n,()=>n.texStorage2D(a,1,s,t,e)),st(n,()=>n.bindTexture(n.TEXTURE_2D,null)),{texture:i,texShape:[e,t]}}function A1(n){return n.internalFormatFloat}function LM(n,t,e,s){const[o,r]=Pi(t,e);return zi(n,o,r,A1(s),s.textureFormatFloat,n.FLOAT)}function D1(n){return n.internalFormatHalfFloat}function MM(n,t,e,s){const[o,r]=Pi(t,e);return zi(n,o,r,D1(s),s.textureFormatFloat,s.textureTypeHalfFloat)}function F1(n){return n.downloadTextureFormat}function PM(n,t,e,s){const[o,r]=Pi(t,e);return zi(n,o,r,F1(s),n.RGBA,n.UNSIGNED_BYTE)}function O1(n){return n.internalFormatPackedFloat}function BM(n,t,e,s){const[o,r]=Uo(t,e);return zi(n,o,r,O1(s),n.RGBA,n.FLOAT)}function _1(n){return n.internalFormatPackedHalfFloat}function zM(n,t,e,s){const[o,r]=Uo(t,e);return zi(n,o,r,_1(s),n.RGBA,s.textureTypeHalfFloat)}function VM(n,t,e){return st(n,()=>n.bindBuffer(n.ARRAY_BUFFER,e)),C1(n,t,"clipSpacePos",e,3,20,0)&&C1(n,t,"uv",e,2,20,12)}function WM(n,t,e,s,o,r){st(n,()=>n.bindTexture(n.TEXTURE_2D,t));let i,a,l;o instanceof Uint8Array?(i=new Uint8Array(e*s*4),a=n.UNSIGNED_BYTE,l=n.RGBA):(i=new Float32Array(e*s*4),a=n.FLOAT,l=r.internalFormatPackedFloat),i.set(o),U().getNumber("WEBGL_VERSION")===2?st(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,e,s,n.RGBA,a,i)):st(n,()=>n.texImage2D(n.TEXTURE_2D,0,l,e,s,0,n.RGBA,a,i)),st(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function UM(n,t,e){st(n,()=>n.bindTexture(n.TEXTURE_2D,t)),e.data instanceof Uint8Array?U().getNumber("WEBGL_VERSION")===2?st(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,e.width,e.height,n.RGBA,n.UNSIGNED_BYTE,e.data)):st(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,e.width,e.height,0,n.RGBA,n.UNSIGNED_BYTE,e.data)):U().getNumber("WEBGL_VERSION")===2?st(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,n.RGBA,n.UNSIGNED_BYTE,e)):st(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,n.RGBA,n.UNSIGNED_BYTE,e)),st(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function GM(n,t,e,s){const o=n.createBuffer();st(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,o));const a=4*4*t*e;return st(n,()=>n.bufferData(n.PIXEL_PACK_BUFFER,a,n.STREAM_READ)),st(n,()=>n.readPixels(0,0,e,t,n.RGBA,n.FLOAT,0)),st(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,null)),o}function HM(n,t,e){const s=n,o=new Float32Array(e);return s.bindBuffer(s.PIXEL_PACK_BUFFER,t),s.getBufferSubData(s.PIXEL_PACK_BUFFER,0,o),s.bindBuffer(s.PIXEL_PACK_BUFFER,null),o}function qM(n,t,e,s){const[o,r]=Pi(t,e),i=4,a=new Uint8Array(cL(t*e,i));return st(n,()=>n.readPixels(0,0,o,r,s.downloadTextureFormat,n.UNSIGNED_BYTE,a)),new Float32Array(a.buffer)}function KM(n,t,e,s,o,r,i,a){const l=n,c=new Float32Array(uL(r,i));return l.bindBuffer(l.PIXEL_PACK_BUFFER,t),l.getBufferSubData(l.PIXEL_PACK_BUFFER,0,c),l.bindBuffer(l.PIXEL_PACK_BUFFER,null),c}function jM(n,t,e){const s=new Float32Array(t*e*4);return st(n,()=>n.readPixels(0,0,e,t,n.RGBA,n.FLOAT,s)),s}class sp{constructor(t){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const e=U().getNumber("WEBGL_VERSION");if(t!=null?(this.gl=t,iL(e,t)):this.gl=kn(e),t=this.gl,U().getNumber("WEBGL_VERSION")===2){const r=t;this.createVertexArray=()=>st(r,()=>r.createVertexArray()),this.bindVertexArray=i=>st(r,()=>r.bindVertexArray(i)),this.deleteVertexArray=i=>st(r,()=>r.deleteVertexArray(i)),this.getVertexArray=()=>st(r,()=>r.getParameter(r.VERTEX_ARRAY_BINDING))}else if(t!=null){const r=t.getExtension("OES_vertex_array_object");if(r==null)throw new Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>st(t,()=>r.createVertexArrayOES()),this.bindVertexArray=i=>st(t,()=>r.bindVertexArrayOES(i)),this.deleteVertexArray=i=>st(t,()=>r.deleteVertexArrayOES(i)),this.getVertexArray=()=>st(t,()=>t.getParameter(r.VERTEX_ARRAY_BINDING_OES))}let s="WEBGL_color_buffer_float";const o="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),U().getNumber("WEBGL_VERSION")===1){const r="OES_texture_float",i="OES_texture_half_float";if(this.textureFloatExtension=ic(this.gl,r),hn(this.gl,i))this.textureHalfFloatExtension=ic(this.gl,i);else if(U().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(s),hn(this.gl,o))this.colorBufferHalfFloatExtension=ic(this.gl,o);else if(U().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(s="EXT_color_buffer_float",hn(this.gl,s))this.colorBufferFloatExtension=this.gl.getExtension(s);else if(hn(this.gl,o))this.colorBufferHalfFloatExtension=this.gl.getExtension(o);else throw new Error("GL context does not support color renderable floats");this.vertexBuffer=OM(this.gl),this.indexBuffer=_M(this.gl),this.framebuffer=vL(this.gl),this.textureConfig=jd(this.gl,this.textureHalfFloatExtension)}get debug(){return U().getBool("DEBUG")}dispose(){if(this.disposed)return;this.program!=null&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),this.outputTexture!=null&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const t=this.gl;st(t,()=>t.finish()),st(t,()=>t.bindFramebuffer(t.FRAMEBUFFER,null)),st(t,()=>t.deleteFramebuffer(this.framebuffer)),st(t,()=>t.bindBuffer(t.ARRAY_BUFFER,null)),st(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,null)),st(t,()=>t.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(t,e){return this.throwIfDisposed(),LM(this.gl,t,e,this.textureConfig)}createFloat16MatrixTexture(t,e){return this.throwIfDisposed(),MM(this.gl,t,e,this.textureConfig)}createUnsignedBytesMatrixTexture(t,e){return this.throwIfDisposed(),PM(this.gl,t,e,this.textureConfig)}uploadPixelDataToTexture(t,e){this.throwIfDisposed(),UM(this.gl,t,e)}uploadDenseMatrixToTexture(t,e,s,o){this.throwIfDisposed(),WM(this.gl,t,e,s,o,this.textureConfig)}createFloat16PackedMatrixTexture(t,e){return this.throwIfDisposed(),zM(this.gl,t,e,this.textureConfig)}createPackedMatrixTexture(t,e){return this.throwIfDisposed(),BM(this.gl,t,e,this.textureConfig)}deleteMatrixTexture(t){this.throwIfDisposed(),this.outputTexture===t&&(I1(this.gl,this.framebuffer),this.outputTexture=null),st(this.gl,()=>this.gl.deleteTexture(t))}downloadByteEncodedFloatMatrixFromOutputTexture(t,e,s){return this.downloadMatrixDriver(t,()=>qM(this.gl,e,s,this.textureConfig))}downloadPackedMatrixFromBuffer(t,e,s,o,r,i){return KM(this.gl,t,e,s,o,r,i,this.textureConfig)}downloadFloat32MatrixFromBuffer(t,e){return HM(this.gl,t,e)}createBufferFromTexture(t,e,s){this.bindTextureToFrameBuffer(t);const o=GM(this.gl,e,s,this.textureConfig);return this.unbindTextureToFrameBuffer(),o}createAndWaitForFence(){const t=this.createFence(this.gl);return this.pollFence(t)}createFence(t){let e,s;if(U().getBool("WEBGL_FENCE_API_ENABLED")){const o=t,r=o.fenceSync(o.SYNC_GPU_COMMANDS_COMPLETE,0);t.flush(),s=()=>{const i=o.clientWaitSync(r,0,0);return i===o.ALREADY_SIGNALED||i===o.CONDITION_SATISFIED},e=r}else U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(e=this.beginQuery(),this.endQuery(),s=()=>this.isQueryAvailable(e,U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):s=()=>!0;return{query:e,isFencePassed:s}}downloadMatrixFromPackedTexture(t,e,s){return this.downloadMatrixDriver(t,()=>jM(this.gl,e,s))}createProgram(t){this.throwIfDisposed();const e=this.gl;this.vertexShader==null&&(this.vertexShader=FM(e));const s=yL(e);st(e,()=>e.attachShader(s,this.vertexShader)),st(e,()=>e.attachShader(s,t)),wL(e,s);const o=Object.assign(s,{vao:this.createVertexArray()});return this.debug&&Xd(e,o),o}buildVao(t){this.setProgram(t),this.bindVertexArray(t.vao);const e=this.gl;st(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),VM(e,t,this.vertexBuffer)}deleteProgram(t){this.throwIfDisposed(),t===this.program&&(this.program=null),t!=null&&(st(this.gl,()=>this.gl.deleteProgram(t)),this.deleteVertexArray(t.vao))}setProgram(t){this.throwIfDisposed(),this.program=t,this.program!=null&&this.debug&&Xd(this.gl,this.program),st(this.gl,()=>this.gl.useProgram(t))}getUniformLocation(t,e,s=!0){return this.throwIfDisposed(),s?NL(this.gl,t,e):TL(this.gl,t,e)}getAttributeLocation(t,e){return this.throwIfDisposed(),st(this.gl,()=>this.gl.getAttribLocation(t,e))}getUniformLocationNoThrow(t,e){return this.throwIfDisposed(),this.gl.getUniformLocation(t,e)}setInputMatrixTexture(t,e,s){this.throwIfDisposed(),this.throwIfNoProgram(),EL(this.gl,t,e,s)}setOutputMatrixTexture(t,e,s){this.setOutputMatrixTextureDriver(t,s,e)}setOutputPackedMatrixTexture(t,e,s){this.throwIfDisposed();const[o,r]=Uo(e,s);this.setOutputMatrixTextureDriver(t,o,r)}setOutputMatrixWriteRegion(t,e,s,o){this.setOutputMatrixWriteRegionDriver(s,t,o,e)}setOutputPackedMatrixWriteRegion(t,e,s,o){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){this.program!=null&&Xd(this.gl,this.program),ac(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const t=this.gl;if(this.debug){const e=this.getVertexArray();console.assert(e===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()}st(t,()=>t.drawElements(t.TRIANGLES,6,t.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),st(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension==null&&(this.disjointQueryTimerExtension=ic(this.gl,U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.createQuery();return s.beginQuery(o.TIME_ELAPSED_EXT,r),r}const t=this.getQueryTimerExtensionWebGL1(),e=t.createQueryEXT();return t.beginQueryEXT(t.TIME_ELAPSED_EXT,e),e}endQuery(){if(U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const e=this.gl,s=this.getQueryTimerExtensionWebGL2();e.endQuery(s.TIME_ELAPSED_EXT);return}const t=this.getQueryTimerExtensionWebGL1();t.endQueryEXT(t.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(t){return await xp(()=>this.disposed||this.isQueryAvailable(t,U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(t,U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(t,e){if(e===0)return null;if(e===2){const s=this.gl;return s.getQueryParameter(t,s.QUERY_RESULT)/1e6}else{const s=this.getQueryTimerExtensionWebGL1();return s.getQueryObjectEXT(t,s.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(t,e){if(e===0)return!0;if(e===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.getQueryParameter(t,s.QUERY_RESULT_AVAILABLE);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(o.GPU_DISJOINT_EXT)),r&&!this.disjoint}else{const s=this.getQueryTimerExtensionWebGL1(),o=s.getQueryObjectEXT(t,s.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(s.GPU_DISJOINT_EXT)),o&&!this.disjoint}}pollFence(t){return new Promise(e=>{this.addItemToPoll(()=>t.isFencePassed(),()=>e())})}pollItems(){const t=XM(this.itemsToPoll.map(e=>e.isDoneFn));for(let e=0;e<=t;++e){const{resolveFn:s}=this.itemsToPoll[e];s()}this.itemsToPoll=this.itemsToPoll.slice(t+1)}addItemToPoll(t,e){if(this.itemsToPoll.push({isDoneFn:t,resolveFn:e}),this.itemsToPoll.length>1)return;let s;"setTimeoutCustom"in U().platform&&(s=U().platform.setTimeoutCustom.bind(U().platform)),xp(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,s)}bindTextureToFrameBuffer(t){this.throwIfDisposed(),Yd(this.gl,t,this.framebuffer),this.debug&&ac(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture!=null?(Yd(this.gl,this.outputTexture,this.framebuffer),this.debug&&ac(this.gl)):I1(this.gl,this.framebuffer)}downloadMatrixDriver(t,e){this.bindTextureToFrameBuffer(t);const s=e();return this.unbindTextureToFrameBuffer(),s}setOutputMatrixTextureDriver(t,e,s){this.throwIfDisposed();const o=this.gl;Yd(o,t,this.framebuffer),this.debug&&ac(o),this.outputTexture=t,st(o,()=>o.viewport(0,0,e,s)),st(o,()=>o.scissor(0,0,e,s))}setOutputMatrixWriteRegionDriver(t,e,s,o){this.throwIfDisposed(),st(this.gl,()=>this.gl.scissor(t,e,s,o))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(this.program==null)throw new Error("No GPU program is currently set.")}}function XM(n){let t=0;for(;t<n.length&&n[t]();++t);return t-1}const{addImpl:YM,bincountImpl:L1,bincountReduceImpl:ZM,bitwiseAndImpl:JM,castImpl:QM,ceilImpl:tP,concatImpl:eP,equalImpl:nP,expImpl:sP,expm1Impl:oP,floorImpl:rP,gatherNdImpl:iP,gatherV2Impl:aP,greaterImpl:lP,greaterEqualImpl:cP,lessImpl:uP,lessEqualImpl:hP,linSpaceImpl:dP,logImpl:pP,maxImpl:fP,maximumImpl:mP,minimumImpl:gP,multiplyImpl:xP,negImpl:bP,notEqualImpl:yP,prodImpl:wP,raggedGatherImpl:CP,raggedRangeImpl:IP,raggedTensorToTensorImpl:$P,rangeImpl:kP,rsqrtImpl:vP,scatterImpl:SP,sigmoidImpl:NP,simpleAbsImpl:M1,sliceImpl:TP,sparseFillEmptyRowsImpl:EP,sparseReshapeImpl:RP,sparseSegmentReductionImpl:P1,sqrtImpl:AP,staticRegexReplaceImpl:DP,stridedSliceImpl:FP,stringNGramsImpl:OP,stringSplitImpl:_P,stringToHashBucketFastImpl:LP,subImpl:MP,tileImpl:PP,topKImpl:BP,transposeImpl:op,uniqueImpl:zP}=HR;function B1(n,t){return["x","y","z","w","u","v"].slice(0,t).map(e=>`${n}.${e}`)}function De(n,t){return t===1?[n]:B1(n,t)}function VP(n,t){if(n===1)return"rc";let e="";for(let s=0;s<n;s++)e+=t[s],s<n-1&&(e+=",");return e}class WP{constructor(t){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=t,this.rank=t.length,this.enableShapeUniforms=Se(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const e=De("rc",this.rank),s=Ot(this.rank),o=this.getOutOfBoundsCondition(e),r=this.getSetup(e),i=this.getOutput(e);this.userCode=`
        void main() {
          ${s} rc = getOutputCoords();

          if(${o}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(t){const e=[];for(let s=0;s<=1;s++)for(let o=0;o<=1;o++){let r=`${s===0?"r":"rp1"}, ${o===0?"c":"cp1"}`;for(let i=2;i<this.rank;i++)r=`${t[t.length-1-i]},`+r;e.push(r)}return e}getOutOfBoundsCondition(t){if(this.rank===1)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let e="";for(let s=this.rank-2;s<this.rank;s++)e+=`${t[s]} >= ${this.enableShapeUniforms?`outShape[${s}]`:this.outputShape[s]}`,s<this.rank-1&&(e+="||");return e}getSetup(t){if(this.rank===1)return"";const e=t.slice(-2),s=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],o=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${e[0]};
      int c = ${e[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${s};
      bool rEdge = rp1 >= ${o};
    `}getOutput(t){const e=this.getSourceCoordsArr(t);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?"outShape":this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${e[0]}),
            cEdge ? 0. : getA(${e[1]}),
            rEdge ? 0. : getA(${e[2]}),
            rEdge || cEdge ? 0. : getA(${e[3]})`}}class z1{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length);let s="";for(let o=0;o<4;o++){let r="thisRC = rc;";o%2===1&&(r+="thisRC.z += 1;"),o>1&&(r+="thisRC.y += 1;"),s+=`
        ${r}
        ${o>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${o}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${o>0?"}":""}
      `}this.userCode=`
      ${UP(e,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?ep():tp(t)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":t[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":t[2]};

        ${s}

        setOutput(result);
      }
    `}}function UP(n,t){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${t?VL(["r","c","d"],"inputShape"):lo(["r","c","d"],n)}
      return ivec3(r, c, d);
    }
  `}class GP{constructor(t){this.gpgpu=t,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(t,e,s){const o=W1(e,s),r=U1(t,o,s);r in this.freeTextures||(this.freeTextures[r]=[]),r in this.usedTextures||(this.usedTextures[r]=[]);const i=V1(t,o,this.gpgpu.gl,this.gpgpu.textureConfig,s);if(this.freeTextures[r].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=i,this.log();const l=this.freeTextures[r].pop();return this.usedTextures[r].push(l),l}let a;return o===be.PACKED_2X2_FLOAT32?a=this.gpgpu.createPackedMatrixTexture(t[0],t[1]):o===be.PACKED_2X2_FLOAT16?a=this.gpgpu.createFloat16PackedMatrixTexture(t[0],t[1]):o===be.UNPACKED_FLOAT32?a=this.gpgpu.createFloat32MatrixTexture(t[0],t[1]):o===be.UNPACKED_FLOAT16?a=this.gpgpu.createFloat16MatrixTexture(t[0],t[1]):o===be.PACKED_4X1_UNSIGNED_BYTE&&(a=this.gpgpu.createUnsignedBytesMatrixTexture(t[0],t[1])),this.usedTextures[r].push(a),this.numUsedTextures++,this._numBytesAllocated+=i,this.log(),a}releaseTexture(t,e,s,o){if(this.freeTextures==null)return;const r=W1(s,o),i=U1(e,r,o);i in this.freeTextures||(this.freeTextures[i]=[]);const a=V1(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,o),l=U().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");l!==-1&&this._numBytesAllocated>l?(this.gpgpu.deleteMatrixTexture(t.texture),this._numBytesAllocated-=a):(this.freeTextures[i].push(t),this.numFreeTextures++,this._numBytesFree+=a),this.numUsedTextures--;const c=this.usedTextures[i],u=c&&c.indexOf(t);if(u==null||u<0)throw new Error("Cannot release a texture that was never provided by this texture manager");c[u]=c[c.length-1],c.pop(),this.log()}log(){if(!this.logEnabled)return;const t=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${t})`);const e=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*e)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(const t in this.freeTextures)this.freeTextures[t].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(const t in this.usedTextures)this.usedTextures[t].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function HP(n,t){const e=n;if(t===e.R32F)return 4;if(t===e.R16F)return 2;if(t===e.RGBA32F)return 16;if(t===n.RGBA)return 16;if(t===e.RGBA16F)return 8;if(t===e.RGBA8)return 4;throw new Error(`Unknown internal format ${t}`)}function V1(n,t,e,s,o){const r=qP(t,s);let i;if(o){const[l,c]=Uo(n[0],n[1]);i=l*c}else{const[l,c]=Pi(n[0],n[1]);i=l*c}const a=HP(e,r);return i*a}function qP(n,t){switch(n){case be.PACKED_2X2_FLOAT32:return O1(t);case be.PACKED_2X2_FLOAT16:return _1(t);case be.UNPACKED_FLOAT32:return A1(t);case be.UNPACKED_FLOAT16:return D1(t);case be.PACKED_4X1_UNSIGNED_BYTE:return F1(t);default:throw new Error(`Unknown physical texture type ${n}`)}}function KP(n){return U().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?n?be.PACKED_2X2_FLOAT32:be.UNPACKED_FLOAT32:n?be.PACKED_2X2_FLOAT16:be.UNPACKED_FLOAT16}function W1(n,t){if(n===Qe.UPLOAD)return be.PACKED_2X2_FLOAT32;if(n===Qe.RENDER||n==null)return KP(t);if(n===Qe.DOWNLOAD||n===Qe.PIXELS)return be.PACKED_4X1_UNSIGNED_BYTE;throw new Error(`Unknown logical texture type ${n}`)}function U1(n,t,e){return`${n[0]}_${n[1]}_${t}_${e}`}class Wn{constructor(t,e){this.variableNames=["A"],this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${e}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}const dn="if (isnan(x)) return x;",jP="return x;",G1="return abs(x);",XP="return (x >= 0.0) ? x : (exp(x) - 1.0);",YP=dn+`
  return (x < 0.0) ? 0.0 : x;
`,ZP=dn+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,ks="return x;",JP="return 1.0 / (1.0 + exp(-1.0 * x));";const QP="return x;",t3=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,e3=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,n3=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,s3="return 1.0 / (1.0 + exp(-1.0 * x));";class vs{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${e}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}class o3{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length);const e=t.length,s=De("rc",e),o=Ot(e),r=VP(e,s),i=s.slice(-2),a=e<=1?"rc":`vec2(${i.join(",")})`;this.userCode=`
      void main() {
        ${o} rc = getOutputCoords();
        vec4 packedInput = getA(${r});

        setOutput(getChannel(packedInput, ${a}));
      }
    `}}const r3=bm,i3=1e-7,a3=1e-4,dc={};function l3(n){return n in dc||(dc[n]={}),dc[n]}const c3=U().getNumber("CPU_HANDOFF_SIZE_THRESHOLD"),u3=600;function h3(){return U().global.screen==null?1024:U().global.screen.height*U().global.screen.width*window.devicePixelRatio*u3/1024/1024}class pc extends $c{nextDataId(){return pc.nextDataId++}constructor(t){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!U().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");let e;if(t!=null){if(t instanceof sp)e=t;else{const s=kn(U().getNumber("WEBGL_VERSION"),t);e=new sp(s)}this.binaryCache={},this.gpgpuCreatedLocally=!1}else{const s=kn(U().getNumber("WEBGL_VERSION"));e=new sp(s),this.binaryCache=l3(U().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0}this.gpgpu=e,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new GP(this.gpgpu),this.numMBBeforeWarning=h3(),this.texData=new gp(this,Nn())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(t,e,s,o,r,i){const a=this.makeTensorInfo(e,s),l=this.texData.get(a.dataId);l.isPacked=!1,l.texture={texture:t,texShape:[o,r]},l.texShape=[o,r];const c=lc(e),u=new R1(c,!1,i),h=this.runWebGLProgram(u,[a],s,[[o,r]]);return h.shape=e,l.texture=null,this.disposeIntermediateTensorInfo(a),h.dataId}write(t,e,s){if((U().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||U().getBool("DEBUG"))&&this.checkNumericalProblems(t),s==="complex64"&&t!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const o={id:this.nextDataId()};return this.texData.set(o,{shape:e,dtype:s,values:t,usage:Qe.UPLOAD,refCount:1}),o}refCount(t){return this.texData.has(t)?this.texData.get(t).refCount:0}incRef(t){const e=this.texData.get(t);e.refCount++}decRef(t){if(this.texData.has(t)){const e=this.texData.get(t);e.refCount--}}move(t,e,s,o,r){if(U().getBool("DEBUG")&&this.checkNumericalProblems(e),o==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(t,{shape:s,dtype:o,values:e,usage:Qe.UPLOAD,refCount:r})}disposeIntermediateTensorInfo(t){this.disposeData(t.dataId)}readSync(t){const e=this.texData.get(t),{values:s,dtype:o,complexTensorInfos:r,slice:i,shape:a,isPacked:l}=e;if(i!=null){let d;l?d=new vs(a,ks):d=new Wn(a,ks);const p=this.runWebGLProgram(d,[{dataId:t,shape:a,dtype:o}],o),f=this.readSync(p.dataId);return this.disposeIntermediateTensorInfo(p),f}if(s!=null)return this.convertAndCacheOnCPU(t);if(o==="string")return s;const c=this.activeTimers!=null;let u;c&&(u=Oe());let h;if(o==="complex64"){const d=this.readSync(r.real.dataId),p=this.readSync(r.imag.dataId);h=Yn(d,p)}else h=this.getValuesFromTexture(t);return c&&(this.downloadWaitMs+=Oe()-u),this.convertAndCacheOnCPU(t,h)}async read(t){if(this.pendingRead.has(t)){const f=this.pendingRead.get(t);return new Promise(m=>f.push(m))}const e=this.texData.get(t),{values:s,shape:o,slice:r,dtype:i,complexTensorInfos:a,isPacked:l}=e;if(r!=null){let f;l?f=new vs(o,ks):f=new Wn(o,ks);const m=this.runWebGLProgram(f,[{dataId:t,shape:o,dtype:i}],i),g=this.read(m.dataId);return this.disposeIntermediateTensorInfo(m),g}if(s!=null)return this.convertAndCacheOnCPU(t);if(U().getBool("DEBUG")&&!U().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&U().getNumber("WEBGL_VERSION")===2)throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let c=null,u;if(i!=="complex64"&&U().get("WEBGL_BUFFER_SUPPORTED")){u=this.decode(t);const f=this.texData.get(u.dataId);c=this.gpgpu.createBufferFromTexture(f.texture.texture,...rc(o))}this.pendingRead.set(t,[]),i!=="complex64"&&await this.gpgpu.createAndWaitForFence();let h;if(i==="complex64"){const f=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),m=f[0],g=f[1];h=Yn(m,g)}else if(c==null)h=this.getValuesFromTexture(t);else{const f=q(o);h=this.gpgpu.downloadFloat32MatrixFromBuffer(c,f)}if(u!=null&&this.disposeIntermediateTensorInfo(u),c!=null){const f=this.gpgpu.gl;st(f,()=>f.deleteBuffer(c))}const d=this.convertAndCacheOnCPU(t,h),p=this.pendingRead.get(t);return this.pendingRead.delete(t),p.forEach(f=>f(d)),this.pendingDisposal.has(t)&&(this.pendingDisposal.delete(t),this.disposeData(t)&&Nn().removeDataId(t,this),this.pendingDeletes--),d}readToGPU(t,e={}){const s=this.texData.get(t),{values:o,shape:r,slice:i,dtype:a,isPacked:l,texture:c}=s;if(a==="complex64")throw new Error("Does not support reading texture for complex64 dtype.");if(i!=null){let p;l?p=new vs(r,ks):p=new Wn(r,ks);const f=this.runWebGLProgram(p,[{dataId:t,shape:r,dtype:a}],a),m=this.readToGPU(f,e);return this.disposeIntermediateTensorInfo(f),m}if(c==null)throw o!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const u=this.decode(t,e.customTexShape),h=Nn().makeTensorFromTensorInfo(u),d=this.texData.get(u.dataId);return Object.assign({tensorRef:h},d.texture)}bufferSync(t){const e=this.readSync(t.dataId);if(t.dtype==="string")try{const s=e.map(o=>as(o));return wt(t.shape,t.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return wt(t.shape,t.dtype,e)}checkNumericalProblems(t){if(t!=null)for(let e=0;e<t.length;e++){const s=t[e];if(!fL(s))throw U().getBool("WEBGL_RENDER_FLOAT32_CAPABLE")?Error(`The value ${s} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${s} cannot be represented on this device.`)}}getValuesFromTexture(t){const{shape:e,dtype:s,isPacked:o}=this.texData.get(t),r=q(e);if(U().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const d=this.decode(t),p=this.texData.get(d.dataId),f=this.gpgpu.downloadMatrixFromPackedTexture(p.texture.texture,...rc(e)).subarray(0,r);return this.disposeIntermediateTensorInfo(d),f}const i=U().getBool("WEBGL_PACK")&&o===!0,a=i?lc(e):e,l=i?new RM(a):new EM(a),c=this.runWebGLProgram(l,[{shape:a,dtype:s,dataId:t}],"float32"),u=this.texData.get(c.dataId),h=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,r);return this.disposeIntermediateTensorInfo(c),h}timerAvailable(){return U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(t){const e=this.activeTimers,s=[];let o=!1;this.programTimersStack==null?(this.programTimersStack=s,o=!0):this.activeTimers.push(s),this.activeTimers=s,t();const r=_s(this.activeTimers.map(l=>l.query)).filter(l=>l!=null),i=_s(this.activeTimers.map(l=>l.name)).filter(l=>l!=null);this.activeTimers=e,o&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const l=await Promise.all(r);a.kernelMs=qy(l),a.getExtraProfileInfo=()=>l.map((c,u)=>({name:i[u],ms:c})).map(c=>`${c.name}: ${c.ms}`).join(", ")}else a.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,a})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:Oe(),endMs:null}}endTimer(t){return U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),t):(t.endMs=Oe(),t)}async getQueryTime(t){if(U().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(t);const e=t;return e.endMs-e.startMs}disposeData(t,e=!1){if(this.pendingDisposal.has(t))return!1;if(!this.texData.has(t))return!0;if(e?this.texData.get(t).refCount=0:this.texData.get(t).refCount--,!e&&this.texData.get(t).refCount>0)return!1;if(this.pendingRead.has(t))return this.pendingDisposal.add(t),this.pendingDeletes++,!1;this.releaseGPUData(t);const{complexTensorInfos:s}=this.texData.get(t);return s!=null&&(this.disposeData(s.real.dataId,e),this.disposeData(s.imag.dataId,e)),this.texData.delete(t),!0}releaseGPUData(t){const{texture:e,dtype:s,texShape:o,usage:r,isPacked:i,slice:a}=this.texData.get(t),l=a&&a.origDataId||t,c=this.dataRefCount.get(l);c>1?this.dataRefCount.set(l,c-1):(this.dataRefCount.delete(l),e!=null&&(this.numBytesInGPU-=this.computeBytes(o,s),this.textureManager.releaseTexture(e,o,r,i)));const u=this.texData.get(t);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(t){return this.uploadToGPU(t),this.texData.get(t).texture.texture}getDataInfo(t){return this.texData.get(t)}shouldExecuteOnCPU(t,e=c3){return U().getBool("WEBGL_CPU_FORWARD")&&t.every(s=>this.texData.get(s.dataId).texture==null&&q(s.shape)<e)}getGPGPUContext(){return this.gpgpu}where(t){je("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const e=t.dataSync();return r3(t.shape,e)}packedUnaryOp(t,e,s){const o=new vs(t.shape,e),r=this.compileAndRun(o,[t],s);return Nn().makeTensorFromTensorInfo(r)}abs(t){if(this.shouldExecuteOnCPU([t])&&t.dtype!=="complex64"){const o=M1(this.texData.get(t.dataId).values);return this.makeOutput(t.shape,t.dtype,o)}if(U().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,G1,t.dtype);const e=new Wn(t.shape,G1),s=this.compileAndRun(e,[t]);return Nn().makeTensorFromTensorInfo(s)}makeTensorInfo(t,e,s){let o;if(e==="string"&&s!=null&&s.length>0&&er(s[0])){const r=s.map(i=>is(i));o=this.write(r,t,e)}else o=this.write(s,t,e);return this.texData.get(o).usage=null,{dataId:o,shape:t,dtype:e}}makeOutput(t,e,s){return Nn().makeTensorFromTensorInfo(this.makeTensorInfo(t,e,s),this)}unpackTensor(t){const e=new o3(t.shape);return this.runWebGLProgram(e,[t],t.dtype)}packTensor(t){const e=new WP(t.shape);return this.runWebGLProgram(e,[t],t.dtype,null,!0)}packedReshape(t,e){const s=[Go(t.shape),...Ho(t.shape)],o={dtype:t.dtype,shape:s,dataId:t.dataId},r=[Go(e),...Ho(e)],i=new z1(r,s),a=!0,l=[s],c=this.runWebGLProgram(i,[o],t.dtype,l,a);return{dataId:c.dataId,shape:e,dtype:c.dtype}}decode(t,e){const s=this.texData.get(t),{isPacked:o,shape:r,dtype:i}=s;if(e!=null){const d=q(r),p=e[0]*e[1]*4;S(d<=p,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}const a=lc(r);let l;o?l=new TM(a):l=new NM(a);const c=!0,u=[e??rc(a)],h=this.runWebGLProgram(l,[{shape:a,dtype:i,dataId:t}],i,u,c,e);return{dtype:i,shape:r,dataId:h.dataId}}runWebGLProgram(t,e,s,o,r=!1,i){const a=this.makeTensorInfo(t.outputShape,s),l=this.texData.get(a.dataId);if(t.packedOutput&&(l.isPacked=!0),t.outPackingScheme===Mi.DENSE){const x=i??rc(t.outputShape);l.texShape=x.map(b=>b*2)}if(t.outTexUsage!=null&&(l.usage=t.outTexUsage),q(a.shape)===0)return l.values=Ce(a.dtype,0),a;const c=[],u=e.map(x=>{if(x.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let b=this.texData.get(x.dataId);if(b.texture==null){if(!t.packedInputs&&q(x.shape)<=U().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:x.shape,texData:null,isUniform:!0,uniformValues:b.values};t.packedInputs&&(b.isPacked=!0,b.shape=x.shape)}if(this.uploadToGPU(x.dataId),!!b.isPacked!=!!t.packedInputs)x=b.isPacked?this.unpackTensor(x):this.packTensor(x),c.push(x),b=this.texData.get(x.dataId);else if(b.isPacked&&!uc(b.shape,x.shape)){const w=x,y=x.shape;x.shape=b.shape,x=this.packedReshape(x,y),c.push(x),b=this.texData.get(x.dataId),w.shape=y}return{shape:x.shape,texData:b,isUniform:!1}});this.uploadToGPU(a.dataId);const h={shape:a.shape,texData:l,isUniform:!1},d=SM(t,u,h),p=this.getAndSaveBinary(d,()=>kM(this.gpgpu,t,u,h)),f=this.activeTimers!=null;let m;f&&(m=this.startTimer()),U().get("ENGINE_COMPILE_ONLY")||vM(this.gpgpu,p,u,h,o),c.forEach(x=>this.disposeIntermediateTensorInfo(x)),f&&(m=this.endTimer(m),this.activeTimers.push({name:t.constructor.name,query:this.getQueryTime(m)}));const g=U().getNumber("WEBGL_FLUSH_THRESHOLD");if(g>0){const x=Oe();x-this.lastGlFlushTime>g&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=x)}if(!U().getBool("WEBGL_LAZILY_UNPACK")&&l.isPacked&&r===!1){const x=this.unpackTensor(a);return this.disposeIntermediateTensorInfo(a),x}return a}compileAndRun(t,e,s,o,r=!1){return s=s||e[0].dtype,this.runWebGLProgram(t,e,s,o,r)}getAndSaveBinary(t,e){return t in this.binaryCache||(this.binaryCache[t]=e()),this.binaryCache[t]}getTextureManager(){return this.textureManager}dispose(){this.disposed||(U().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<"u"&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return this.floatPrecisionValue==null&&(this.floatPrecisionValue=V(()=>{if(!U().get("WEBGL_RENDER_FLOAT32_ENABLED")){const t=U().getBool("DEBUG");U().set("DEBUG",!1);const e=this.abs(At(1e-8)).dataSync()[0];if(U().set("DEBUG",t),e>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?i3:a3}uploadToGPU(t){const e=this.texData.get(t),{shape:s,dtype:o,values:r,texture:i,usage:a,isPacked:l}=e;if(i!=null)return;const c=this.activeTimers!=null;let u;c&&(u=Oe());let h=e.texShape;if(h==null&&(h=DL(s,l),e.texShape=h),r!=null){const d=lc(s);let p,f=h[1],m=h[0];const g=r instanceof Uint8Array||r instanceof Uint8ClampedArray;(l||!g)&&([f,m]=Uo(h[0],h[1])),l?p=new DM(d,g):p=new R1(d,g);const x=g?[m,f]:h,b=this.makeTensorInfo(x,o),w=this.texData.get(b.dataId);g?w.usage=Qe.PIXELS:w.usage=Qe.UPLOAD,w.texShape=x,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(b.dataId),f,m,r);const y=[[m,f]],$=this.runWebGLProgram(p,[b],o,y,!0),N=this.texData.get($.dataId);e.texShape=N.texShape,e.isPacked=N.isPacked,e.usage=N.usage,U().get("ENGINE_COMPILE_ONLY")?this.disposeData($.dataId):(e.texture=N.texture,e.values=null,this.texData.delete($.dataId)),this.disposeIntermediateTensorInfo(b),c&&(this.uploadWaitMs+=Oe()-u)}else{const d=this.acquireTexture(h,a,o,l);e.texture=d}}convertAndCacheOnCPU(t,e){const s=this.texData.get(t),{dtype:o}=s;return e!=null&&(s.values=d3(e,o)),s.values}acquireTexture(t,e,s,o){if(this.numBytesInGPU+=this.computeBytes(t,s),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){const r=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${r} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(t,e,o)}computeBytes(t,e){return t[0]*t[1]*Yi(e)}checkCompileCompletion(){for(const[,t]of Object.entries(this.binaryCache))this.checkCompletion_(t)}async checkCompileCompletionAsync(){const t=[];if(this.gpgpu.parallelCompilationExtension){for(const[,e]of Object.entries(this.binaryCache))t.push(this.checkCompletionAsync_(e));return Promise.all(t)}else{for(const[,e]of Object.entries(this.binaryCache)){const s=new Promise(o=>{try{this.checkCompletion_(e),o(!0)}catch(r){throw r}});t.push(s)}return Promise.all(t)}}async checkCompletionAsync_(t){return this.gpgpu.gl.getProgramParameter(t.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(t):(await zm(),this.checkCompletionAsync_(t))}checkCompletion_(t){if(this.gpgpu.gl.getProgramParameter(t.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(t.webGLProgram)),this.gpgpu.gl.getShaderParameter(t.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(w1(t.source,this.gpgpu.gl.getShaderInfoLog(t.fragmentShader)),new Error("Failed to compile fragment shader.")):new Error("Failed to link vertex and fragment shaders.");return!0}getUniformLocations(){for(const t of Object.values(this.binaryCache)){this.gpgpu.buildVao(t.webGLProgram);const{variablesLocations:e,customUniformLocations:s,infLoc:o,nanLoc:r,outShapeLocation:i,outShapeStridesLocation:a,outTexShapeLocation:l}=T1(this.gpgpu,t.program,t.webGLProgram);t.variablesLocations=e,t.customUniformLocations=s,t.infLoc=o,t.nanLoc=r,t.outShapeLocation=i,t.outShapeStridesLocation=a,t.outTexShapeLocation=l}}createTensorFromGPUData(t,e,s){t.channels=t.channels||"RGBA";const{texture:o,height:r,width:i,channels:a}=t,l=Nn().backend;if(!l.gpgpu.gl.isTexture(o))throw new Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");const c=l.writeTexture(o,e,s,r,i,a);return Nn().makeTensorFromDataId(c,e,s,l)}}pc.nextDataId=0;function d3(n,t){if(t==="float32"||t==="complex64")return n;if(t==="int32"||t==="bool"){const e=t==="int32"?new Int32Array(n.length):new Uint8Array(n.length);for(let s=0;s<e.length;++s)e[s]=Math.round(n[s]);return e}else throw new Error(`Unknown dtype ${t}`)}mf()&&If("webgl",()=>new pc,2);const rp=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class uo{constructor(t,e,s){this.variableNames=["A","B"],this.outputShape=mt(e,s),this.enableShapeUniforms=Se(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${t}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}const ho=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class Yo{constructor(t,e,s,o=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=mt(e,s);const r=this.outputShape.length;this.enableShapeUniforms=Se(r);let i="";if(o)if(r===0||q(this.outputShape)===1)i=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(i=`
          ${Ot(r)} coords = getOutputCoords();
        `,r===1)this.enableShapeUniforms?i+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:i+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const l=De("coords",r);this.enableShapeUniforms?i+=`
            bool nextRowOutOfBounds =
              (${l[r-2]} + 1) >= outShape[${r} - 2];
            bool nextColOutOfBounds =
              (${l[r-1]} + 1) >= outShape[${r} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:i+=`
            bool nextRowOutOfBounds =
              (${l[r-2]} + 1) >= ${this.outputShape[r-2]};
            bool nextColOutOfBounds =
              (${l[r-1]} + 1) >= ${this.outputShape[r-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${t}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${i}

        setOutput(result);
      }
    `}}function Ke(n){const{inputs:t,backend:e}=n,{x:s}=t;return e.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const p3={kernelName:Ir,backendName:"webgl",kernelFunc:Ke};function Ss(n){const{inputs:t,backend:e}=n,{real:s,imag:o}=t,r=e.makeTensorInfo(s.shape,"complex64"),i=e.texData.get(r.dataId),a=Ke({inputs:{x:s},backend:e}),l=Ke({inputs:{x:o},backend:e});return i.complexTensorInfos={real:a,imag:l},r}const f3={kernelName:Wc,backendName:"webgl",kernelFunc:Ss};const H1="return (a < 0.) ? b * a : a;",q1=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function m3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{alpha:r}=s,i=e.makeTensorInfo([],"float32",rs(r,"float32")),a=U().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Yo(q1,o.shape,i.shape):new uo(H1,o.shape,i.shape),l=e.runWebGLProgram(a,[o,i],"float32");return e.disposeIntermediateTensorInfo(i),l}const g3={kernelName:xa,backendName:"webgl",kernelFunc:m3};const K1="return (a < 0.) ? b * a : a;",j1=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function x3(n){const{inputs:t,backend:e}=n,{x:s,alpha:o}=t,r=U().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Yo(j1,s.shape,o.shape):new uo(K1,s.shape,o.shape);return e.runWebGLProgram(r,[s,o],"float32")}const b3={kernelName:La,backendName:"webgl",kernelFunc:x3};const Zo="if (isnan(x)) return x;";function St({opSnippet:n,packedOpSnippet:t,cpuKernelImpl:e,dtype:s}){return({inputs:o,backend:r})=>{const{x:i}=o,a=r,l=s||i.dtype;if(a.shouldExecuteOnCPU([i])&&e!=null){const h=a.texData.get(i.dataId),d=e(h.values,l);return a.makeTensorInfo(i.shape,l,d)}const c=U().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&t!=null;let u;return c?u=new vs(i.shape,t):u=new Wn(i.shape,n),a.runWebGLProgram(u,[i],l)}}function ye({opSnippet:n,packedOpSnippet:t,checkOutOfBounds:e=!1,supportsComplex:s=!1,cpuKernelImpl:o,dtype:r}){return({inputs:i,backend:a})=>{const{a:l,b:c}=i,u=a;if(s&&l.dtype==="complex64"){const f=u.texData.get(l.dataId),m=u.texData.get(c.dataId),[g,x]=[[f.complexTensorInfos.real,m.complexTensorInfos.real],[f.complexTensorInfos.imag,m.complexTensorInfos.imag]].map(w=>{const[y,C]=w,$={dataId:y.dataId,dtype:y.dtype,shape:l.shape},N={dataId:C.dataId,dtype:C.dtype,shape:c.shape},T=new uo(n,l.shape,c.shape);return u.runWebGLProgram(T,[$,N],Ue(y.dtype,C.dtype))}),b=Ss({inputs:{real:g,imag:x},backend:u});return u.disposeIntermediateTensorInfo(g),u.disposeIntermediateTensorInfo(x),b}const h=r||Ue(l.dtype,c.dtype);if((l.dtype==="string"||c.dtype==="string"||u.shouldExecuteOnCPU([l,c]))&&o!=null){const f=u.texData.get(l.dataId).values,m=u.texData.get(c.dataId).values,g=l.dtype==="string"?Zn(f):f,x=l.dtype==="string"?Zn(m):m,[b,w]=o(l.shape,c.shape,g,x,h),y=u.makeTensorInfo(w,h),C=u.texData.get(y.dataId);return C.values=b,y}const d=U().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&t!=null;let p;return d?p=new Yo(t,l.shape,c.shape,e):p=new uo(n,l.shape,c.shape),u.runWebGLProgram(p,[l,c],h)}}function Vi(n,t=!1){if(n==="linear")return t?QP:jP;if(n==="relu")return t?e3:YP;if(n==="elu")return t?t3:XP;if(n==="relu6")return t?n3:ZP;if(n==="prelu")return t?j1:K1;if(n==="leakyrelu")return t?q1:H1;if(n==="sigmoid")return t?s3:JP;throw new Error(`Activation ${n} has not been implemented for the WebGL backend.`)}class X1{constructor(t,e,s,o=!1,r=!1,i=!1,a=null,l=!1,c=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s,this.enableShapeUniforms=Se(this.outputShape.length);const u=o?t[1]:t[2],h=Math.ceil(u/2),d=o?"i * 2, rc.y":"rc.y, i * 2",p=r?"rc.z, i * 2":"i * 2, rc.z",f=o?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],m=r?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let g="",x="";a&&(l?g=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${a}
        }`:c?g=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${a}
        }`:g=`vec4 activation(vec4 x) {
          ${a}
        }`,x="result = activation(result);");const b=i?"result += getBiasAtOutCoords();":"";i&&this.variableNames.push("bias"),l&&this.variableNames.push("preluActivationWeights"),c&&this.variableNames.push("leakyreluAlpha");let w="rc.x",y="rc.x";t[0]<e[0]?w=`imod(rc.x, ${t[0]})`:e[0]<t[0]&&(y=`imod(rc.x, ${e[0]})`),this.userCode=`
      ${g}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${h}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${w};
        int batchB = ${y};
        for (int i = 0; i < ${h}; i++) {
          vec4 a = getMatrixA(batchA, ${d});
          vec4 b = getMatrixB(batchB, ${p});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${f[0]} * ${m[0]});
          result += (${f[1]} * ${m[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${b}

        ${x}

        setOutput(result);
      }
    `}}const Y1={REAL:"return areal * breal - aimag * bimag;",IMAG:"return areal * bimag + aimag * breal;"};class Z1{constructor(t,e,s){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=mt(e,s),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${t}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}const J1="return a * b;";function ip(n){const{inputs:t,backend:e}=n,{a:s,b:o}=t,r=Ue(s.dtype,o.dtype);if(s.dtype==="complex64"){const a=e.texData.get(s.dataId),l=e.texData.get(o.dataId),c=new Z1(Y1.REAL,s.shape,o.shape),u=new Z1(Y1.IMAG,s.shape,o.shape),h=[{dataId:a.complexTensorInfos.real.dataId,dtype:a.complexTensorInfos.real.dtype,shape:s.shape},{dataId:a.complexTensorInfos.imag.dataId,dtype:a.complexTensorInfos.imag.dtype,shape:s.shape},{dataId:l.complexTensorInfos.real.dataId,dtype:l.complexTensorInfos.real.dtype,shape:o.shape},{dataId:l.complexTensorInfos.imag.dataId,dtype:l.complexTensorInfos.imag.dtype,shape:o.shape}],d=e.runWebGLProgram(c,h,"float32"),p=e.runWebGLProgram(u,h,"float32"),f=Ss({inputs:{real:d,imag:p},backend:e});return e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),f}if(e.shouldExecuteOnCPU([s,o])){const a=e.texData.get(s.dataId),l=e.texData.get(o.dataId),[c,u]=xP(s.shape,o.shape,a.values,l.values,r),h=e.makeTensorInfo(u,r),d=e.texData.get(h.dataId);return d.values=c,h}let i;return U().getBool("WEBGL_PACK_BINARY_OPERATIONS")?i=new Yo(J1,s.shape,o.shape):i=new uo(J1,s.shape,o.shape),e.runWebGLProgram(i,[s,o],r)}const y3={kernelName:Ar,backendName:"webgl",kernelFunc:ip};function w3(n,t,e){const s=[Go(n.shape),...Ho(n.shape)],o={dtype:n.dtype,shape:s,dataId:n.dataId},r=[Go(t),...Ho(t)],i=new z1(r,s),a=!0,l=[s],c=e.runWebGLProgram(i,[o],n.dtype,l,a);return{dataId:c.dataId,shape:t,dtype:c.dtype}}function et(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{shape:r}=s,i=e,a=q(o.shape),l=bp(r,a),c=q(l);S(a===c,()=>`The new shape (${l}) has ${c} elements and the old shape (${o.shape}) has ${a} elements. The new shape and old shape must have the same number of elements.`);const u=i.texData.get(o.dataId);return u.isPacked&&!uc(o.shape,l)&&!(u.texture!==null&&uc(u.shape,l))?w3(o,l,i):(i.incRef(o.dataId),{dataId:o.dataId,shape:l,dtype:o.dtype})}const C3={kernelName:Pa,backendName:"webgl",kernelFunc:et};class Q1{constructor(t,e){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:i}=t;this.outputShape=[o,i];const a=Math.floor(s/4)*4,l=s%4;let c="sumValue += dot(values, ones);";if(e!=null){const h=1/e;c=`sumValue += dot(values * ${xo(h)?h.toPrecision(2):h}, ones);`}let u="";r%s>0&&(u=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        float sumValue = 0.0;

        for (int i = 0; i < ${a}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${c}
        }

        int inIdx = inOffset + ${a};
        if (${l===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${c}
        } else if (${l===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${c}
        } else if (${l===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${c}
        }
        setOutput(sumValue);
      }
    `}}class I3{constructor(t,e){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:i}=t;this.outputShape=[o,i];let a="0.0",l="";e==="prod"?a="1.0":e==="min"?(a="1.0 / 1e-20",l="min"):e==="max"&&(a="-1.0 / 1e-20",l="max");let c=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="sum"?c="sumValue":e==="prod"?c="prodValue":e==="all"?c="allValue":e==="any"&&(c="anyValue");const u=Math.floor(s/4)*4,h=s%4;let d=`
      if (${e==="sum"}) {
        sumValue += dot(values, ones);
      } else if (${e==="prod"}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${l}(values, minMaxValue);
        if (${e==="min"} || ${e==="max"}) {
          minMaxValue = ${l}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,p="vec4";e==="all"?(a="1.0",d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,p="bvec4"):e==="any"&&(a="0.0",d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,p="bvec4");let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${a};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${f}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        vec4 minMaxValue = vec4(${a});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${u};
        if (${h===1}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${h===2}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${h===3}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${c});
      }
    `}}function $3(n){const t=[];for(;t.length===0||t[t.length-1].outSize!==1;){const e=t.length?t[t.length-1].outSize:n[1],s=kl(e);t.push({inSize:e,windowSize:s,outSize:Math.ceil(e/s)})}return t}function po(n,t,e,s){const o=$3(n.shape);let r=n;for(let i=0;i<o.length;i++){const{inSize:a,windowSize:l,outSize:c}=o[i];let u,h;e==="mean"?u=i===0?new Q1({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c},a):new Q1({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c}):u=new I3({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c},e),h=r,r=s.runWebGLProgram(u,[r],t),h.dataId!==n.dataId&&s.disposeIntermediateTensorInfo(h)}return r}class k3{constructor(t,e){this.variableNames=["A"];const s=new Array(t.length);for(let i=0;i<s.length;i++)s[i]=t[e[i]];this.outputShape=s,this.rank=s.length;const o=Ot(this.rank),r=v3(e);this.userCode=`
    void main() {
      ${o} resRC = getOutputCoords();
      setOutput(getA(${r}));
    }
    `}}function v3(n){const t=n.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);const e=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],s=new Array(t);for(let o=0;o<n.length;o++)s[n[o]]=e[o];return s.join()}class S3{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const s=new Array(t.length);for(let u=0;u<s.length;u++)s[u]=t[e[u]];if(this.outputShape=s,this.rank=s.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const o=Ot(this.rank),r=B1("rc",this.rank),i=new Array(this.rank);for(let u=0;u<e.length;u++)i[e[u]]=r[u];const a=`vec2(${i.slice(-2).join()})`,l=`++${r[this.rank-1]} < ${s[this.rank-1]}`,c=`getChannel(getA(${i.join()}), ${a})`;this.userCode=`
    void main() {
      ${o} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${c};
      if(${l}) {
        result[1] = ${c};
      }
      --${r[this.rank-1]};
      if(++${r[this.rank-2]} < ${s[this.rank-2]}) {
        result[2] = ${c};
        if(${l}) {
          result[3] = ${c};
        }
      }
      setOutput(result);
    }
    `}}function fc(n,t,e){const s=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new S3(n.shape,t):new k3(n.shape,t);return e.runWebGLProgram(s,[n],n.dtype)}function N3(n,t,e,s){const o=t,r=n.shape.length,i=yt(o,n.shape);let a=i;const l=qt(a,r),c=l!=null;let u=n;c&&(u=fc(n,l,s),a=Jt(a.length,r)),xe("sum",a,r);const[h,d]=de(u.shape,a);let p=h;e&&(p=ne(h,i));const f=q(d),g=q(n.shape)/f,x=et({inputs:{x:u},attrs:{shape:[g,f]},backend:s}),b=Du(n.dtype),w=po(x,b,"sum",s),y=et({inputs:{x:w},attrs:{shape:p},backend:s});return s.disposeIntermediateTensorInfo(x),s.disposeIntermediateTensorInfo(w),c&&s.disposeIntermediateTensorInfo(u),y}function mc(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;return N3(o,r,i,e)}const T3={kernelName:Ga,backendName:"webgl",kernelFunc:mc};function Fe(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{perm:r}=s,i=e,a=o.shape.length,l=new Array(a);for(let u=0;u<l.length;u++)l[u]=o.shape[r[u]];let c;if(i.shouldExecuteOnCPU([o])){const h=i.texData.get(o.dataId).values,d=op(h,o.shape,o.dtype,r,l);c=i.makeTensorInfo(l,o.dtype);const p=i.texData.get(c.dataId);p.values=d}else c=fc(o,r,i);return c}const E3={kernelName:Io,backendName:"webgl",kernelFunc:Fe};const ty=1e3;function gc({a:n,b:t,transposeA:e,transposeB:s,backend:o,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:a=0,activation:l=null}){const c=n.shape.length,u=t.shape.length,h=e?n.shape[c-2]:n.shape[c-1],d=s?t.shape[u-1]:t.shape[u-2],p=e?n.shape[c-1]:n.shape[c-2],f=s?t.shape[u-2]:t.shape[u-1],m=n.shape.slice(0,-2),g=t.shape.slice(0,-2),x=q(m),b=q(g),y=mt(n.shape.slice(0,-2),t.shape.slice(0,-2)).concat([p,f]);S(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${n.shape} and ${t.shape} and transposeA=${e} and transposeB=${s} must match.`);const C=e?[x,h,p]:[x,p,h],$=s?[b,f,d]:[b,d,f],N=et({inputs:{x:n},backend:o,attrs:{shape:C}}),T=et({inputs:{x:t},backend:o,attrs:{shape:$}}),k=[N,T],v=Math.max(x,b),I=e?N.shape[1]:N.shape[2],R=r!=null,A=i!=null,P=l==="leakyrelu",_=l!=null?Vi(l,!0):null,B=R||A||P||_!=null;let z;if((p===1||f===1)&&I>ty&&B===!1){let H=N,K=T;e&&(H=Fe({inputs:{x:N},backend:o,attrs:{perm:[0,2,1]}}),k.push(H)),s&&(K=Fe({inputs:{x:T},backend:o,attrs:{perm:[0,2,1]}}),k.push(K));const j=f!==1,X=f===1;let Y=H;j&&(Y=et({inputs:{x:H},backend:o,attrs:{shape:[v,I,1]}}),k.push(Y));const tt=f===1?2:1;let Q=K;X&&(Q=et({inputs:{x:K},backend:o,attrs:{shape:[v,1,I]}}),k.push(Q));const ot=ip({inputs:{a:Y,b:Q},backend:o});z=mc({inputs:{x:ot},backend:o,attrs:{axis:tt,keepDims:!0}}),k.push(ot)}else{const H=Ue(n.dtype,t.dtype),K=new X1(C,$,[v,p,f],e,s,R,_,A,P),j=[N,T];if(r!=null&&j.push(r),A&&j.push(i),P){const X=o.makeTensorInfo([],"float32",rs(a,"float32"));j.push(X),k.push(X)}z=o.runWebGLProgram(K,j,H)}const W=et({inputs:{x:z},backend:o,attrs:{shape:y}});k.push(z);for(const H of k)o.disposeIntermediateTensorInfo(H);return W}function R3(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r,bias:i,preluActivationWeights:a}=t,{transposeA:l,transposeB:c,activation:u,leakyreluAlpha:h}=s;return gc({a:o,b:r,transposeA:l,transposeB:c,backend:e,bias:i,preluActivationWeights:a,leakyreluAlpha:h,activation:u})}const A3={kernelName:Za,backendName:"webgl",kernelFunc:R3};const ey="return abs(x);";function D3(n){const{inputs:t,backend:e}=n,{x:s}=t;if(e.shouldExecuteOnCPU([s])&&s.dtype!=="complex64"){const r=e.texData.get(s.dataId),i=M1(r.values);return e.makeTensorInfo(s.shape,s.dtype,i)}let o;return U().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new vs(s.shape,ey):o=new Wn(s.shape,ey),e.runWebGLProgram(o,[s],s.dtype)}const F3={kernelName:Zi,backendName:"webgl",kernelFunc:D3};const O3=dn+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`,_3=St({opSnippet:O3}),L3={kernelName:nr,backendName:"webgl",kernelFunc:_3};const M3=dn+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`,P3=St({opSnippet:M3}),B3={kernelName:sr,backendName:"webgl",kernelFunc:P3};const ny="return a + b;",z3=ye({opSnippet:ny,packedOpSnippet:ny,supportsComplex:!0,cpuKernelImpl:YM}),V3={kernelName:Co,backendName:"webgl",kernelFunc:z3};class W3{constructor(t,e){this.outputShape=[],this.outputShape=t,this.variableNames=e.map((r,i)=>`T${i}`);const s=[];this.variableNames.forEach(r=>{s.push(`float v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        float result = ${o};
        setOutput(result);
      }
    `}}class U3{constructor(t,e){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.variableNames=e.map((r,i)=>`T${i}`);const s=[];this.variableNames.forEach(r=>{s.push(`vec4 v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        vec4 result = ${o};
        setOutput(result);
      }
    `}}function xc(n){const{inputs:t,backend:e}=n,s=t;if(s.length===1)return Ke({inputs:{x:s[0]},backend:e});if(s.length>U().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const l=Math.floor(s.length/2),c=xc({inputs:s.slice(0,l),backend:e}),u=xc({inputs:s.slice(l),backend:e});return xc({inputs:[c,u],backend:e})}const o=s.map(l=>l.dtype).reduce((l,c)=>Ue(l,c)),r=s.map(l=>l.shape),a=U().getBool("WEBGL_PACK")?new U3(s[0].shape,r):new W3(s[0].shape,r);return e.runWebGLProgram(a,s,o)}const G3={kernelName:_c,backendName:"webgl",kernelFunc:xc};function H3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=yt(r,o.shape);let c=l;const u=qt(c,a);let h=o;u!=null&&(h=Fe({inputs:{x:o},backend:e,attrs:{perm:u}}),c=Jt(c.length,a)),xe("all",c,a);const[d,p]=de(h.shape,c),f=q(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=po(m,m.dtype,"all",e);let x;if(i){const b=ne(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const q3={kernelName:Lc,backendName:"webgl",kernelFunc:H3};function K3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=yt(r,o.shape);let c=l;const u=qt(c,a);let h=o;u!=null&&(h=Fe({inputs:{x:o},backend:e,attrs:{perm:u}}),c=Jt(c.length,a)),xe("any",c,a);const[d,p]=de(h.shape,c),f=q(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=po(m,m.dtype,"any",e);let x;if(i){const b=ne(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const j3={kernelName:Mc,backendName:"webgl",kernelFunc:K3};class X3{constructor(t,e,s){this.variableNames=["A"];const{windowSize:o,batchSize:r,outSize:i}=t;s||this.variableNames.push("bestIndicesA"),this.outputShape=[r,i];const a=e==="max"?">":"<",l=s?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${o};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${o}; i++) {
          int inIdx = ${l};
          float candidate = getA(batch, inIdx);
          if (candidate ${a} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}class Y3{constructor(t,e,s,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,S(t.length>2,()=>`Packed arg${s.charAt(0).toUpperCase()+s.slice(1)} supports only inputs with rank above 2.`);const r=t[t.length-1],i=Math.ceil(r/e);this.outputShape=t.slice(0,-1),i>1&&this.outputShape.push(i),o||this.variableNames.push("bestIndicesA");const a=this.outputShape,l=a.length,c=Ot(l),u=De("coords",l);let h,d;if(i===1){d=l+1;const T=Ot(d);h=`
        ${T} sourceLocR = ${T}(${u.join()}, 0);
        ++${u[l-1]};
        ${T} sourceLocG = ${T}(${u.join()}, 0);
        ++${u[l-2]};
        ${T} sourceLocA = ${T}(${u.join()}, 0);
        --${u[l-1]};
        ${T} sourceLocB = ${T}(${u.join()}, 0);
        --${u[l-2]};`}else d=l,h=`
        ${c} sourceLocR = coords;
        ++${u[l-1]};
        ${c} sourceLocG = coords;
        ++${u[l-2]};
        ${c} sourceLocA = coords;
        --${u[l-1]};
        ${c} sourceLocB = coords;
        --${u[l-2]};`;const p=["x","y","z","w","u","v"].slice(0,d),f="."+p[d-1],m=p.map(T=>"int "+T),g=De("sourceLocR",d-1).concat("inIdx.r"),x=De("sourceLocG",d-1).concat("inIdx.g"),b=De("sourceLocB",d-1).concat("inIdx.b"),w=De("sourceLocA",d-1).concat("inIdx.a"),y=s==="max"?"greaterThan":"lessThan",C=o?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${x.join()}),
                             getBestIndicesAChannel(${b.join()}),
                             getBestIndicesAChannel(${w.join()})));`,$=`vec4(
            getAChannel(${g.join()}),
            hasNextCol ? getAChannel(${x.join()}) : 0.,
            hasNextRow ? getAChannel(${b.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${w.join()}) : 0.)`,N=o?"":`
      float getBestIndicesAChannel(${m.join()}) {
        return getChannel(getBestIndicesA(${p.join()}),
                                          vec2(${p.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${m.join()}) {
        return getChannel(getA(${p.join()}),
                               vec2(${p.slice(-2).join()}));
      }
      ${N}
      void main() {
        ${c} coords = getOutputCoords();
        bool hasNextCol = ${u[l-1]} < ${a[l-1]-1};
        bool hasNextRow = ${u[l-2]} < ${a[l-2]-1};
        ${h}
        ivec4 srcIdx = ivec4(sourceLocR${f}, sourceLocG${f},
          sourceLocB${f}, sourceLocA${f}) * ${e};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${$};

        for (int i = 0; i < ${e}; i++) {
          inIdx = srcIdx;
          ${C}
          vec4 candidate = ${$};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${y}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}function sy(n,t,e,s=null){let o=t.shape[0],r=t.shape[1];s!=null&&(o=s.shape[0],r=s.shape[1]);const i=kl(r),a={windowSize:i,inSize:r,batchSize:o,outSize:Math.ceil(r/i)},l=new X3(a,e,s==null),c=[t];s!=null&&c.push(s);const u=n.runWebGLProgram(l,c,"int32");if(u.shape[1]===1)return u;const h=sy(n,t,e,u);return n.disposeIntermediateTensorInfo(u),h}function oy(n,t,e,s=null){const o=s!=null?s.shape:t.shape,r=o[o.length-1],i=kl(r),a=new Y3(o,i,e,s==null),l=s==null?[t]:[t,s],c=n.runWebGLProgram(a,l,"int32");if(c.shape.length===t.shape.length){const u=oy(n,t,e,c);return n.disposeIntermediateTensorInfo(c),u}return c}function ry(n,t,e,s){const o=[e];if(xe("arg"+s.charAt(0).toUpperCase()+s.slice(1),o,t.shape.length),!U().getBool("WEBGL_PACK_REDUCE")||t.shape.length<=2){const r=[],i=n.texData.get(t.dataId),a=i!==null&&i.isPacked;let l=t;a&&(l=n.unpackTensor(t),r.push(l));const[c,u]=de(l.shape,o),h=q(u),d=et({inputs:{x:l},backend:n,attrs:{shape:[-1,h]}});r.push(d);const p=sy(n,d,s);r.push(p);const f=et({inputs:{x:p},backend:n,attrs:{shape:c}});return r.forEach(m=>n.disposeIntermediateTensorInfo(m)),f}return oy(n,t,s)}function Z3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;let i=yt(r,o.shape);const a=qt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=Fe({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=Jt(i.length,l.shape.length)),xe("argMax",[i[0]],l.shape.length);const u=ry(e,l,i[0],"max");return c.forEach(h=>e.disposeIntermediateTensorInfo(h)),u}const J3={kernelName:Ji,backendName:"webgl",kernelFunc:Z3};function Q3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;let i=yt(r,o.shape);const a=qt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=Fe({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=Jt(i.length,l.shape.length)),xe("argMin",[i[0]],l.shape.length);const u=ry(e,l,i[0],"min");return c.forEach(h=>e.disposeIntermediateTensorInfo(h)),u}const tB={kernelName:Qi,backendName:"webgl",kernelFunc:Q3};const eB=dn+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`,nB=St({opSnippet:eB}),sB={kernelName:or,backendName:"webgl",kernelFunc:nB};const oB=dn+"return log(x + sqrt(x * x + 1.0));",rB=St({opSnippet:oB}),iB={kernelName:rr,backendName:"webgl",kernelFunc:rB};const aB=dn+`
  return atan(x);
`,lB=St({opSnippet:aB}),cB={kernelName:ir,backendName:"webgl",kernelFunc:lB};const uB=rp+`
  return atan(a, b);
`,hB=`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+ho+`
  return result;
`,dB=ye({opSnippet:uB,packedOpSnippet:hB}),pB={kernelName:lr,backendName:"webgl",kernelFunc:dB};const fB=dn+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`,mB=St({opSnippet:fB}),gB={kernelName:ar,backendName:"webgl",kernelFunc:mB};class Wi{constructor(t,e,s,o=!1,r=!1){if(this.variableNames=["x"],e==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=t.filterWidth,a=t.strideHeight,l=t.strideWidth,c=t.dilationHeight,u=t.dilationWidth,h=t.effectiveFilterHeight,d=t.effectiveFilterWidth,p=t.padInfo.top,f=t.padInfo.left;this.outputShape=t.outShape;const m=e==="avg",g=`((batch  * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + d`,x=`(xR * ${t.inWidth} + xC) * ${t.inChannels} + d`;let b="0.0";if(m||(b="-1.0 / 1e-20"),s){this.userCode=`
        const ivec2 strides = ivec2(${a}, ${l});
        const ivec2 pads = ivec2(${p}, ${f});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${h};
              wR += ${c}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${o?r?g:x:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const w="max";let y=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="avg"&&(y="avgValue / max(count, 1.0)");const C=Math.floor(i/4)*4,$=i%4,N=`
      if (${m}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${w}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${a}, ${l});
      const ivec2 pads = ivec2(${p}, ${f});
      const float initializationValue = ${b};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${t.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${b});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${h};
            wR += ${c}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${t.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${C}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${N}
          }

          int xC = xCCorner + ${C};
          if (${$===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${N}
          } else if (${$===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${N}
          } else if (${$===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${N}
          }
        }
        setOutput(${y});
      }
    `}}class ap{constructor(t,e,s,o=!1,r=!1){if(this.variableNames=["x"],e==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=t.filterWidth,a=t.strideDepth,l=t.strideHeight,c=t.strideWidth,u=t.dilationDepth,h=t.dilationHeight,d=t.dilationWidth,p=t.effectiveFilterDepth,f=t.effectiveFilterHeight,m=t.effectiveFilterWidth,g=t.padInfo.front,x=t.padInfo.top,b=t.padInfo.left;this.outputShape=t.outShape;const w=e==="avg";let y="0.0";if(w||(y="-1.0 / 1e-20"),s){this.userCode=`
        const ivec3 strides =
            ivec3(${a}, ${l}, ${c});
        const ivec3 pads = ivec3(${g}, ${x}, ${b});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${p};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${t.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${f};
                wR += ${h}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${t.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${m};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${t.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${o?r?`(((batch * ${t.inDepth} + xD) * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`((xD * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`wD * ${f} * ${m} +
                      wR * ${m} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const C="max";let $=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="avg"&&($="avgValue / max(count, 1.0)");const N=Math.floor(i/4)*4,T=i%4,k=`
      if (${w}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${C}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${a}, ${l}, ${c});
      const ivec3 pads = ivec3(${g}, ${x}, ${b});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${t.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${p};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${t.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${f};
            wR += ${h}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${N}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${k}
            }

            int xC = xCCorner + ${N};
            if (${T===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${k}
            } else if (${T===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${k}
            } else if (${T===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${k}
            }
          }
        }
        setOutput(${$});
      }
    `}}function xB(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;Bi(o,"avgPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;S($e(i,c),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=nn(o.shape,r,i,c,a,l);if(u.filterWidth===1&&u.filterHeight===1&&Et(u.inShape,u.outShape))return Ke({inputs:{x:o},backend:e});const h=new Wi(u,"avg",!1);return e.runWebGLProgram(h,[o],"float32")}const bB={kernelName:ta,backendName:"webgl",kernelFunc:xB};function yB(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s,u=[1,1,1],h=Hn(o.shape,r,i,u,a,l,c),d=new ap(h,"avg",!1);return e.runWebGLProgram(d,[o],"float32")}const wB={kernelName:ea,backendName:"webgl",kernelFunc:yB};class CB{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=t.dilationHeight,a=t.dilationWidth,l=t.effectiveFilterHeight,c=t.effectiveFilterWidth,u=l-1-t.padInfo.top,h=c-1-t.padInfo.left,d=1/(e*s);this.userCode=`
      const ivec2 pads = ivec2(${u}, ${h});
      const float avgMultiplier = float(${d});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${l};
            wR += ${i}) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${c};
            wC+= ${a}) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class IB{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterDepth,s=t.filterHeight,o=t.filterWidth,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=t.dilationDepth,c=t.dilationHeight,u=t.dilationWidth,h=t.effectiveFilterDepth,d=t.effectiveFilterHeight,p=t.effectiveFilterWidth,f=h-1-t.padInfo.front,m=d-1-t.padInfo.top,g=p-1-t.padInfo.left,x=1/(e*s*o);this.userCode=`
      const ivec3 pads = ivec3(${f}, ${m}, ${g});
      const float avgMultiplier = float(${x});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${h};
            wD += ${l}) {
          float dyD = float(dyDCorner + wD) / ${r}.0;

          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${c}) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${p};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function $B(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r,{filterSize:a,strides:l,pad:c,dimRoundingMode:u}=s,h=[1,1,1],d=Hn(i.shape,a,l,h,c,u),p=new IB(d);return e.runWebGLProgram(p,[o],i.dtype)}const kB={kernelName:Bc,backendName:"webgl",kernelFunc:$B};function vB(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r;Bi([o,r],"avgPoolGrad");const{filterSize:a,strides:l,pad:c}=s,u=nn(i.shape,a,l,1,c),h=new CB(u);return e.runWebGLProgram(h,[o],i.dtype)}const SB={kernelName:Pc,backendName:"webgl",kernelFunc:vB};function NB(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r}=t,{transposeA:i,transposeB:a}=s;return gc({a:o,b:r,transposeA:i,transposeB:a,backend:e})}const TB={kernelName:na,backendName:"webgl",kernelFunc:NB};class EB{constructor(t,e,s,o,r,i){this.outputShape=[],this.variableNames=["x","mean","variance"],mt(t,e),mt(t,s);let a="0.0";o!=null&&(mt(t,o),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let l="1.0";r!=null&&(mt(t,r),this.variableNames.push("scale"),l="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${a};
        float scale = ${l};
        float inv = scale * inversesqrt(variance + float(${i}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}class RB{constructor(t,e,s,o,r,i){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],mt(t,e),mt(t,s);let a="vec4(0.0)";o!=null&&(mt(t,o),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let l="vec4(1.0)";r!=null&&(mt(t,r),this.variableNames.push("scale"),l="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`
      void main() {
        vec4 offset = ${a};
        vec4 scale = ${l};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${i}));

        setOutput((x - mean) * inv + offset);
      }
    `}}const AB={kernelName:fa,backendName:"webgl",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,mean:o,variance:r,offset:i,scale:a}=n;S(o.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),S(i==null||o.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),S(a==null||o.shape.length===a.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:l}=e;l==null&&(l=.001);const c=[s,o,r];let u=null;i!=null&&(u=i.shape,c.push(i));let h=null;a!=null&&(h=a.shape,c.push(a));const d=U().getBool("WEBGL_PACK_NORMALIZATION")?new RB(s.shape,o.shape,r.shape,u,h,l):new EB(s.shape,o.shape,r.shape,u,h,l);return t.runWebGLProgram(d,c,c[0].dtype)}};class DB{constructor(t){this.variableNames=["source"],this.outputShape=t,this.rank=t.length;const e=Ot(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const s=FB(this.rank);let o;const r=t.map((i,a)=>`sourceLoc.${lp[a]} = start[${a}] + coords.${lp[a]};`);o=`
        ${e} sourceLoc;
        ${e} coords = getOutputCoords();
        ${r.join(`
`)}
      `,this.userCode=`
      void main() {
        ${o}
        setOutput(getSource(${s}));
      }
    `}}const lp=["x","y","z","w","u","v"];function FB(n){if(n===1)return"sourceLoc";if(n<=6)return lp.slice(0,n).map(t=>"sourceLoc."+t).join(",");throw Error(`Slicing for rank ${n} is not yet supported`)}class OB{constructor(t){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.rank=t.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const e=Ot(this.rank),s=De("coords",this.rank),o=De("sourceLoc",this.rank),r=this.rank===1?"sourceLoc":`vec2(${o.slice(-2).join()})`,i=`getChannel(getSource(${o.join()}), ${r})`,a=`
      result.x = ${i};
      if (++${s[this.rank-1]} < ${t[this.rank-1]}) {
        ++${o[this.rank-1]};
        result.y = ${i};
        --${o[this.rank-1]};
      }
    `,l=this.rank===1?"":`
      --${s[this.rank-1]};
      if (++${s[this.rank-2]} < ${t[this.rank-2]}) {
        ++${o[this.rank-2]};
        result.z = ${i};
        if (++${s[this.rank-1]} < ${t[this.rank-1]}) {
          ++${o[this.rank-1]};
          result.w = ${i};
        }
      }
    `,c=this.rank<=4?`sourceLoc = coords +
            ${e}(${t.map((u,h)=>`start[${h}]`).join()});`:t.map((u,h)=>`${o[h]} = ${s[h]} + start[${h}];`).join(`
`);this.userCode=`
      void main() {
        ${e} coords = getOutputCoords();
        ${e} sourceLoc;
        ${c}
        vec4 result = vec4(0.);
        ${a}
        ${l}
        setOutput(result);
      }
    `}}function _B(n,t,e,s){const o=s.texData.get(n.dataId),r=s.makeTensorInfo(e,n.dtype),i=s.texData.get(r.dataId);Object.assign(i,o),i.refCount=1,i.shape=e,i.dtype=n.dtype;let a=vh(t,lt(n.shape));o.slice&&(a+=o.slice.flatOffset),i.slice={flatOffset:a,origDataId:o.slice&&o.slice.origDataId||n.dataId};const l=s.dataRefCount.get(i.slice.origDataId)||1;return s.dataRefCount.set(i.slice.origDataId,l+1),r}function Jo(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,size:i}=s,[a,l]=$l(o,r,i);if(Ih(o,a,l),q(l)===0)return e.makeTensorInfo(l,o.dtype,[]);if(e.shouldExecuteOnCPU([o])||o.dtype==="string"){const h=e.texData.get(o.dataId),d=TP(h.values,a,l,o.shape,o.dtype);return e.makeTensorInfo(l,o.dtype,d)}const{isPacked:c}=e.texData.get(o.dataId),u=kh(o.shape,a,l);if(c||!u){const h=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new OB(l):new DB(l),d=[a];return e.runWebGLProgram(h,[o],o.dtype,d)}return e.uploadToGPU(o.dataId),_B(o,a,l,e)}const LB={kernelName:Ua,backendName:"webgl",kernelFunc:Jo};const MB={kernelName:sa,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,crops:i}=s;S(o.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const a=r.reduce((b,w)=>b*w),l=fi(o.shape,r,a),c=mi(l.length,r.length),u=gi(o.shape,r,a),h=Rh(i,r.length),d=Ah(u,i,r.length),p=[],f=et({inputs:{x:o},backend:e,attrs:{shape:l}}),m=Fe({inputs:{x:f},backend:e,attrs:{perm:c}}),g=et({inputs:{x:m},backend:e,attrs:{shape:u}}),x=Jo({inputs:{x:g},backend:e,attrs:{begin:h,size:d}});return p.push(f),p.push(m),p.push(g),p.forEach(b=>e.disposeIntermediateTensorInfo(b)),x}};function PB(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i}=s,a=e.readSync(o.dataId),l=e.readSync(r.dataId),c=L1(a,l,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,c)}const BB={kernelName:zc,backendName:"webgl",kernelFunc:PB};const zB=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,VB=`
  return float(int(a.r) & int(b.r));
`;function WB(n){const{inputs:t,backend:e}=n,{a:s,b:o}=t,r=U().getBool("WEBGL_PACK_BINARY_OPERATIONS"),i=U().getNumber("WEBGL_VERSION");if(e.shouldExecuteOnCPU([s,o])||i===1){const l=e.texData.get(s.dataId).values,c=e.texData.get(o.dataId).values,[u,h]=JM(s.shape,o.shape,l,c,s.dtype),d=e.makeTensorInfo(h,s.dtype),p=e.texData.get(d.dataId);return p.values=u,d}let a;return r?a=new Yo(zB,s.shape,o.shape,!1):a=new uo(VB,s.shape,o.shape),e.runWebGLProgram(a,[s,o],s.dtype)}const UB={kernelName:Vc,backendName:"webgl",kernelFunc:WB};function GB(n){const{inputs:t,backend:e}=n,{s0:s,s1:o}=t,r=e.readSync(s.dataId),i=e.readSync(o.dataId),a=mt(Array.from(r),Array.from(i));return e.makeTensorInfo([a.length],"int32",Int32Array.from(a))}const HB={kernelName:vp,backendName:"webgl",kernelFunc:GB};const iy=ye({opSnippet:"return float(a != b);",cpuKernelImpl:yP,dtype:"bool"}),qB={kernelName:Aa,backendName:"webgl",kernelFunc:iy};function Ui(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.texData.get(s.dataId);return Ke({inputs:{x:o.complexTensorInfos.real},backend:e})}const KB={kernelName:fu,backendName:"webgl",kernelFunc:Ui};const jB="return float(int(x));";function XB(n,t){const e=new Wn(n.shape,jB),s=t.runWebGLProgram(e,[n],"int32");return{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}function cp(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return Ke({inputs:{x:o},backend:e});const i=pe(o.shape),a=cp({inputs:{x:o},backend:e,attrs:{dtype:"float32"}}),l=Ss({inputs:{real:a,imag:i},backend:e});return i.dispose(),e.disposeIntermediateTensorInfo(a),l}if(o.dtype==="complex64"){const i=Ui({inputs:{input:o},backend:e}),a=cp({inputs:{x:i},backend:e,attrs:{dtype:r}});return e.disposeIntermediateTensorInfo(i),a}if(!yp(o.dtype,r)){const i=Ke({inputs:{x:o},backend:e});return{dataId:i.dataId,shape:i.shape,dtype:r}}if(e.shouldExecuteOnCPU([o])){const i=e.texData.get(o.dataId).values,[a,l,c]=QM(i,o.shape,o.dtype,r);return e.makeTensorInfo(a,l,c)}if(r==="int32")return XB(o,e);if(r==="bool"){const i=e.makeTensorInfo([],"bool",Ce("bool",1)),l=iy({inputs:{a:o,b:i},backend:e});return e.disposeIntermediateTensorInfo(i),l}throw new Error(`Error in Cast: failed to cast ${o.dtype} to ${r}`)}const YB={kernelName:cr,backendName:"webgl",kernelFunc:cp};const ay="return ceil(x);",ZB=St({opSnippet:ay,packedOpSnippet:ay,cpuKernelImpl:tP}),JB={kernelName:ur,backendName:"webgl",kernelFunc:ZB};class QB{constructor(t){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=t,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}class tz{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=t,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}function ez(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{clipValueMin:r,clipValueMax:i}=s;let a;U().getBool("WEBGL_PACK_CLIP")?a=new tz(o.shape):a=new QB(o.shape);const l=[[r],[i]];return e.runWebGLProgram(a,[o],o.dtype,l)}const nz={kernelName:hr,backendName:"webgl",kernelFunc:ez};class sz{constructor(t){this.variableNames=["real","imag"],this.outputShape=t,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}function ly(n,t){return{dataId:t.dataId,dtype:t.dtype,shape:n.shape}}function oz(n){const{inputs:t,backend:e}=n,{x:s}=t,o=e.texData.get(s.dataId),r=new sz(s.shape),i=[ly(s,o.complexTensorInfos.real),ly(s,o.complexTensorInfos.imag)];return e.runWebGLProgram(r,i,i[0].dtype)}const rz={kernelName:oa,backendName:"webgl",kernelFunc:oz};class iz{constructor(t){this.outputShape=[],this.outputShape=Fn(t,1),this.variableNames=t.map((i,a)=>`T${a}`);const e=new Array(t.length-1);e[0]=t[0][1];for(let i=1;i<e.length;i++)e[i]=e[i-1]+t[i][1];const s=[`if (yC < ${e[0]}) setOutput(getT0(yR, yC));`];for(let i=1;i<e.length;i++){const a=e[i-1];s.push(`else if (yC < ${e[i]}) setOutput(getT${i}(yR, yC-${a}));`)}const o=e.length,r=e[e.length-1];s.push(`else setOutput(getT${o}(yR, yC-${r}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${s.join(`
        `)}
      }
    `}}class az{constructor(t,e){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=Fn(t,e);const s=this.outputShape,o=s.length,r=Ot(o),i=De("coords",o),a=["x","y","z","w","u","v"].slice(0,o);this.variableNames=t.map((m,g)=>`T${g}`);const l=new Array(t.length-1);l[0]=t[0][e];for(let m=1;m<l.length;m++)l[m]=l[m-1]+t[m][e];const c=a[e],u=a.slice(-2),h=a.join();let d=`if (${c} < ${l[0]}) {
        return getChannel(
            getT0(${h}), vec2(${u.join()}));
        }`;for(let m=1;m<l.length;m++){const g=l[m-1];d+=`
        if (${c} < ${l[m]}  && ${c} >= ${l[m-1]}) {
          return getChannel(
            getT${m}(${bc(a,c,g)}),
            vec2(${bc(u,c,g)}));
        }`}const p=l.length,f=l[l.length-1];d+=`
        return getChannel(
          getT${p}(${bc(a,c,f)}),
          vec2(${bc(u,c,f)}));`,this.userCode=`
      float getValue(${a.map(m=>"int "+m)}) {
        ${d}
      }

      void main() {
        ${r} coords = getOutputCoords();
        vec4 result = vec4(getValue(${i}), 0., 0., 0.);

        ${i[o-1]} = ${i[o-1]} + 1;
        if (${i[o-1]} < ${s[o-1]}) {
          result.g = getValue(${i});
        }

        ${i[o-2]} = ${i[o-2]} + 1;
        if (${i[o-2]} < ${s[o-2]}) {
          result.a = getValue(${i});
        }

        ${i[o-1]} = ${i[o-1]} - 1;
        if (${i[o-2]} < ${s[o-2]} &&
            ${i[o-1]} < ${s[o-1]}) {
          result.b = getValue(${i});
        }
        setOutput(result);
      }
    `}}function bc(n,t,e){const s=n.indexOf(t);return n.map((r,i)=>i===s?`${r} - ${e}`:r).join()}function yc(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.texData.get(s.dataId);return Ke({inputs:{x:o.complexTensorInfos.imag},backend:e})}const lz={kernelName:iu,backendName:"webgl",kernelFunc:yc};function Gi(n,t,e){const s=n[0].dtype;if(s==="complex64"){const p=n.map(b=>Ui({inputs:{input:b},backend:e})),f=n.map(b=>yc({inputs:{input:b},backend:e})),m=Gi(p,t,e),g=Gi(f,t,e),x=Ss({inputs:{real:m,imag:g},backend:e});return p.forEach(b=>e.disposeIntermediateTensorInfo(b)),f.forEach(b=>e.disposeIntermediateTensorInfo(b)),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),x}let o=e.shouldExecuteOnCPU(n);if(s==="string"&&(o=!0),o){const p=n.map(y=>{const $=[-1,q(y.shape.slice(t))];return et({inputs:{x:y},backend:e,attrs:{shape:$}})}),f=p.map(y=>({vals:e.readSync(y.dataId),shape:y.shape})),m=Fn(p.map(y=>y.shape),1),g=p[0].shape[0]===1,x=eP(f,m,s,g),b=Fn(n.map(y=>y.shape),t),w=e.makeTensorInfo(b,s,x);return p.forEach(y=>e.disposeIntermediateTensorInfo(y)),w}const r=n.filter(p=>q(p.shape)>0),i=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&r[0].shape.length>1;if(r.length===1){const p=i?new Wn(n[0].shape,ks):new vs(n[0].shape,ks);return e.runWebGLProgram(p,n,s)}const a=U().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(r.length>a){const p=[];for(let m=0;m<r.length;m+=a){const g=r.slice(m,m+a);p.push(Gi(g,t,e))}const f=Gi(p,t,e);for(const m of p)e.disposeIntermediateTensorInfo(m);return f}if(i){const p=new az(r.map(f=>f.shape),t);return e.runWebGLProgram(p,r,s)}const{tensors2D:l,outShape:c}=cz(r,t,e),u=new iz(l.map(p=>p.shape)),h=e.runWebGLProgram(u,l,s);l.forEach(p=>e.disposeIntermediateTensorInfo(p));const d=et({inputs:{x:h},attrs:{shape:c},backend:e});return e.disposeIntermediateTensorInfo(h),d}function cz(n,t,e){const s=Fn(n.map(r=>r.shape),t);return{tensors2D:n.map(r=>et({inputs:{x:r},attrs:{shape:[-1,q(r.shape.slice(t))]},backend:e})),outShape:s}}function cy(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s,r=yt(o,t[0].shape)[0],i=t.map(c=>c.shape);Nh(i,r);const a=Fn(t.map(c=>c.shape),r);if(q(a)===0)return e.makeTensorInfo(a,t[0].dtype,[]);const l=t.filter(c=>q(c.shape)>0);return l.length===1?Ke({inputs:{x:l[0]},backend:e}):Gi(l,r,e)}const uz={kernelName:ra,backendName:"webgl",kernelFunc:cy};class uy{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.outputShape=t.outShape;const i=t.padInfo.top,a=t.padInfo.left,l=t.strideHeight,c=t.strideWidth,u=t.dilationHeight,h=t.dilationWidth,d=t.filterHeight,p=t.filterWidth,f=Math.floor(t.inChannels/4)*4,m=t.inChannels%4,g=t.dataFormat==="channelsLast",x=g?1:2,b=g?2:3,w=g?3:1;let y="",C="";s&&(o?y=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?y=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:y=`
          float activation(float x) {
            ${s}
          }
        `,C="result = activation(result);");const $=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${l}, ${c});
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${w}];

        ivec2 xRCCorner =
            ivec2(coords[${x}], coords[${b}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${t.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${p}; wC++) {
            int xC = xCCorner + wC * ${h};

            if (xC < 0 || xC >= ${t.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${f}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${g}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${m===1}) {

              if (${g}) {
                dotProd +=
                    getX(batch, xR, xC, ${f}) *
                    getW(wR, wC, ${f}, d2);
              } else {
                dotProd +=
                    getX(batch, ${f}, xR, xC) *
                    getW(wR, wC, ${f}, d2);
              }

            } else if (${m===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2)
              );

              if (${g}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${m===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2),
                getW(wR, wC, ${f} + 2, d2)
              );

              if (${g}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1),
                  getX(batch, xR, xC, ${f} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC),
                  getX(batch, ${f} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${$}
        ${C}
        setOutput(result);
      }
    `}}class hz{constructor(t){this.variableNames=["x","W"],this.outputShape=t.outShape;const e=t.padInfo.front,s=t.padInfo.top,o=t.padInfo.left,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=t.dilationDepth,c=t.dilationHeight,u=t.dilationWidth,h=t.filterDepth,d=t.filterHeight,p=t.filterWidth,f=Math.floor(t.inChannels/4)*4,m=t.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${r}, ${i}, ${a});
      const ivec3 pads = ivec3(${e}, ${s}, ${o});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${h}; wF++) {
          int xF = xFCorner + wF * ${l};

          if (xF < 0 || xF >= ${t.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${c};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${p}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${f}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${m===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${f}) *
                  getW(wF, wR, wC, ${f}, d2);
              } else if (${m===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${m===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1),
                  getX(batch, xF, xR, xC, ${f} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2),
                  getW(wF, wR, wC, ${f} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class hy{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Se(this.outputShape.length);const i=t.padInfo.left,a=t.strideWidth,l=t.dilationWidth,c=t.filterHeight,u=t.filterWidth,h=u;let d=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let g=0;g<u;g++)d+=`
           vec4 xTexelC${g*2};
           int xTexelC${g*2}Ready;
           vec4 xTexelC${g*2+1};
           int xTexelC${g*2+1}Ready;
           vec4 xC${g};`;d+=`
     for (int r = 0; r < ${c}; r++) {
      for (int d1 = 0; d1 < ${t.inChannels}; d1 += 2) {
       `;for(let g=0;g<u;g++)d+=`
           xTexelC${g*2} = vec4(0.0);
           xTexelC${g*2}Ready = 0;
           xTexelC${g*2+1} = vec4(0.0);
           xTexelC${g*2+1}Ready = 0;
           xC${g} = vec4(0.0);`;d+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let g=0;g<(h+1)/2;g++){const x=g*2;if(d+=`
           xC = xCCorner + ${x*l};
           `,a===1){if(x<u&&(i%2===1?(d+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }
               `,l===1&&x>0?d+=`
                 xC${x} = vec4(xTexelC${x-2}.zw, xTexelC${x}.xy);
                 `:d+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${x} = vec4(previous.zw, xTexelC${x}.xy);
                   } else {
                     xC${x} = vec4(0.0, 0.0, xTexelC${x}.xy);
                   }
                   `):d+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 xC${x} = xTexelC${x};
                 `,x+1<u)){const b=i%2===0?kc(l):l;l%2===0&&i%2===1||l%2!==0&&i%2!==1?(d+=`
                   xCOffset = xC + imod(pads[1], 2) + ${b};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                     xTexelC${x+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${x+1}.zw = vec2(0.0);
                     }
                     xTexelC${x+1}Ready = 1;
                   }
                   `,l>1?d+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${x+1} = vec4(previous.zw, xTexelC${x+1}.xy);
                     } else {
                      xC${x+1} = vec4(0.0, 0.0, xTexelC${x+1}.xy);
                     }
                     `:d+=`
                     xC${x+1} = vec4(xTexelC${x}.zw, xTexelC${x+1}.xy);
                     `):b===1?d+=`
                     xC${x+1} = xTexelC${x};
                     `:d+=`
                     xCOffset = xC + ${b};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                       xTexelC${x+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${x+1}.zw = vec2(0.0);
                       }
                       xTexelC${x+1}Ready = 1;
                     }

                     xC${x+1} = xTexelC${x+1};
                     `}}else x<u&&(i%2===1?(d+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${x+1}Ready == 0) {
                   xTexelC${x+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${x+1}.zw = vec2(0.0);
                   }
                   xTexelC${x+1}Ready = 1;
                 }

                 xC${x} = vec4(xTexelC${x}.zw, xTexelC${x+1}.zw);
               `,x+1<u&&(d+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${x+1} = vec4(xTexelC${x+1}.xy, final.xy);
                 `)):(d+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                   xTexelC${x+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x+1}.zw = vec2(0.);
                   }
                   xTexelC${x+1}Ready = 1;
                 }

                 xC${x} = vec4(
                   xTexelC${x}.xy, xTexelC${x+1}.xy);
               `,x+1<u&&(d+=`
                   xC${x+1} = vec4(xTexelC${x}.zw, xTexelC${x+1}.zw);
                 `)));x<u&&(d+=`
             wTexel = getW(r, ${x}, d1, d2);
             dotProd += xC${x}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${t.inChannels}) {
               dotProd += xC${x}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,x+1<u&&(d+=`
               wTexel = getW(r, ${x+1}, d1, d2);
               dotProd += xC${x+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${t.inChannels}) {
                 dotProd += xC${x+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}d+=`
     }
   `,d+=`
     }
   `,d+=`
     }
   `;let p="",f="";s&&(o?p=`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${s}
         }`:r?p=`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${s}
         }`:p=`vec4 activation(vec4 x) {
           ${s}
         }`,f="result = activation(result);");const m=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${p}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${d}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${m}
         ${f}
         setOutput(result);
       }
     `}}class dz{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=t,this.enableShapeUniforms=Se(this.outputShape.length);const{dataFormat:s}=e,o=Ae(),r=s==="channelsLast",i=r?1:2,a=r?2:3,l=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${t[2]} && pos < ${t[1]}) {`;let c="";for(let u=0;u<=1;u++)for(let h=0;h<=1;h++)c+=`
          blockIndex = rc.z + ${h};
          pos = rc.y + ${u};

          ${l}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${i}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${a}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${r}) {
                  innerDims = vec2(d1, ch);
                  result[${u*2+h}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${u*2+h}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${c}

        ${o.output} = result;
      }
    `}}function wc(n,t){const e=n.length;return e>=3?t?[...n.slice(0,-3),n[e-3]*n[e-2],n[e-1]]:[...n.slice(0,-3),n[e-3],n[e-2]*n[e-1]]:!t&&e===1&&n[0]>1?[n[0],1]:null}function dy({x:n,filter:t,convInfo:e,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:i=0,activation:a=null}){const l=n.shape,c=s.texData.get(n.dataId),u=e.inChannels,h=l[0]*l[1]*l[2],d=e.outChannels,p=e.dataFormat==="channelsLast",f=!1,m=!1;let g;const x=[];if(r!=null){const y=wc(r.shape,p);y!=null&&(r=et({inputs:{x:r},backend:s,attrs:{shape:y}}),x.push(r))}if(o!=null){const y=wc(o.shape,p);y!=null&&(o=et({inputs:{x:o},backend:s,attrs:{shape:y}}),x.push(o))}if(!((h===1||d===1)&&u>ty)&&c.isPacked&&p&&c.texture!=null&&l[2]%2!==0&&Et(c.shape.slice(-3),l.slice(-3))){const y=l[0]*l[1]*(l[2]+1),C={dataId:n.dataId,shape:[1,y,e.inChannels],dtype:n.dtype},$=c.shape;c.shape=c.shape.slice(),c.shape[c.shape.length-2]++,S(uc(c.shape,C.shape),()=>`packed reshape ${c.shape} to ${C.shape} isn't free`);const N=et({inputs:{x:t},backend:s,attrs:{shape:[1,e.inChannels,e.outChannels]}});x.push(N);const T=gc({a:C,b:N,backend:s,transposeA:f,transposeB:m,bias:o,activation:a,preluActivationWeights:r,leakyreluAlpha:i}),k=s.texData.get(T.dataId);S(k.isPacked,()=>"batchMatMul result is expected to be packed"),c.shape=$,k.shape=e.outShape,g=Ke({inputs:{x:T},backend:s}),g.shape=e.outShape,x.push(T)}else{const y=e.outHeight*e.outWidth,C=et({inputs:{x:n},backend:s,attrs:{shape:p?[e.batchSize,y,e.inChannels]:[e.batchSize,e.inChannels,y]}}),$=et({inputs:{x:t},backend:s,attrs:{shape:[1,e.inChannels,e.outChannels]}}),N=gc({a:p?C:$,b:p?$:C,transposeA:!p,transposeB:m,backend:s,bias:o,activation:a,preluActivationWeights:r,leakyreluAlpha:i});g=et({inputs:{x:N},backend:s,attrs:{shape:e.outShape}}),x.push(C),x.push($),x.push(N)}for(const y of x)s.disposeIntermediateTensorInfo(y);return g}function py({x:n,filter:t,convInfo:e,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:i=0,activation:a=null}){const{filterWidth:l,filterHeight:c,inChannels:u,outWidth:h,outHeight:d,dataFormat:p}=e,f=p==="channelsLast",m=l*c*u,g=d*h,x=[e.batchSize,m,g],b=!0,w=!1,y=[];if(r!=null){const W=wc(r.shape,f);W!=null&&(r=et({inputs:{x:r},backend:s,attrs:{shape:W}}),y.push(r))}if(o!=null){const W=wc(o.shape,f);W!=null&&(o=et({inputs:{x:o},backend:s,attrs:{shape:W}}),y.push(o))}const C=et({inputs:{x:t},backend:s,attrs:{shape:[1,m,q(t.shape)/m]}});y.push(C);const $=new dz(x,e),N=[n.shape,[e.padInfo.top,e.padInfo.left],[e.strideHeight,e.strideWidth],[e.dilationHeight,e.dilationWidth],[e.inChannels],[e.filterWidth*e.inChannels],[e.outWidth]],T=s.runWebGLProgram($,[n],"float32",N),k=et({inputs:{x:T},backend:s,attrs:{shape:x}});y.push(T),y.push(k);const v=o!=null,I=r!=null,R=a==="leakyrelu",A=a?Vi(a,!0):null,P=new X1(f?k.shape:C.shape,f?C.shape:k.shape,f?[e.batchSize,g,e.outChannels]:[e.batchSize,e.outChannels,g],b,w,v,A,I,R),_=f?[k,C]:[C,k];if(o&&_.push(o),I&&_.push(r),R){const W=s.makeTensorInfo([],"float32",rs(i,"float32"));_.push(W),y.push(W)}const B=s.runWebGLProgram(P,_,"float32"),z=et({inputs:{x:B},backend:s,attrs:{shape:e.outShape}});y.push(B);for(const W of y)s.disposeIntermediateTensorInfo(W);return z}function pz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dataFormat:l,dilations:c,dimRoundingMode:u}=s,h=qn(l),d=ge(o.shape,r.shape,i,c,a,u,!1,h);let p;if(d.filterHeight===1&&d.filterWidth===1&&d.dilationHeight===1&&d.dilationWidth===1&&d.strideHeight===1&&d.strideWidth===1&&(d.padInfo.type==="SAME"||d.padInfo.type==="VALID"))p=dy({x:o,filter:r,convInfo:d,backend:e});else if(d.strideWidth<=2&&h==="channelsLast"&&U().getBool("WEBGL_EXP_CONV")){const m=new hy(d),g=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];p=e.runWebGLProgram(m,[o,r],"float32",g)}else if(U().getBool("WEBGL_CONV_IM2COL"))p=py({x:o,filter:r,convInfo:d,backend:e});else{const m=new uy(d);p=e.runWebGLProgram(m,[o,r],"float32")}const f=et({inputs:{x:p},backend:e,attrs:{shape:d.outShape}});return e.disposeIntermediateTensorInfo(p),f}const fz={kernelName:ia,backendName:"webgl",kernelFunc:pz};class mz{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,s=t.strideWidth,o=t.padInfo.top,r=t.padInfo.left,i=t.dataFormat==="channelsLast";this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yR = 0; yR < ${t.outHeight}; yR++) {
            int xR = wR + yR * ${e} - ${o};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${t.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              ${i?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class gz{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=t.dataFormat==="channelsLast",a=e-1-t.padInfo.top,l=s-1-t.padInfo.left,c=i?1:2,u=i?2:3,h=i?3:1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${l});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${h}];

        ivec2 dyCorner = ivec2(coords[${c}], coords[${u}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            for (int d2 = 0; d2 < ${t.outChannels}; d2++) {

              if (${i}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class xz{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideDepth,s=t.strideHeight,o=t.strideWidth,r=t.padInfo.front,i=t.padInfo.top,a=t.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yF = 0; yF < ${t.outDepth}; yF++) {
            int xF = wF + yF * ${e} - ${r};

            if (xF < 0 || xF >= ${t.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${t.outHeight}; yR++) {
              int xR = wR + yR * ${s} - ${i};

              if (xR < 0 || xR >= ${t.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${t.outWidth}; yC++) {
                int xC = wC + yC * ${o} - ${a};

                if (xC < 0 || xC >= ${t.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class bz{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterDepth,s=t.filterHeight,o=t.filterWidth,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=e-1-t.padInfo.front,c=s-1-t.padInfo.top,u=o-1-t.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${l}, ${c}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${e}; wF++) {
          float dyF = float(dyFCorner + wF) / ${r}.0;

          if (dyF < 0.0 || dyF >= ${t.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${e} - 1 - wF;

          for (int wR = 0; wR < ${s}; wR++) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${s} - 1 - wR;

            for (int wC = 0; wC < ${o}; wC++) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${o} - 1 - wC;

              for (int d2 = 0; d2 < ${t.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function yz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,dataFormat:l,dimRoundingMode:c,filterShape:u}=s,h=qn(l),d=ge(o.shape,u,i,1,a,c,!1,h),p=new mz(d);return e.runWebGLProgram(p,[o,r],"float32")}const wz={kernelName:Uc,backendName:"webgl",kernelFunc:yz};class Cz{constructor(t){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=t.inShape,this.enableShapeUniforms=Se(this.outputShape.length);const e=t.filterHeight,s=t.filterWidth,o=e-1-t.padInfo.top,r=s-1-t.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${r});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            int wCPerm = ${s} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${t.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${t.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}function Iz(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{inputShape:i,strides:a,pad:l,dataFormat:c,dimRoundingMode:u}=s,h=qn(c),d=ge(i,r.shape,a,1,l,u,!1,h);if(U().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&h==="channelsLast"){const p=[[d.strideHeight,d.strideWidth]],f=new Cz(d);return e.runWebGLProgram(f,[o,r],"float32",p)}else{const p=new gz(d);return e.runWebGLProgram(p,[o,r],"float32")}}const $z={kernelName:aa,backendName:"webgl",kernelFunc:Iz};function kz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s,c=cs(o.shape,r.shape,i,l,a),u=new hz(c);return e.runWebGLProgram(u,[o,r],"float32")}const vz={kernelName:la,backendName:"webgl",kernelFunc:kz};function Sz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,filterShape:l}=s,c=cs(o.shape,l,i,1,a),u=new xz(c);return e.runWebGLProgram(u,[o,r],"float32")}const Nz={kernelName:Gc,backendName:"webgl",kernelFunc:Sz};function Tz(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{pad:i,strides:a,inputShape:l}=s,c=cs(l,r.shape,a,1,i),u=new bz(c);return e.runWebGLProgram(u,[o,r],"float32")}const Ez={kernelName:Hc,backendName:"webgl",kernelFunc:Tz};const Rz=Zo+`
  return cos(x);
`,Az=`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${ho}
  return result;
`,Dz=St({opSnippet:Rz,packedOpSnippet:Az}),Fz={kernelName:dr,backendName:"webgl",kernelFunc:Dz};const Oz=St({opSnippet:`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`}),_z={kernelName:pr,backendName:"webgl",kernelFunc:Oz};class Lz{constructor(t,e,s,o,r){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[i,a,l,c]=t,[u]=e,[h,d]=s;this.outputShape=[u,h,d,c];const p=o==="bilinear"?1:0,[f,m]=[`${a-1}.0`,`${l-1}.0`],[g,x,b]=h>1?[`${(a-1)/(h-1)}`,"(y2-y1) * height_ratio",`y1*${f} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${f}`],[w,y,C]=d>1?[`${(l-1)/(d-1)}`,"(x2-x1) * width_ratio",`x1*${m} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${m}`];this.userCode=`
      const float height_ratio = float(${g});
      const float width_ratio = float(${w});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${i}) {
          return;
        }

        float height_scale = ${x};
        float width_scale = ${y};

        float in_y = ${b};
        if( in_y < 0.0 || in_y > ${f} ) {
          setOutput(float(${r}));
          return;
        }
        float in_x = ${C};
        if( in_x < 0.0 || in_x > ${m} ) {
          setOutput(float(${r}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${p} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}const Mz={kernelName:Kc,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{image:o,boxes:r,boxInd:i}=t,{cropSize:a,method:l,extrapolationValue:c}=s,u=new Lz(o.shape,r.shape,a,l,c);return e.runWebGLProgram(u,[o,r,i],"float32")}};var Hi;(function(n){n.Prod="*",n.Sum="+"})(Hi||(Hi={}));class fy{constructor(t,e,s,o){this.op=t,this.outputShape=e,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const r=this.outputShape.length,i=this.op===Hi.Prod?"1.0":"0.0",a=s?i:`getX(${my(r,"coords",this.op)})`,l=this.outputShape[this.outputShape.length-1];let c="",u="";s?(c=o?`end != ${l-1}`:"end != 0",u=o?"end + 1":"end - 1"):(c=o?`end + pow2 < ${l}`:"end >= pow2",u=o?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${Ot(r)} coords = getOutputCoords();
        int end = ${gy(r,"coords",this.op)};
        float val = ${a};
        int pow2 = int(pow(2.0, index));
        if (${c}) {
          int idx = ${u};
          ${gy(r,"coords",this.op)} = idx;
          val ${this.op}= getX(${my(r,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function my(n,t,e){if(n===1)return`${t}`;if(n===2)return`${t}.x, ${t}.y`;if(n===3)return`${t}.x, ${t}.y, ${t}.z`;if(n===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw new Error(`Cumulative ${e} for rank ${n} is not yet supported`)}function gy(n,t,e){if(n===1)return`${t}`;if(n===2)return`${t}.y`;if(n===3)return`${t}.z`;if(n===4)return`${t}.w`;throw new Error(`Cumulative ${e} for rank ${n} is not yet supported`)}function xy(n,t,e,s,o,r){const i=t.shape.length,a=qt([s],i);let l=t;a!=null&&(l=Fe({inputs:{x:t},backend:e,attrs:{perm:a}}));const c=Jt(1,i)[0];if(c!==i-1)throw new Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${s}`);const u=l.shape[c];let h=Ke({inputs:{x:l},backend:e});for(let d=0;d<=Math.ceil(Math.log2(u))-1;d++){const p=new fy(n,l.shape,!1,r),f=[[d]],m=h;h=e.runWebGLProgram(p,[h],h.dtype,f),e.disposeIntermediateTensorInfo(m)}if(o){const d=new fy(n,l.shape,o,r),p=h;h=e.runWebGLProgram(d,[h],h.dtype),e.disposeIntermediateTensorInfo(p)}if(a!=null){const d=us(a),p=Fe({inputs:{x:h},backend:e,attrs:{perm:d}});return e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(l),p}return h}function Pz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;return xy(Hi.Prod,o,e,r,i,a)}const Bz={kernelName:qc,backendName:"webgl",kernelFunc:Pz};function zz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;return xy(Hi.Sum,o,e,r,i,a)}const Vz={kernelName:ca,backendName:"webgl",kernelFunc:zz};function Wz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i,binaryOutput:a}=s;if(o.shape.length===1){const l=e.readSync(o.dataId),c=e.readSync(r.dataId),u=L1(l,c,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,u)}else if(o.shape.length===2){const l=e.bufferSync(o),c=e.bufferSync(r),u=ZM(l,c,i,a);return e.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const Uz={kernelName:jc,backendName:"webgl",kernelFunc:Wz};class Gz{constructor(t,e,s){this.variableNames=["x"],this.outputShape=[],this.outputShape=t,this.blockSize=e,this.dataFormat=s,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${e};
      int offset_h = imod(h, ${e});
      int in_w = w / ${e};
      int offset_w = imod(w, ${e});
      int offset_d = (offset_h * ${e} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat==="NHWC"?"coords[1]":"coords[2]"}getWidthCoordString(){return this.dataFormat==="NHWC"?"coords[2]":"coords[3]"}getDepthCoordString(){return this.dataFormat==="NHWC"?"coords[3]":"coords[1]"}getOutputDepthSize(){return this.dataFormat==="NHWC"?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat==="NHWC"?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}function Hz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockSize:r,dataFormat:i}=s,a=o.shape[0],l=i==="NHWC"?o.shape[1]:o.shape[2],c=i==="NHWC"?o.shape[2]:o.shape[3],u=i==="NHWC"?o.shape[3]:o.shape[1],h=l*r,d=c*r,p=u/(r*r),f=i==="NHWC"?[a,h,d,p]:[a,p,h,d],m=new Gz(f,r,i);return e.runWebGLProgram(m,[o],o.dtype)}const qz={kernelName:Xc,backendName:"webgl",kernelFunc:Hz};class by{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Se(this.outputShape.length);const i=t.filterHeight,a=t.filterWidth,l=t.outChannels/t.inChannels;let c="",u="";s&&(o?c=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?c=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:c=`
          float activation(float x) {
            ${s}
          }
        `,u="result = activation(result);");const h=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${c}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${l};
        int q = d2 - d1 * ${l};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${i}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${a}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${h}
        ${u}
        setOutput(result);
      }
    `}}class yy{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Se(this.outputShape.length);const i=t.outChannels/t.inChannels,a=t.padInfo.left,l=t.strideWidth,c=t.dilationWidth,u=t.filterHeight,h=t.filterWidth,d=h;let p=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let x=0;x<h;x++)p+=`
          vec4 xTexelC${x*2};
          int xTexelC${x*2}Ready;
          vec4 xTexelC${x*2+1};
          int xTexelC${x*2+1}Ready;
          vec4 xC${x};`;p+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let x=0;x<h;x++)p+=`
          xTexelC${x*2} = vec4(0.0);
          xTexelC${x*2}Ready = 0;
          xTexelC${x*2+1} = vec4(0.0);
          xTexelC${x*2+1}Ready = 0;
          xC${x} = vec4(0.0);`;p+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let x=0;x<(d+1)/2;x++){const b=x*2;if(p+=`
          xC = xCCorner + ${b*c};
          `,l===1){if(b<h&&(a%2===1?(p+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }
              `,c===1&&b>0?p+=`
                xC${b} = vec4(xTexelC${b-2}.zw, xTexelC${b}.xy);
                `:p+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${b} = vec4(previous.zw, xTexelC${b}.xy);
                  } else {
                    xC${b} = vec4(0.0, 0.0, xTexelC${b}.xy);
                  }
                  `):p+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xC${b} = xTexelC${b};
                `,b+1<h)){const w=a%2===0?kc(c):c;c%2===0&&a%2===1||c%2!==0&&a%2!==1?(p+=`
                  xCOffset = xC + imod(pads[1], 2) + ${w};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                    xTexelC${b+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${b+1}.zw = vec2(0.0);
                    }
                    xTexelC${b+1}Ready = 1;
                  }
                  `,c>1?p+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${b+1} = vec4(previous.zw, xTexelC${b+1}.xy);
                    } else {
                     xC${b+1} = vec4(0.0, 0.0, xTexelC${b+1}.xy);
                    }
                    `:p+=`
                    xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.xy);
                    `):w===1?p+=`
                    xC${b+1} = xTexelC${b};
                    `:p+=`
                    xCOffset = xC + ${w};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                      xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${b+1}.zw = vec2(0.0);
                      }
                      xTexelC${b+1}Ready = 1;
                    }

                    xC${b+1} = xTexelC${b+1};
                    `}}else b<h&&(a%2===1?(p+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.0);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
              `,b+1<h&&(p+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${b+1} = vec4(xTexelC${b+1}.xy, final.xy);
                `)):(p+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(
                  xTexelC${b}.xy, xTexelC${b+1}.xy);
              `,b+1<h&&(p+=`
                  xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
                `)));b<h&&(p+=`
            wTexel = getW(r, ${b}, d1, q);
            dotProd += xC${b} * vec4(wTexel.xz, wTexel.xz);
          `,b+1<h&&(p+=`
              wTexel = getW(r, ${b+1}, d1, q);
              dotProd += xC${b+1} * vec4(wTexel.xz, wTexel.xz);
            `))}p+=`
    }
  `,p+=`
      }
    `;let f="",m="";s&&(o?f=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?f=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:f=`vec4 activation(vec4 x) {
          ${s}
        }`,m="result = activation(result);");const g=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${f}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${i};
        int q = d2 - d1 * ${i};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${p}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${g}
        ${m}
        setOutput(result);
      }
    `}}function Kz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l,dimRoundingMode:c}=s;let u=l;u==null&&(u=[1,1]),S($e(i,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${u}'`);const h=ge(o.shape,r.shape,i,u,a,c,!0);let d;U().getBool("WEBGL_PACK_DEPTHWISECONV")&&h.strideWidth<=2&&h.outChannels/h.inChannels===1?d=new yy(h):d=new by(h);const p=[[h.padInfo.top,h.padInfo.left],[h.strideHeight,h.strideWidth],[h.dilationHeight,h.dilationWidth],[h.inHeight,h.inWidth]];return e.runWebGLProgram(d,[o,r],"float32",p)}const jz={kernelName:ua,backendName:"webgl",kernelFunc:Kz};class Xz{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,s=t.strideWidth,o=t.padInfo.top,r=t.padInfo.left,i=t.outChannels/t.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${i} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yR = 0; yR < ${t.outHeight}; yR++) {
            int xR = wR + yR * ${e} - ${o};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${t.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Yz{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=e-1-t.padInfo.top,a=s-1-t.padInfo.left,l=t.outChannels/t.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${l}; dm++) {
              int d2 = d1 * ${l} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function Zz(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,filterShape:u}=s,h=ge(o.shape,u,i,a,l,c,!0),d=new Xz(h);return e.runWebGLProgram(d,[o,r],"float32")}const Jz={kernelName:Yc,backendName:"webgl",kernelFunc:Zz};function Qz(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,inputShape:u}=s,h=ge(u,r.shape,i,a,l,c,!0),d=new Yz(h);return e.runWebGLProgram(d,[o,r],"float32")}const tV={kernelName:Zc,backendName:"webgl",kernelFunc:Qz};class eV{constructor(t){this.variableNames=["X"],this.outputShape=[t,t],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}function nV(n){const{inputs:t,backend:e}=n,{x:s}=t,o=[...s.shape,...s.shape],r=q(s.shape),i=et({inputs:{x:s},backend:e,attrs:{shape:[r]}}),a=new eV(r),l=e.runWebGLProgram(a,[i],i.dtype),c=et({inputs:{x:l},backend:e,attrs:{shape:o}});return e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(l),c}const sV={kernelName:Sp,backendName:"webgl",kernelFunc:nV};class oV{constructor(t){this.variableNames=["x","W"],this.outputShape=t.outShape;const{inHeight:e,inWidth:s,padInfo:o,strideHeight:r,strideWidth:i,filterHeight:a,filterWidth:l,dilationHeight:c,dilationWidth:u}=t,{top:h,left:d}=o;this.userCode=`
      const ivec2 strides = ivec2(${r}, ${i});
      const ivec2 pads = ivec2(${h}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${a}; h++) {
          int hIn = hBeg + h * ${c};

          if (hIn >= 0 && hIn < ${e}) {
            for (int w = 0; w < ${l}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${s}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}function rV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s,c=si(o.shape,r.shape,i,a,"NHWC",l);let u;const h=new oV(c);u=e.runWebGLProgram(h,[o,r],"float32");const d=et({inputs:{x:u},backend:e,attrs:{shape:c.outShape}});return e.disposeIntermediateTensorInfo(u),d}const iV={kernelName:ha,backendName:"webgl",kernelFunc:rV};function aV(n){const{inputs:t,backend:e,attrs:s}=n,{equation:o}=s,r=t,{allDims:i,summedDims:a,idDims:l}=zh(o,r.length);Wh(i.length,l,r);const{path:c,steps:u}=Uh(a,l),h=u.length;let d=null,p=i.length;const f=[];for(let m=0;m<h;++m){for(const g of u[m]){const{permutationIndices:x,expandDims:b}=Vh(p,l[g]);let w;Gh(x)?w=r[g]:(w=Fe({inputs:{x:r[g]},backend:e,attrs:{perm:x}}),f.push(w));const y=w.shape.slice();for(let C=0;C<b.length;++C)y.splice(b[C],0,1);Et(w.shape,y)||(w=et({inputs:{x:w},backend:e,attrs:{shape:y}}),f.push(w)),d===null?d=w:(d=ip({inputs:{a:w,b:d},backend:e}),f.push(d))}m<h-1&&(c[m]>=0&&(d=mc({inputs:{x:d},backend:e,attrs:{axis:c[m]-(i.length-p),keepDims:!1}}),f.push(d)),p--)}for(const m of f)m!==d&&e.disposeIntermediateTensorInfo(m);return d}const lV={kernelName:tu,backendName:"webgl",kernelFunc:aV};const cV=St({opSnippet:"return (x >= 0.0) ? x : (exp(x) - 1.0);",packedOpSnippet:`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`}),uV={kernelName:mr,backendName:"webgl",kernelFunc:cV};const hV="return (b >= 0.0) ? a : a * (b + 1.0);",dV=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,pV={kernelName:eu,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e}=n,{dy:s,y:o}=t,r=U().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new Yo(dV,s.shape,o.shape):new uo(hV,s.shape,o.shape);return e.runWebGLProgram(r,[s,o],s.dtype)}};const fV=ye({opSnippet:"return float(a == b);",packedOpSnippet:`
  return vec4(equal(a, b));
`,dtype:"bool",cpuKernelImpl:nP}),mV={kernelName:da,backendName:"webgl",kernelFunc:fV};const gV=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Dh};
  float a1 = ${Fh};
  float a2 = ${Oh};
  float a3 = ${_h};
  float a4 = ${Lh};
  float a5 = ${Mh};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`,xV=St({opSnippet:gV}),bV={kernelName:gr,backendName:"webgl",kernelFunc:xV};const yV=Zo+`
  return exp(x);
`,wy=St({opSnippet:yV,packedOpSnippet:`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:sP,dtype:"float32"}),wV={kernelName:xr,backendName:"webgl",kernelFunc:wy};function up(n){const{inputs:t,attrs:e,backend:s}=n,{dim:o}=e,{input:r}=t,i=r.shape.length,a=r.shape.slice();let l=o;return o<0&&(S(-(i+1)<=o,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+o+1),a.splice(l,0,1),et({inputs:{x:r},backend:s,attrs:{shape:a}})}const CV={kernelName:pa,backendName:"webgl",kernelFunc:up};const Cy="return exp(x) - 1.0;",IV=St({opSnippet:Cy,packedOpSnippet:Cy,cpuKernelImpl:oP}),$V={kernelName:br,backendName:"webgl",kernelFunc:IV};class Iy{constructor(t,e,s){this.variableNames=["real","imag"];const o=e[1];this.outputShape=e;const r=s?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,i=s?`${o}.0`:"1.0";let a;if(t==="real")a="return real * expR - imag * expI;";else if(t==="imag")a="return real * expI + imag * expR;";else throw new Error(`FFT component must be either "real" or "imag", got ${t}.`);this.userCode=`
      const float exponentMultiplier = ${r};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${a}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${o});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${o}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${i};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}function $y(n,t,e){const s=e.texData.get(n.dataId),o=q(n.shape),r=n.shape[n.shape.length-1],i=o/r,a=et({inputs:{x:n},backend:e,attrs:{shape:[i,r]}}),l=a.shape,c=new Iy("real",l,t),u=new Iy("imag",l,t),h=[{dataId:s.complexTensorInfos.real.dataId,dtype:s.complexTensorInfos.real.dtype,shape:l},{dataId:s.complexTensorInfos.imag.dataId,dtype:s.complexTensorInfos.imag.dtype,shape:l}],d=e.runWebGLProgram(c,h,"float32"),p=e.runWebGLProgram(u,h,"float32"),f=Ss({inputs:{real:d,imag:p},backend:e});e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p);const m=et({inputs:{x:f},backend:e,attrs:{shape:n.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(f),m}function kV(n){const{inputs:t,backend:e}=n,{input:s}=t;return $y(s,!1,e)}const vV={kernelName:nu,backendName:"webgl",kernelFunc:kV};class SV{constructor(t,e){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=t,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}function qi(n){const{backend:t,attrs:e}=n,{shape:s,value:o}=e;let{dtype:r}=e;if(r=r||yo(o),r==="string"){const i=Yt(r,q(s));return i.fill(o),t.makeTensorInfo(s,r,i)}else{const i=new SV(s,o),a=[[o]];return t.runWebGLProgram(i,[],r,a)}}const NV={kernelName:su,backendName:"webgl",kernelFunc:qi};class TV{constructor(t){this.variableNames=["Image"],this.outputShape=[];const e=t[2];this.outputShape=t,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${e} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${e}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const EV={kernelName:ou,backendName:"webgl",kernelFunc:({inputs:n,backend:t})=>{const{image:e}=n,s=t,o=new TV(e.shape);return s.runWebGLProgram(o,[e],e.dtype)}};const ky="return floor(x);",RV=St({opSnippet:ky,packedOpSnippet:ky,cpuKernelImpl:rP}),AV={kernelName:yr,backendName:"webgl",kernelFunc:RV};const DV=ye({opSnippet:`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,packedOpSnippet:`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,dtype:"int32"}),FV={kernelName:wr,backendName:"webgl",kernelFunc:DV};class OV{constructor(t){this.variableNames=["A"];const e=Ae(),[s,o]=t;this.outputShape=t,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${o}.0, ${s}.0);

        vec4 values = ${e.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}class _V{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const e=Ae(),[s,o]=t;this.outputShape=t,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${o}.0, ${s}.0);
            vec4 values = ${e.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${e.output} = result;
      }
    `}}const LV={kernelName:aw,backendName:"webgl",kernelFunc:MV};let Qo,hp=U().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");function MV(n){const{inputs:t,backend:e,attrs:s}=n;let{pixels:o}=t;const{numChannels:r}=s,i=typeof HTMLVideoElement<"u"&&o instanceof HTMLVideoElement,a=typeof HTMLImageElement<"u"&&o instanceof HTMLImageElement,[l,c]=i?[o.videoWidth,o.videoHeight]:[o.width,o.height],u=[c,l],h=[c,l,r];if(a||i){const m=U().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(Qo==null||m!==hp)&&(hp=m,Qo=document.createElement("canvas").getContext("2d",{willReadFrequently:hp})),Qo.canvas.width=l,Qo.canvas.height=c,Qo.drawImage(o,0,0,l,c),o=Qo.canvas}const d=e.makeTensorInfo(u,"int32");e.texData.get(d.dataId).usage=Qe.PIXELS,e.gpgpu.uploadPixelDataToTexture(e.getTexture(d.dataId),o);const p=U().getBool("WEBGL_PACK")?new _V(h):new OV(h),f=e.runWebGLProgram(p,[d],"int32");return e.disposeData(d.dataId),f}function PV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s,m=qn(u),g=ge(o.shape,r.shape,l,h,c,d,!1,m);let x;const b=[],w=i!=null,y=a!=null,C=p==="leakyrelu",$=()=>{const T=[o,r],k=(v,I)=>{if(I==="NCHW"&&v.shape.length===1&&v.shape[0]!==1){const R=et({inputs:{x:v},backend:e,attrs:{shape:[v.shape[0],1,1]}});return b.push(R),R}return v};if(w&&T.push(k(i,u)),y&&T.push(k(a,u)),C){const v=e.makeTensorInfo([],"float32",rs(f,"float32"));T.push(v),b.push(v)}return T};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type==="SAME"||g.padInfo.type==="VALID"))x=dy({x:o,filter:r,convInfo:g,backend:e,bias:i,activation:p,preluActivationWeights:a,leakyreluAlpha:f});else if(g.strideWidth<=2&&m==="channelsLast"&&U().getBool("WEBGL_EXP_CONV")){const T=p?Vi(p,!0):null,k=new hy(g,w,T,y,C),v=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],I=$();x=e.runWebGLProgram(k,I,"float32",v)}else if(U().getBool("WEBGL_CONV_IM2COL"))x=py({x:o,filter:r,convInfo:g,backend:e,bias:i,activation:p,preluActivationWeights:a,leakyreluAlpha:f});else{const T=p?Vi(p,!1):null,k=new uy(g,w,T,y,C),v=$();x=e.runWebGLProgram(k,v,"float32")}const N=et({inputs:{x},backend:e,attrs:{shape:g.outShape}});return b.push(x),b.forEach(T=>e.disposeIntermediateTensorInfo(T)),N}const BV={kernelName:Ja,backendName:"webgl",kernelFunc:PV};function zV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dilations:u,dimRoundingMode:h,activation:d,leakyreluAlpha:p}=s,f=[];let m=u;m==null&&(m=[1,1]),S($e(l,m),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${l} and dilations '${m}'`);const g=ge(o.shape,r.shape,l,m,c,h,!0),x=U().getBool("WEBGL_PACK_DEPTHWISECONV")&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,b=d?Vi(d,x):null,w=[o,r],y=i!=null,C=a!=null,$=d==="leakyrelu";if(y&&w.push(i),C&&w.push(a),$){const v=e.makeTensorInfo([],"float32",rs(p,"float32"));w.push(v),f.push(v)}let N;x?N=new yy(g,y,b,C,$):N=new by(g,y,b,C,$);const T=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],k=e.runWebGLProgram(N,w,"float32",T);return f.forEach(v=>e.disposeIntermediateTensorInfo(v)),k}const VV={kernelName:Hp,backendName:"webgl",kernelFunc:zV};class WV{constructor(t,e,s,o){this.sliceDim=t,this.strides=e,this.paramsShape=o,this.variableNames=["x","indices"],this.outputShape=s;const r=Ot(s.length);let i=`
    int index;`;for(let a=0;a<this.sliceDim;a++)i+=`
          index = round(getIndices(coords[0], ${a}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[a]};
          flattenIndex += index * ${this.strides[a]};`;this.userCode=`
         void main() {
          ${r} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${i}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}function UV(n){const{inputs:t,backend:e}=n,{params:s,indices:o}=t,r=o.shape,i=r[r.length-1],a=q(s.shape),[l,c,u,h]=wh(s,o),d=et({inputs:{x:o},backend:e,attrs:{shape:[c,i]}}),p=et({inputs:{x:s},backend:e,attrs:{shape:[q(s.shape)/u,u]}});if(e.shouldExecuteOnCPU([s,o])||s.dtype==="string"){const x=e.readSync(o.dataId),b=e.bufferSync(s),w=iP(x,b,s.dtype,c,i,u,h,s.shape,a);return e.makeTensorInfo(l,s.dtype,w.values)}const f=new WV(i,h,[c,u],s.shape),m=e.runWebGLProgram(f,[p,d],p.dtype),g=et({inputs:{x:m},backend:e,attrs:{shape:l}});return e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(m),g}const GV={kernelName:Np,backendName:"webgl",kernelFunc:UV};class HV{constructor(t,e){this.variableNames=["A","indices"],this.outputShape=e,this.rank=e.length;const s=Ot(this.rank),o=qV(t);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${t[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${o}));
      }
    `}}function qV(n,t){const e=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[];for(let o=0;o<n.length;o++)o===2?s.push("index"):s.push(`${e[o]}`);return s.join()}function vy(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,indices:r}=t,{axis:i,batchDims:a}=s,l=yt(i,o.shape)[0];if(U().get("DEBUG")){const b=e.readSync(r.dataId),w=o.shape[l];for(let y=0;y<b.length;++y){const C=b[y];S(C<=w-1&&C>=0,()=>`GatherV2: the index value ${C} is not in [0, ${w-1}]`)}}const c=Kh(o,r,l,a),u=q(r.shape),h=[],d=et({inputs:{x:o},backend:e,attrs:{shape:[c.batchSize,c.outerSize,c.dimSize,c.sliceSize]}}),p=et({inputs:{x:r},backend:e,attrs:{shape:[c.batchSize,u/c.batchSize]}});h.push(d),h.push(p);const f=[c.batchSize,c.outerSize,u/c.batchSize,c.sliceSize];if(e.shouldExecuteOnCPU([o,r])||o.dtype==="string"){const b=e.bufferSync(p),w=e.bufferSync(d),y=aP(w,b,f);return h.forEach(C=>e.disposeIntermediateTensorInfo(C)),e.makeTensorInfo(c.outputShape,y.dtype,y.values)}const m=new HV(d.shape,f),g=e.runWebGLProgram(m,[d,p],d.dtype);h.push(g);const x=et({inputs:{x:g},backend:e,attrs:{shape:c.outputShape}});return h.forEach(b=>e.disposeIntermediateTensorInfo(b)),x}const KV={kernelName:ma,backendName:"webgl",kernelFunc:vy};const jV=ye({opSnippet:"return float(a > b);",packedOpSnippet:`
  return vec4(greaterThan(a, b));
`,cpuKernelImpl:lP,dtype:"bool"}),XV={kernelName:ga,backendName:"webgl",kernelFunc:jV};const YV=ye({opSnippet:"return float(a >= b);",packedOpSnippet:`
  return vec4(greaterThanEqual(a, b));
`,dtype:"bool",cpuKernelImpl:cP}),ZV={kernelName:Cr,backendName:"webgl",kernelFunc:YV};function JV(n){const{inputs:t,backend:e}=n,{input:s}=t;return $y(s,!0,e)}const QV={kernelName:ru,backendName:"webgl",kernelFunc:JV};const tW=St({opSnippet:"return float(!isnan(x) && !isinf(x));",dtype:"bool"}),eW={kernelName:$r,backendName:"webgl",kernelFunc:tW};const nW=St({opSnippet:"return float(isinf(x));",dtype:"bool"}),sW={kernelName:kr,backendName:"webgl",kernelFunc:nW};const oW=St({opSnippet:"return float(isnan(x));",dtype:"bool"}),rW={kernelName:vr,backendName:"webgl",kernelFunc:oW};const iW=ye({opSnippet:"return float(a < b);",packedOpSnippet:`
  return vec4(lessThan(a, b));
`,cpuKernelImpl:uP,dtype:"bool"}),aW={kernelName:ba,backendName:"webgl",kernelFunc:iW};const lW=ye({opSnippet:"return float(a <= b);",packedOpSnippet:`
  return vec4(lessThanEqual(a, b));
`,cpuKernelImpl:hP,dtype:"bool"}),cW={kernelName:ya,backendName:"webgl",kernelFunc:lW};function uW(n){const{backend:t,attrs:e}=n,{start:s,stop:o,num:r}=e,i=dP(s,o,r);return t.makeTensorInfo([i.length],"float32",i)}const hW={kernelName:Tp,backendName:"webgl",kernelFunc:uW};const dW=Zo+`
  return x < 0.0 ? 0./0. : log(x);
`,pW=St({opSnippet:dW,packedOpSnippet:`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,cpuKernelImpl:pP}),fW={kernelName:Sr,backendName:"webgl",kernelFunc:pW};const mW=Zo+`
  return log(1.0 + x);
`,gW=St({opSnippet:mW}),xW={kernelName:Nr,backendName:"webgl",kernelFunc:gW};const bW=ye({opSnippet:"return float(a >= 1.0 && b >= 1.0);",packedOpSnippet:`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,dtype:"bool"}),yW={kernelName:wa,backendName:"webgl",kernelFunc:bW};const wW=St({opSnippet:"return float(!(x >= 1.0));"}),CW={kernelName:Ca,backendName:"webgl",kernelFunc:wW};const IW=ye({opSnippet:"return float(a >= 1.0 || b >= 1.0);",packedOpSnippet:`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,dtype:"bool"}),$W={kernelName:Ia,backendName:"webgl",kernelFunc:IW};class kW{constructor(t,e,s,o,r){this.variableNames=["x"],this.outputShape=[];const i=e,a=t[3]-1;this.outputShape=t;let l;const c=`float(${s}) + float(${o}) * sum`;r===.5?l=`inversesqrt(${c})`:r===1?l=`1.0/(${c})`:l=`exp(log(${c}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${i}; j <= ${i}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${a}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${l};
        setOutput(val);
      }
    `}}class vW{constructor(t,e,s,o,r){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const i=e,a=t[3]-1;this.outputShape=t;let l;const c=`float(${s}) + float(${o}) * sum`;r===.5?l=`inversesqrt(${c})`:r===1?l=`1.0/(${c})`:l=`exp(log(${c}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${i};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${i}; j <= ${i}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${a}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${l};
        setOutput(result);
      }
    `}}const SW={kernelName:$a,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{depthRadius:r,bias:i,alpha:a,beta:l}=s,c=U().getBool("WEBGL_PACK_NORMALIZATION")?new vW(o.shape,r,i,a,l):new kW(o.shape,r,i,a,l);return e.runWebGLProgram(c,[o],o.dtype)}};class NW{constructor(t,e,s,o,r){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=t,this.depth=t[3],this.depthRadius=e,this.bias=s,this.alpha=o,this.beta=r,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${e})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${e} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${o}) * norm + float(${s});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${o})
                * float(${r})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${r});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}const TW={kernelName:au,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o,y:r,dy:i}=t,{depthRadius:a,bias:l,alpha:c,beta:u}=s,h=new NW(o.shape,a,l,c,u);return e.runWebGLProgram(h,[o,r,i],o.dtype)}};function EW(n,t,e,s){const o=q(t),i=q(n.shape)/o,a=et({inputs:{x:n},attrs:{shape:[i,o]},backend:s}),l=po(a,n.dtype,"max",s),c=et({inputs:{x:l},attrs:{shape:e},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(l),c}function Sy(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reductionIndices:r,keepDims:i}=s,a=o.shape.length,l=yt(r,o.shape);let c=l;const u=qt(c,a),h=u!=null,d=e.shouldExecuteOnCPU([o]);let p=o;if(h){if(d){const w=e.texData.get(p.dataId).values,y=new Array(a);for(let N=0;N<y.length;N++)y[N]=o.shape[u[N]];const C=op(w,o.shape,o.dtype,u,y);p=e.makeTensorInfo(y,o.dtype);const $=e.texData.get(p.dataId);$.values=C}else p=fc(o,u,e);c=Jt(c.length,a)}xe("max",c,a);const[f,m]=de(p.shape,c);let g=f;i&&(g=ne(f,l));let x;if(d){const w=e.texData.get(p.dataId).values,y=fP(w,q(m),g,o.dtype);x=e.makeTensorInfo(g,o.dtype);const C=e.texData.get(x.dataId);C.values=y}else x=EW(p,m,g,e);return h&&e.disposeIntermediateTensorInfo(p),x}const RW={kernelName:ka,backendName:"webgl",kernelFunc:Sy};const AW=rp+`
  return max(a, b);
`,DW=`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+ho+`
  return result;
`,FW=ye({opSnippet:AW,packedOpSnippet:DW,cpuKernelImpl:mP}),OW={kernelName:Tr,backendName:"webgl",kernelFunc:FW};function _W(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;Bi(o,"maxPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;S($e(i,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=nn(o.shape,r,i,c,a,l);if(u.filterWidth===1&&u.filterHeight===1&&Et(u.inShape,u.outShape))return Ke({inputs:{x:o},backend:e});const h=new Wi(u,"max",!1);return e.runWebGLProgram(h,[o],o.dtype)}const LW={kernelName:va,backendName:"webgl",kernelFunc:_W};function MW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dataFormat:l,dimRoundingMode:c}=s,u=[1,1,1],h=Hn(o.shape,r,i,u,a,c,l),d=new ap(h,"max",!1);return e.runWebGLProgram(d,[o],o.dtype)}const PW={kernelName:Sa,backendName:"webgl",kernelFunc:MW};class BW{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideHeight,s=t.strideWidth,o=t.dilationHeight,r=t.effectiveFilterHeight,i=t.effectiveFilterWidth,a=r-1-t.padInfo.top,l=i-1-t.padInfo.left,c=r*i-1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${l});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${r};
          wR += ${o}) {
          float dyR = float(dyRCorner + wR) / ${e}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${i}; wC++) {
            float dyC = float(dyCCorner + wC) / ${s}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${c} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${i} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class zW{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideDepth,s=t.strideHeight,o=t.strideWidth,r=t.dilationDepth,i=t.dilationHeight,a=t.dilationWidth,l=t.effectiveFilterDepth,c=t.effectiveFilterHeight,u=t.effectiveFilterWidth,h=l-1-t.padInfo.front,d=c-1-t.padInfo.top,p=u-1-t.padInfo.left,f=l*c*u-1;this.userCode=`
      const ivec3 pads = ivec3(${h}, ${d}, ${p});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${l};
           wD += ${r}) {
          float dyD = float(dyDCorner + wD) / ${e}.0;

          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${c};
              wR += ${i}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${a}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${f} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${c} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function VW(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r,{filterSize:a,strides:l,pad:c,dimRoundingMode:u}=s,h=[1,1,1],d=Hn(i.shape,a,l,h,c,u),p=new ap(d,"max",!0),f=e.runWebGLProgram(p,[i],i.dtype),m=new zW(d),g=e.runWebGLProgram(m,[o,f],i.dtype);return e.disposeIntermediateTensorInfo(f),g}const WW={kernelName:cu,backendName:"webgl",kernelFunc:VW};function UW(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r,output:i}=t,a=r;Bi([r,i],"maxPoolGrad");const{filterSize:l,strides:c,pad:u,dimRoundingMode:h}=s,d=nn(a.shape,l,c,1,u,h),p=!0,f=new Wi(d,"max",p),m=e.runWebGLProgram(f,[a],a.dtype),g=new BW(d),x=e.runWebGLProgram(g,[o,m],a.dtype);return e.disposeIntermediateTensorInfo(m),x}const GW={kernelName:lu,backendName:"webgl",kernelFunc:UW};function HW(n,t,e,s){let o=new Wi(e,"max",!1);const r=s.runWebGLProgram(o,[n],"float32");o=new Wi(e,"max",!0,!0,t);const i=s.runWebGLProgram(o,[n],"float32");return[r,i]}const qW={kernelName:Ep,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{filterSize:o,strides:r,pad:i,includeBatchInIndex:a}=t,l=e;S(s.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);const c=[1,1];S($e(r,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${r} and dilations '${c}'`);const u=nn(s.shape,o,r,c,i),[h,d]=HW(s,a,u,l);return[h,d]}};function KW(n,t,e,s){const o=q(t),i=q(n.shape)/o,a=et({inputs:{x:n},attrs:{shape:[i,o]},backend:s}),l=po(a,"float32","mean",s),c=et({inputs:{x:l},attrs:{shape:e},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(l),c}const jW={kernelName:Na,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{keepDims:o,axis:r}=t,i=e,a=s.shape.length,l=yt(r,s.shape);let c=l;const u=qt(c,a),h=u!=null,d=i.shouldExecuteOnCPU([s]),p=[];let f=s;if(h){if(d){const y=i.texData.get(f.dataId).values,C=new Array(a);for(let T=0;T<C.length;T++)C[T]=s.shape[u[T]];const $=op(y,s.shape,s.dtype,u,C);f=i.makeTensorInfo(C,s.dtype);const N=i.texData.get(f.dataId);N.values=$}else f=fc(s,u,i);p.push(f),c=Jt(c.length,a)}xe("sum",c,a);const[m,g]=de(f.shape,c);let x=m;o&&(x=ne(m,l));const b=KW(f,g,x,i);for(const w of p)i.disposeIntermediateTensorInfo(w);return b}};function XW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=yt(r,o.shape);let c=l;const u=qt(c,a);let h=o;u!=null&&(h=Fe({inputs:{x:o},backend:e,attrs:{perm:u}}),c=Jt(c.length,o.shape.length)),xe("min",c,a);const[d,p]=de(h.shape,c),f=q(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=po(m,m.dtype,"min",e);let x;if(i){const b=ne(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const YW={kernelName:Ta,backendName:"webgl",kernelFunc:XW};const ZW=rp+`
  return min(a, b);
`,JW=`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+ho+`
  return result;
`,QW=ye({opSnippet:ZW,packedOpSnippet:JW,cpuKernelImpl:gP}),t4={kernelName:Er,backendName:"webgl",kernelFunc:QW};class e4{constructor(t,e,s){this.variableNames=["x"],this.outputShape=e.map((u,h)=>u[0]+t[h]+u[1]);const o=t.length,r=Ot(o),i=e.map(u=>u[0]).join(","),a=e.map((u,h)=>u[0]+t[h]).join(","),l=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o),c=s==="reflect"?0:1;if(o===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${c};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${c};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${i});
      ${r} end = ${r}(${a});

      void main() {
        ${r} outC = getOutputCoords();
        for (int i = 0; i < ${o}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${c};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${c};
          }
        }
        ${r} coords = outC - start;
        setOutput(getX(${l}));
      }
    `}}class n4{constructor(t,e,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e.map((f,m)=>f[0]+t[m]+f[1]);const o=t.length,r=Ot(o),i=e.map(f=>f[0]).join(","),a=e.map((f,m)=>f[0]+t[m]).join(","),l=De("rc",o),c=De("source",o),u=`${l[o-1]} < ${this.outputShape[o-1]}`,h=o===1?"source":`vec2(${c.slice(-2).join()})`,d=s==="reflect"?0:1;let p="";if(o===1){const f=`
        ${r} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;p=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${c.join()}), ${h});
        ${l[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${c.join()}), ${h});
        }
      `}else{const f=`
        ${r} source = rc;
        ${r} lt = ${r}(lessThan(source, start));
        ${r} gte = ${r}(greaterThanEqual(source, end));
        ${r} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;p=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${c.join()}), ${h});
        ${l[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${c.join()}), ${h});
        }
        rc = outputLoc;
        ${l[o-2]} += 1;
        if(${l[o-2]} < ${this.outputShape[o-2]}) {
          ${f}
          result[2] = getChannel(getX(${c.join()}), ${h});
          ${l[o-1]} += 1;
          if(${u}) {
            ${f}
            result[3] = getChannel(getX(${c.join()}), ${h});
          }
        }
      `}this.userCode=`
      const ${r} start = ${r}(${i});
      const ${r} end = ${r}(${a});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}}const s4={kernelName:Ea,backendName:"webgl",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s}=n,{paddings:o,mode:r}=e,i=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new n4(s.shape,o,r):new e4(s.shape,o,r);return t.runWebGLProgram(i,[s],s.dtype)}};const o4=`if (b == 0.0) return NAN;
  return mod(a, b);`,r4=`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+ho+`
  return result;
`,i4=ye({opSnippet:o4,packedOpSnippet:r4}),a4={kernelName:Rr,backendName:"webgl",kernelFunc:i4};class l4{constructor(t,e,s){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[t,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${e-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${e-1}));
      }
    `}}const Ny=ye({opSnippet:`
if (a == b) {
  return 1.0;
};
return a / b;`,packedOpSnippet:`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,checkOutOfBounds:!0}),c4={kernelName:fr,backendName:"webgl",kernelFunc:Ny};const Ty="return a - b;",Ey=ye({opSnippet:Ty,packedOpSnippet:Ty,supportsComplex:!0,cpuKernelImpl:MP}),u4={kernelName:qr,backendName:"webgl",kernelFunc:Ey};function Ry(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{dim:r}=s,i=yt([r],o.shape),a=Sy({inputs:{x:o},backend:e,attrs:{reductionIndices:i,keepDims:!1}}),l=ne(a.shape,i),c=et({inputs:{x:a},backend:e,attrs:{shape:l}}),u=Ey({inputs:{a:o,b:c},backend:e}),h=wy({inputs:{x:u},backend:e}),d=mc({inputs:{x:h},backend:e,attrs:{axis:i,keepDims:!1}}),p=et({inputs:{x:d},backend:e,attrs:{shape:l}}),f=Ny({inputs:{a:h,b:p},backend:e});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),f}const h4={kernelName:Ka,backendName:"webgl",kernelFunc:Ry};function d4(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{numSamples:r,seed:i,normalized:a}=s,l=a?o:Ry({inputs:{logits:o},backend:e,attrs:{dim:o.shape.length-1}}),c=l.shape[0],u=l.shape[1],h=new l4(c,u,r),d=[[i]],p=e.runWebGLProgram(h,[l],"int32",d);return a||e.disposeIntermediateTensorInfo(l),p}const p4={kernelName:Rp,backendName:"webgl",kernelFunc:d4};const f4=dn+`
  return -x;
`,m4=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function g4(n){const{inputs:t,backend:e}=n,{x:s}=t;if(e.shouldExecuteOnCPU([s])){const r=e.texData.get(s.dataId),[i,a]=bP(r.values,s.shape,s.dtype);return e.makeTensorInfo(a,s.dtype,i)}let o;return U().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new vs(s.shape,m4):o=new Wn(s.shape,f4),e.runWebGLProgram(o,[s],s.dtype)}const x4={kernelName:Ra,backendName:"webgl",kernelFunc:g4};const b4=mh;function y4(n){je("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l}=s,c=e.readSync(o.dataId),u=e.readSync(r.dataId),{selectedIndices:h}=b4(c,u,i,a,l);return e.makeTensorInfo([h.length],"int32",new Int32Array(h))}const w4={kernelName:uu,backendName:"webgl",kernelFunc:y4};const C4=gh;function I4(n){je("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,padToMaxOutputSize:c}=s,u=e.readSync(o.dataId),h=e.readSync(r.dataId),{selectedIndices:d,validOutputs:p}=C4(u,h,i,a,l,c);return[e.makeTensorInfo([d.length],"int32",new Int32Array(d)),e.makeTensorInfo([],"int32",new Int32Array([p]))]}const $4={kernelName:hu,backendName:"webgl",kernelFunc:I4};const k4=xh;function v4(n){je("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,softNmsSigma:c}=s,u=e.readSync(o.dataId),h=e.readSync(r.dataId),d=i,p=a,f=l,m=c,{selectedIndices:g,selectedScores:x}=k4(u,h,d,p,f,m);return[e.makeTensorInfo([g.length],"int32",new Int32Array(g)),e.makeTensorInfo([x.length],"float32",new Float32Array(x))]}const S4={kernelName:du,backendName:"webgl",kernelFunc:v4};class N4{constructor(t,e,s,o){this.variableNames=["indices"],this.outputShape=[t,e],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${o}), float(${s}),
                      float(index == coords.y)));
      }
    `}}const T4={kernelName:Fa,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{indices:o}=t,{dtype:r,depth:i,onValue:a,offValue:l}=s,c=q(o.shape),u=new N4(c,i,a,l),h=et({inputs:{x:o},backend:e,attrs:{shape:[c]}}),d=e.runWebGLProgram(u,[h],r);e.disposeIntermediateTensorInfo(h);const p=[...o.shape,i],f=et({inputs:{x:d},backend:e,attrs:{shape:p}});return e.disposeIntermediateTensorInfo(d),f}};function Cc(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="complex64"){const o=Ui({inputs:{input:s},backend:e}),r=Cc({inputs:{x:o},backend:e}),i=yc({inputs:{input:s},backend:e}),a=Cc({inputs:{x:i},backend:e}),l=Ss({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return qi({attrs:{shape:s.shape,dtype:s.dtype,value:s.dtype==="string"?"":0},backend:e})}const E4={kernelName:Ya,backendName:"webgl",kernelFunc:Cc};function Ay(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("onesLike is not supported under string dtype");if(s.dtype==="complex64"){const o=Ui({inputs:{input:s},backend:e}),r=Ay({inputs:{x:o},backend:e}),i=yc({inputs:{input:s},backend:e}),a=Cc({inputs:{x:i},backend:e}),l=Ss({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return qi({attrs:{shape:s.shape,dtype:s.dtype,value:1},backend:e})}const R4={kernelName:Da,backendName:"webgl",kernelFunc:Ay};function A4(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s;if(t.length===1)return up({inputs:{input:t[0]},backend:e,attrs:{dim:o}});const r=t[0].shape,i=t[0].dtype;t.forEach(u=>{vc(r,u.shape,"All tensors passed to stack must have matching shapes"),S(i===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=[],l=t.map(u=>{const h=up({inputs:{input:u},backend:e,attrs:{dim:o}});return a.push(h),h}),c=cy({inputs:l,backend:e,attrs:{axis:o}});return a.forEach(u=>e.disposeIntermediateTensorInfo(u)),c}const D4={kernelName:Oa,backendName:"webgl",kernelFunc:A4};class F4{constructor(t,e,s){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=e.map((c,u)=>c[0]+t[u]+c[1]);const o=t.length,r=Ot(o),i=e.map(c=>c[0]).join(","),a=e.map((c,u)=>c[0]+t[u]).join(","),l=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o);if(o===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${i});
      ${r} end = ${r}(${a});

      void main() {
        ${r} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${r} coords = outC - start;
          setOutput(getX(${l}));
        }
      }
    `}}class O4{constructor(t,e,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=e.map((m,g)=>m[0]+t[g]+m[1]);const o=t.length,r=Ot(o),i=e.map(m=>m[0]).join(","),a=e.map((m,g)=>m[0]+t[g]).join(","),l=De("rc",o),c=De("source",o),u=`${l[o-1]} < ${this.outputShape[o-1]}`,h=o===1?"source":`vec2(${c.slice(-2).join()})`,d=[`${r} rc = outputLoc;`,`${l[o-1]} += 1;
       if(${u}) {
      `,o===1?"":`}
       rc = outputLoc;
       ${l[o-2]} += 1;
       if(${l[o-2]} < ${this.outputShape[o-2]}) {`,o===1?"":`  ${l[o-1]} += 1;
         if(${u}) {`],p=o===1?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let f="";for(let m=0,g=o===1?2:4;m<g;m++)f+=`
        ${d[m]}
        if (${p}) {
          result[${m}] = float(value);
        } else {
          ${r} source = rc - start;
          result[${m}] = getChannel(getX(${c.join()}), ${h});
        }
      `;f+=o===1?"} ":"}}",this.userCode=`
      const ${r} start = ${r}(${i});
      const ${r} end = ${r}(${a});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}}const Dy=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,constantValue:i}=s;if(q(o.shape)===0){const c=r.map((u,h)=>u[0]+o.shape[h]+u[1]);return qi({backend:e,attrs:{shape:c,value:i,dtype:o.dtype}})}const a=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new O4(o.shape,r,i):new F4(o.shape,r,i),l=[[i]];return e.runWebGLProgram(a,[o],o.dtype,l)},_4={kernelName:_a,backendName:"webgl",kernelFunc:Dy};const L4=`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,M4=`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+ho+`
  return result;
`,P4=ye({opSnippet:L4,packedOpSnippet:M4}),B4={kernelName:Dr,backendName:"webgl",kernelFunc:P4};function z4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=[],c=yt(r,o.shape);let u=c;const h=qt(u,a);let d=o;h!=null&&(d=Fe({inputs:{x:o},backend:e,attrs:{perm:h}}),u=Jt(u.length,a),l.push(d)),xe("prod",u,a);let p;if(e.shouldExecuteOnCPU([d])){const f=e.texData.get(d.dataId).values,{outVals:m,outShape:g,outDtype:x}=wP(d.shape,d.dtype,f,u);p=e.makeTensorInfo(g,x,m)}else{const[f,m]=de(d.shape,u),g=q(m),x=et({inputs:{x:d},backend:e,attrs:{shape:[-1,g]}}),b=Du(o.dtype),w=po(x,b,"prod",e);p=et({inputs:{x:w},backend:e,attrs:{shape:f}}),l.push(x),l.push(w)}if(i){l.push(p);const f=ne(p.shape,c);p=et({inputs:{x:p},backend:e,attrs:{shape:f}})}return l.forEach(f=>e.disposeIntermediateTensorInfo(f)),p}const V4={kernelName:Ma,backendName:"webgl",kernelFunc:z4};function W4(n){const{inputs:t,backend:e,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:i}=t,{outputRaggedRank:a}=s,l=o.map(x=>e.readSync(x.dataId)),c=o.map(x=>x.shape),u=e.readSync(r.dataId),h=e.readSync(i.dataId),[d,p,f]=CP(l,c,u,r.shape,r.dtype,h,i.shape,a),m=d.map(x=>e.makeTensorInfo([x.length],"int32",x)),g=e.makeTensorInfo(f,r.dtype,p);return m.concat([g])}const U4={kernelName:Ap,backendName:"webgl",kernelFunc:W4};function G4(n){const{inputs:t,backend:e}=n,{starts:s,limits:o,deltas:r}=t,i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=IP(i,s.shape,s.dtype,a,o.shape,l,r.shape),h=e.makeTensorInfo([c.length],"int32",c),d=e.makeTensorInfo([u.length],s.dtype,u);return[h,d]}const H4={kernelName:Dp,backendName:"webgl",kernelFunc:G4};function q4(n){const{inputs:t,backend:e,attrs:s}=n,{shape:o,values:r,defaultValue:i,rowPartitionTensors:a}=t,{rowPartitionTypes:l}=s,c=e.readSync(o.dataId),u=e.readSync(r.dataId),h=e.readSync(i.dataId),d=a.map(g=>e.readSync(g.dataId)),p=a.map(g=>g.shape),[f,m]=$P(c,o.shape,u,r.shape,r.dtype,h,i.shape,d,p,l);return e.makeTensorInfo(f,r.dtype,m)}const K4={kernelName:Fp,backendName:"webgl",kernelFunc:q4};const Fy=n=>{const{backend:t,attrs:e}=n,{start:s,stop:o,step:r,dtype:i}=e,a=kP(s,o,r,i);return t.makeTensorInfo([a.length],i,a)},j4={kernelName:pu,backendName:"webgl",kernelFunc:Fy};const X4=St({opSnippet:"return 1.0 / x;"}),Y4={kernelName:Fr,backendName:"webgl",kernelFunc:X4};const Z4=dn+`
  return (x < 0.0) ? 0.0 : x;
`,J4=St({opSnippet:Z4,packedOpSnippet:`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`}),Q4={kernelName:Or,backendName:"webgl",kernelFunc:J4};const tU=dn+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,eU=St({opSnippet:tU,packedOpSnippet:`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`}),nU={kernelName:_r,backendName:"webgl",kernelFunc:eU};class sU{constructor(t,e,s,o,r){this.variableNames=["A"],this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s];let d;r?d="(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":d="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${l}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}class oU{constructor(t,e,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s];let d;r?d="(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":d="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${l}.0,
                                     ${l}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${s-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}function rU(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s,[l,c]=a,u=U().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new oU(o.shape,l,c,r,i):new sU(o.shape,l,c,r,i);return e.runWebGLProgram(u,[o],"float32")}const iU={kernelName:za,backendName:"webgl",kernelFunc:rU};class aU{constructor(t,e,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e;const[,o,r]=e,[,i,a]=t,l=[s&&i>1?o-1:o,s&&a>1?r-1:r],c=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=l[0]/c[0],h=l[1]/c[1],d=1/u,p=1/h,f=Math.ceil(d)*2+2,m=Math.ceil(p)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${f});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${o-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${r-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function lU(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s,a=new aU(r.shape,o.shape,i);return e.runWebGLProgram(a,[r],r.dtype)}const cU={kernelName:gu,backendName:"webgl",kernelFunc:lU};class uU{constructor(t,e,s,o,r){this.variableNames=["A"],this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s],d=o?"0.5":"0.0";let p;r?p="max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":p="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${l}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${p};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}class hU{constructor(t,e,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s],d=o?"0.5":"0.0";let p;r?p="max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":p="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${l}.0,
                                     ${l}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${p};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${s-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}function dU(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s,[l,c]=a,u=U().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new hU(o.shape,l,c,r,i):new uU(o.shape,l,c,r,i);return e.runWebGLProgram(u,[o],o.dtype)}const pU={kernelName:Ba,backendName:"webgl",kernelFunc:dU};class fU{constructor(t,e,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e;const[,o,r]=e,[,i,a]=t,l=[s&&i>1?o-1:o,s&&a>1?r-1:r],c=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=l[0]/c[0],h=l[1]/c[1],d=1/u,p=1/h,f=Math.ceil(d)*2+2,m=Math.ceil(p)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${f});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float sourceFracRow =
              float(${l[0]}) *
                (float(dyR) / float(${c[0]}));

            float sourceFracCol =
                float(${l[1]}) *
                  (float(dyC) / float(${c[1]}));

            int sourceNearestRow = int(min(
                float(int(${o}) - 1),
                ${s} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${r}) - 1),
                ${s} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function mU(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s,a=new fU(r.shape,o.shape,i);return e.runWebGLProgram(a,[r],r.dtype)}const gU={kernelName:mu,backendName:"webgl",kernelFunc:mU};class xU{constructor(t,e){this.variableNames=["x"];const s=t.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);if(this.outputShape=t,s===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${t[0]} - coord - 1));
        }
      `;return}const o=a=>e.indexOf(a)!==-1&&t[a]!==1?`${t[a]} - coords[${a}] - 1`:`coords[${a}]`,r=t.map((a,l)=>o(l)).join(","),i=Ot(s);this.userCode=`
      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${r}));
      }
    `}}class bU{constructor(t,e){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const s=t.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);this.outputShape=t;const o=De("rc",s),r=`${o[s-1]} + 1 < ${this.outputShape[s-1]}`,i=`${o[s-2]} + 1 < ${this.outputShape[s-2]}`,a=Ot(s);s===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${t[0]} - rc - 1),
            ${t[0]} - rc - 1);
          if(${r}){
              result.g = getChannel(getX(${t[0]} - (rc  + 1) - 1),
                ${t[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${a} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${l(o.slice())};
          if(${r}){
            result.g = ${c(o.slice())};
          }
          if(${i}) {
            result.b = ${u(o.slice())};
            if(${r}) {
              result.a = ${h(o.slice())};
            }
          }
          setOutput(result);
        }
    `;function l(f){return d(f)}function c(f){return f[s-1]="("+f[s-1]+" + 1)",d(f)}function u(f){return f[s-2]="("+f[s-2]+" + 1)",d(f)}function h(f){return f[s-1]="("+f[s-1]+" + 1)",f[s-2]="("+f[s-2]+" + 1)",d(f)}function d(f){const m=t.map((b,w)=>p(w,f)),g=m.join(","),x=m.slice(-2).join(",");return`getChannel(getX(${g}), vec2(${x}))`}function p(f,m){return e.indexOf(f)!==-1&&t[f]!==1?`${t[f]} - ${m[f]} - 1`:`${m[f]}`}}}function yU(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dims:r}=s,i=o.shape.length,a=yt(r,o.shape);if(i===0)return Ke({inputs:{x:o},backend:e});const l=U().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new bU(o.shape,a):new xU(o.shape,a);return e.runWebGLProgram(l,[o],o.dtype)}const wU={kernelName:Va,backendName:"webgl",kernelFunc:yU};class CU{constructor(t,e){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const s=t[1],o=t[2];this.outputShape=t;let r="";typeof e=="number"?r=`float outputValue = ${e.toFixed(2)};`:r=`
        vec3 fill = vec3(${e.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${r}
          if(coordX >= 0 && coordX < ${o} && coordY >= 0 && coordY < ${s}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const IU={kernelName:$u,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,{radians:o,fillValue:r,center:i}=t,a=e,l=new CU(s.shape,r),[c,u]=Eh(i,s.shape[1],s.shape[2]),h=[[c,u,Math.sin(o),Math.cos(o)]];return a.runWebGLProgram(l,[s],s.dtype,h)}};const $U=St({opSnippet:`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`}),kU={kernelName:Lr,backendName:"webgl",kernelFunc:$U};const vU=St({opSnippet:"return inversesqrt(x);",cpuKernelImpl:vP}),SU={kernelName:Mr,backendName:"webgl",kernelFunc:vU};class dp{constructor(t,e,s,o,r,i,a=!0,l=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=i;const c=Ot(r.length),u=Ot(i.length);let h="";s===1?h="i":s===2&&(h="i, j");const d=`getIndices(${h})`;let p="";o===1?p="i":o===2&&(p="i, coords[1]");const f=`getUpdates(${p})`;let m="";l&&(m="coords[0], coords[1]");const g=`getDefaultValue(${m})`,x=e>1?"strides[j]":"strides";this.userCode=`
        ${c} strides = ${c}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${t}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${e}; j++) {
              int index = round(${d});
              flattenedIndex += index * ${x};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${f};
              found = true;
            }
          }
          setOutput(mix(${g}, sum, float(found)));
        }
      `}}class NU{constructor(t,e,s,o,r,i,a=!0,l=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=i;const c=Ot(r.length),u=Ot(i.length);let h="";s===1?h="i":s===2&&(h="i, j");const d=`getIndices(${h})`;let p="";o===1?p="i":o===2&&(p="i, coords[1]");const f=`getUpdates(${p})`;let m="";l&&(m="coords[0], coords[1]");const g=`getDefaultValue(${m})`,x=e>1?"strides[j]":"strides",b=e>1?"strides[j + 1]":"strides";this.userCode=`
        ${c} strides = ${c}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${t}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${e}; j+=2) {
              ivec4 index = round(${d});
              flattenedIndex += index.xz * ${x};
              if (j + 1 < ${e}) {
                flattenedIndex += index.yw * ${b};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${f};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${g}, sum, found));
        }
      `}}function TU(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o,updates:r}=t,{shape:i}=s,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Ks(r,o,i),d=[h/c,c];if(h===0)return e.makeTensorInfo(i,o.dtype);const p=et({inputs:{x:o},backend:e,attrs:{shape:[l,a]}}),f=et({inputs:{x:r},backend:e,attrs:{shape:[l,c]}}),m=e.makeTensorInfo([],"float32",new Float32Array([0]));let g;U().getBool("WEBGL_PACK")?g=new NU(l,a,p.shape.length,f.shape.length,u,d):g=new dp(l,a,p.shape.length,f.shape.length,u,d);const x=e.runWebGLProgram(g,[f,p,m],f.dtype),b=et({inputs:{x},backend:e,attrs:{shape:i}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(m),b}const EU={kernelName:Op,backendName:"webgl",kernelFunc:TU};class RU{constructor(t,e,s,o){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[t,s];const r="while (left < right) {",i=`for (int i = 0; i < ${Math.ceil(Math.log2(e+1))}; ++i) { if (left >= right) break;`,a=U().getNumber("WEBGL_VERSION")===2?r:i,l=o==="left"?"<":"<=";this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${l} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}function AU(n){const{inputs:t,backend:e,attrs:s}=n,{sortedSequence:o,values:r}=t,{side:i}=s,a=new RU(o.shape[0],o.shape[1],r.shape[1],i),l=[[o.shape[1]]];return e.runWebGLProgram(a,[o,r],"int32",l)}const DU={kernelName:Lp,backendName:"webgl",kernelFunc:AU};class FU{constructor(t,e,s){this.variableNames=["c","a","b"],this.outputShape=e;let o,r;if(s>4)throw Error(`Where for rank ${s} is not yet supported`);if(s===1)r="resRC",o="resRC";else{const a=["resRC.x","resRC.y","resRC.z","resRC.w"],l=[],c=[];for(let u=0;u<e.length;u++)c.push(`${a[u]}`),u<t&&l.push(`${a[u]}`);o=l.join(),r=c.join()}const i=Ot(s);this.userCode=`
      void main() {
        ${i} resRC = getOutputCoords();
        float cVal = getC(${o});
        if (cVal >= 1.0) {
          setOutput(getA(${r}));
        } else {
          setOutput(getB(${r}));
        }
      }
    `}}function OU(n){const{inputs:t,backend:e}=n,{condition:s,t:o,e:r}=t,i=new FU(s.shape.length,o.shape,o.shape.length);return e.runWebGLProgram(i,[s,o,r],Ue(o.dtype,r.dtype))}const _U={kernelName:Wa,backendName:"webgl",kernelFunc:OU};const LU=`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${vl};
  float scale = ${Sl};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`,MU=St({opSnippet:LU}),PU={kernelName:Pr,backendName:"webgl",kernelFunc:MU};const BU=Zo+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,zU=St({opSnippet:BU,packedOpSnippet:`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,cpuKernelImpl:NP}),VU={kernelName:Wr,backendName:"webgl",kernelFunc:zU};const WU=St({opSnippet:`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`}),UU={kernelName:Vr,backendName:"webgl",kernelFunc:WU};const GU=Zo+`
  return sin(x);
`,HU=`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${ho}
  return result;
`,qU=St({opSnippet:GU,packedOpSnippet:HU}),KU={kernelName:Br,backendName:"webgl",kernelFunc:qU};const jU=St({opSnippet:`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`}),XU={kernelName:zr,backendName:"webgl",kernelFunc:jU};const YU=St({opSnippet:`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`}),ZU={kernelName:Ur,backendName:"webgl",kernelFunc:YU};const JU={kernelName:Ha,backendName:"webgl",kernelFunc:n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,paddings:i}=s;S(o.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const a=r.reduce((x,b)=>x*b),l=[[0,0]];l.push(...i);for(let x=1+r.length;x<o.shape.length;++x)l.push([0,0]);const c=[],u=Dy({inputs:{x:o},backend:e,attrs:{paddings:l,constantValue:0}}),h=fi(u.shape,r,a,!1),d=mi(h.length,r.length,!1),p=gi(u.shape,r,a,!1),f=et({inputs:{x:u},backend:e,attrs:{shape:h}}),m=Fe({inputs:{x:f},backend:e,attrs:{perm:d}}),g=et({inputs:{x:m},backend:e,attrs:{shape:p}});return c.push(u),c.push(f),c.push(m),c.forEach(x=>e.disposeIntermediateTensorInfo(x)),g}};function QU(n){const{inputs:t,backend:e}=n,{indices:s,values:o,denseShape:r,defaultValue:i}=t;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
         ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
         ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
         ${o.shape}`);if(i.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${i.shape}`);const a=e.readSync(s.dataId),l=e.readSync(o.dataId),c=e.readSync(r.dataId),u=e.readSync(i.dataId)[0],[h,d,p,f,m]=EP(a,s.shape,s.dtype,l,o.dtype,c,u);return[e.makeTensorInfo(d,s.dtype,h),e.makeTensorInfo([d[0]],o.dtype,p),e.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(g=>Number(g)))),e.makeTensorInfo([m.length],s.dtype,new Int32Array(m))]}const tG={kernelName:Mp,backendName:"webgl",kernelFunc:QU};function eG(n){const{inputs:t,backend:e}=n,{inputIndices:s,inputShape:o,newShape:r}=t;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const i=Array.from(e.readSync(o.dataId)),a=e.readSync(s.dataId),l=Array.from(e.readSync(r.dataId)),[c,u,h]=RP(a,s.shape,s.dtype,i,l);return[e.makeTensorInfo(u,s.dtype,c),e.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}const nG={kernelName:Pp,backendName:"webgl",kernelFunc:eG};function sG(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
              ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
              ${r.shape}`);const i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=P1(i,s.shape,s.dtype,a,l,!0);return e.makeTensorInfo(u,s.dtype,c)}const oG={kernelName:Bp,backendName:"webgl",kernelFunc:sG};function rG(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
             ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
             ${r.shape}`);const i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=P1(i,s.shape,s.dtype,a,l);return e.makeTensorInfo(u,s.dtype,c)}const iG={kernelName:zp,backendName:"webgl",kernelFunc:rG};function aG(n){const{inputs:t,backend:e,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:i}=t,{outputShape:a}=s,{sliceRank:l,numUpdates:c,sliceSize:u,strides:h,outputSize:d}=Ks(r,o,a),p=!1;if(r.dtype==="string"){const x=e.bufferSync(o),b=e.bufferSync(r),w=as(e.readSync(i.dataId)[0]),y=SP(x,b,a,d,u,c,l,h,w,p);return e.makeTensorInfo(a,y.dtype,y.values)}const f=new dp(c,l,o.shape.length,r.shape.length,h,[d,1],p),m=e.runWebGLProgram(f,[r,o,i],r.dtype),g=et({inputs:{x:m},backend:e,attrs:{shape:a}});return e.disposeIntermediateTensorInfo(m),g}const lG={kernelName:Vp,backendName:"webgl",kernelFunc:aG};function cG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{numOrSizeSplits:r,axis:i}=s,a=yt(i,o.shape)[0],l=Hh(o,r,a),c=o.shape.length,u=new Array(c).fill(0),h=o.shape.slice();return l.map(d=>{const p=[...h];p[a]=d;const f=Jo({inputs:{x:o},backend:e,attrs:{begin:u,size:p}});return u[a]+=d,f})}const uG={kernelName:qa,backendName:"webgl",kernelFunc:cG};const Oy="return sqrt(x);",hG=St({opSnippet:Oy,packedOpSnippet:Oy,cpuKernelImpl:AP}),dG={kernelName:Gr,backendName:"webgl",kernelFunc:hG};const pG=St({opSnippet:"return x * x;"}),fG={kernelName:xu,backendName:"webgl",kernelFunc:pG};const _y="return (a - b) * (a - b);",mG=ye({opSnippet:_y,packedOpSnippet:_y}),gG={kernelName:Hr,backendName:"webgl",kernelFunc:mG};function xG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;if(o.dtype!=="string")throw new Error("Input must be of datatype string");const r=e.readSync(o.dataId),i=Zn(r),a=DP(i,"string",s);return e.makeTensorInfo(o.shape,"string",a)}const bG={kernelName:bu,backendName:"webgl",kernelFunc:xG};function yG({inputs:n,attrs:t,backend:e}){const{x:s}=n,o=dn+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,r=new Wn(s.shape,o);return e.runWebGLProgram(r,[s],s.dtype)}const wG={kernelName:Yr,backendName:"webgl",kernelFunc:yG};class CG{constructor(t,e,s){this.variableNames=["x"],this.outputShape=s;const o=s.length,r=Ot(s.length),i=Ot(s.length);let a="";if(o===1)a="coords * strides + begin";else{let l=0;a=s.map((c,u)=>(l++,s.length===1?`coords * strides[${u}] + begin[${u}]`:`coords[${l-1}] * strides[${u}] + begin[${u}]`)).join(",")}this.userCode=`
      ${r} begin = ${r}(${t});
      ${r} strides = ${r}(${e});

      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${a}));
      }
    `}}function IG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,end:i,strides:a,beginMask:l,endMask:c,ellipsisMask:u,newAxisMask:h,shrinkAxisMask:d}=s,{finalShapeSparse:p,finalShape:f,isIdentity:m,sliceDim0:g,isSimpleSlice:x,begin:b,end:w,strides:y}=Sh(o.shape,r,i,a,l,c,u,h,d);let C;if(m)C=et({inputs:{x:o},backend:e,attrs:{shape:f}});else if(g||x){S(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const N=$h(b,w,y),T=Jo({inputs:{x:o},backend:e,attrs:{begin:b,size:N}});C=et({inputs:{x:T},backend:e,attrs:{shape:f}}),e.disposeIntermediateTensorInfo(T)}else if(e.shouldExecuteOnCPU([o])){const T=e.readSync(o.dataId),k=wt(o.shape,o.dtype,T),v=FP(p,k,y,b);C=e.makeTensorInfo(f,o.dtype,v.values)}else{const T=new CG(b,y,p);C=e.runWebGLProgram(T,[o],o.dtype)}const $=et({inputs:{x:C},backend:e,attrs:{shape:f}});return e.disposeIntermediateTensorInfo(C),$}const $G={kernelName:yu,backendName:"webgl",kernelFunc:IG};function kG(n){const{inputs:t,backend:e,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:i,rightPad:a,padWidth:l,preserveShortSequences:c}=s,{data:u,dataSplits:h}=t,d=e.readSync(u.dataId),p=e.readSync(h.dataId),[f,m]=OP(d,p,o,r,i,a,l,c);return[e.makeTensorInfo([f.length],"string",f),e.makeTensorInfo(h.shape,"int32",m)]}const vG={kernelName:Wp,backendName:"webgl",kernelFunc:kG};function SG(n){const{inputs:t,backend:e,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:i}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(i.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${i.shape}`);const a=e.readSync(r.dataId),l=e.readSync(i.dataId)[0],[c,u,h]=_P(a,l,o),d=u.length;return[e.makeTensorInfo([d,2],"int32",c),e.makeTensorInfo([d],"string",u),e.makeTensorInfo([2],"int32",new Int32Array(h))]}const NG={kernelName:Up,backendName:"webgl",kernelFunc:SG};function TG(n){const{inputs:t,backend:e,attrs:s}=n,{numBuckets:o}=s,{input:r}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const i=e.readSync(r.dataId),a=LP(i,o);return e.makeTensorInfo(r.shape,"int32",a)}const EG={kernelName:Gp,backendName:"webgl",kernelFunc:TG};const RG=St({opSnippet:"return tan(x);"}),AG={kernelName:Kr,backendName:"webgl",kernelFunc:RG};const DG=St({opSnippet:`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`}),FG={kernelName:jr,backendName:"webgl",kernelFunc:DG};function OG(n){const{inputs:t,backend:e,attrs:s}=n,{tensor:o,indices:r,updates:i}=t,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Ks(i,r,o.shape),d=[h/c,c];if(h===0)return e.makeTensorInfo(o.shape,r.dtype);const p=et({inputs:{x:r},backend:e,attrs:{shape:[l,a]}}),f=et({inputs:{x:i},backend:e,attrs:{shape:[l,c]}}),m=et({inputs:{x:o},backend:e,attrs:{shape:d}}),g=new dp(l,a,p.shape.length,f.shape.length,u,d,!1,!0),x=e.runWebGLProgram(g,[f,p,m],m.dtype),b=et({inputs:{x},backend:e,attrs:{shape:o.shape}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(x),b}const _G={kernelName:_p,backendName:"webgl",kernelFunc:OG};class LG{constructor(t,e){this.variableNames=["A"];const s=new Array(t.length);for(let i=0;i<s.length;i++)s[i]=t[i]*e[i];this.outputShape=s,this.rank=s.length;const o=Ot(this.rank),r=MG(t);this.userCode=`
      void main() {
        ${o} resRC = getOutputCoords();
        setOutput(getA(${r}));
      }
    `}}function MG(n){const t=n.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(t===1)return`imod(resRC, ${n[0]})`;const e=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],s=[];for(let o=0;o<n.length;o++)s.push(`imod(${e[o]}, ${n[o]})`);return s.join()}function Ly(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reps:r}=s;if(o.dtype==="string"||o.shape.length>5){const l=e.readSync(o.dataId),c=o.dtype==="string"?l.map(d=>as(d)):l,u=wt(o.shape,o.dtype,c),h=PP(u,r);return e.makeTensorInfo(h.shape,h.dtype,h.values)}const i=new LG(o.shape,r);return e.runWebGLProgram(i,[o],o.dtype)}const PG={kernelName:Xr,backendName:"webgl",kernelFunc:Ly};class BG{constructor(t){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=t,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class zG{constructor(t){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=t,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}function fo(n,t){t!==null&&n.disposeIntermediateTensorInfo(t)}function My(n){let t=1;for(;t<n;)t*=2;return t}function VG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{k:r,sorted:i}=s,a=U().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),l=U().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),c=o.shape,u=c[c.length-1];if(e.shouldExecuteOnCPU([o])||u<a||r>l){const v=e.readSync(o.dataId),[I,R]=BP(v,c,o.dtype,r,i);return[e.makeTensorInfo(I.shape,I.dtype,I.values),e.makeTensorInfo(R.shape,R.dtype,R.values)]}if(r===0)return c[c.length-1]=0,[e.makeTensorInfo(c,o.dtype,[]),e.makeTensorInfo(c,"int32",[])];if(u===1)return[o,qi({attrs:{shape:c,dtype:"int32",value:0},backend:e})];const h=e.texData.get(o.dataId),d=h!==null&&h.isPacked,p=d?e.unpackTensor(o):o,m=q(c)/u,g=et({inputs:{x:p},attrs:{shape:[m,u]},backend:e});d&&fo(e,p);const x=My(r),b=My(u);let w=null;const y=()=>w===null?[g,g]:[g,w],C=(v,I,R)=>{const A=y(),P=new BG(R),B=[[u],[w===null?1:0],[Number.NEGATIVE_INFINITY],[v],[I]],z=w;w=e.runWebGLProgram(P,A,"int32",B),fo(e,z)};for(let v=1;v<x;v*=2){const I=v*2;for(let R=v;R>=1;R/=2)C(I,R,[m,b])}for(let v=b;v>x;v/=2){const I=y(),R=new zG([m,v/2]),P=[[u],[w===null?1:0],[x]],_=w;w=e.runWebGLProgram(R,I,"int32",P),fo(e,_);const B=x/2,z=B*2;for(let W=B;W>=1;W/=2)C(z,W,w.shape)}let $=w;w=Jo({inputs:{x:w},backend:e,attrs:{begin:0,size:[m,r]}}),fo(e,$);let N=vy({inputs:{x:g,indices:w},backend:e,attrs:{axis:1,batchDims:1}});fo(e,g);const T=c.slice(0,-1);T.push(r),$=w,w=et({inputs:{x:w},attrs:{shape:T},backend:e}),fo(e,$);const k=N;return N=et({inputs:{x:N},attrs:{shape:T},backend:e}),fo(e,k),[N,w]}const WG={kernelName:wu,backendName:"webgl",kernelFunc:VG};class UG{constructor(t,e,s,o,r,i){this.variableNames=["Image","Transforms"],this.outputShape=i;const a=s==="nearest"?1:2;let l;switch(o){case"constant":l=1;break;case"reflect":l=2;break;case"wrap":l=3;break;case"nearest":l=4;break;default:l=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${l} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${l} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${l} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${t} && 0 <= coordX && coordX < ${e}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${r});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${r});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${e}));
                float mapY = mapCoord(inY, float(${t}));

                if (${a} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}function GG(n){const{inputs:t,backend:e,attrs:s}=n,{image:o,transforms:r}=t,{interpolation:i,fillMode:a,fillValue:l,outputShape:c}=s,[u,h,d,p]=o.shape,[f,m]=c??[h,d],g=[u,f,m,p],x=new UG(h,d,i,a,l,g);return e.runWebGLProgram(x,[o,r],"float32")}const HG={kernelName:Cu,backendName:"webgl",kernelFunc:GG};function qG(n){const{inputs:t,attrs:e,backend:s}=n,{axis:o}=e,{x:r}=t;Bi(r,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");const i=s.readSync(r.dataId),{outputValues:a,outputShape:l,indices:c}=zP(i,o,r.shape,r.dtype);return[s.makeTensorInfo(l,r.dtype,a),s.makeTensorInfo([c.length],"int32",c)]}const KG={kernelName:Iu,backendName:"webgl",kernelFunc:qG};function jG(n){const{inputs:t,backend:e,attrs:s}=n,{value:o}=t;let{axis:r}=s;r<0&&(r+=o.shape.length);const i=o,a=i.shape.length,l=o.shape[r],c=new Array(a-1);let u=0;for(let m=0;m<a;m++)m!==r&&(c[u++]=i.shape[m]);const h=[],d=new Array(a).fill(0),p=i.shape.slice();p[r]=1;const f=new Array(l);for(let m=0;m<f.length;m++){d[r]=m;const g=Jo({inputs:{x:i},backend:e,attrs:{begin:d,size:p}}),x=et({inputs:{x:g},backend:e,attrs:{shape:c}});f[m]=x,h.push(g)}return h.forEach(m=>e.disposeIntermediateTensorInfo(m)),f}const XG={kernelName:ja,backendName:"webgl",kernelFunc:jG};class YG{constructor(t,e){this.variableNames=["x","segmentIds"];const s=t.windowSize,o=t.batchSize,r=t.inSize,i=t.numSegments,a=i*Math.ceil(r/s);this.outputShape=[o,a];const l="0.0",c="sumValue",u=Math.floor(s/4)*4,h=s%4,d=`
        sumValue += dot(values, segFilter);
    `;let p="";r%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `);let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = ${l};

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${f}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${i})) * float(${s}));
        int currentSeg = int(mod(float(outIdx), float(${i})));

        float sumValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${d}
        }

        int inIdx = inOffset + ${u};
        if (${h===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${d}
        } else if (${h===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${d}
        } else if (${h===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${d}
        }
        setOutput(${c});
      }
    `}}function ZG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,segmentIds:r}=t,{numSegments:i}=s,a=o.shape.length,l=[];let c=0;const u=qt([c],a);let h=o;u!=null&&(h=Fe({inputs:{x:o},backend:e,attrs:{perm:u}}),l.push(h),c=Jt(1,a)[0]);const d=hg(h.shape,c,i),p=q([h.shape[c]]),f=et({inputs:{x:h},backend:e,attrs:{shape:[-1,p]}});l.push(f);const m=Du(o.dtype),g=(y,C,$,N,T)=>{const k=y.shape[0],v=y.shape[1],I=ug(v,T),R={windowSize:I,inSize:v,batchSize:k,numSegments:T},A=new YG(R,C),P=e.compileAndRun(A,[y,$],N);if(l.push(P),P.shape[1]===T)return P;const _=Fy({backend:e,attrs:{start:0,stop:T,step:1,dtype:"float32"}}),B=Ly({inputs:{x:_},backend:e,attrs:{reps:[v/I]}});return l.push(_),l.push(B),g(P,C,B,N,T)},x=g(f,"unsortedSegmentSum",r,m,i),b=et({inputs:{x},backend:e,attrs:{shape:d}});let w=b;if(u!=null){l.push(b);const y=us(u);w=Fe({inputs:{x:w},backend:e,attrs:{perm:y}})}return l.forEach(y=>e.disposeIntermediateTensorInfo(y)),w}const JG={kernelName:Xa,backendName:"webgl",kernelFunc:ZG};const QG=[A3,F3,L3,B3,V3,G3,q3,j3,J3,tB,sB,iB,cB,pB,gB,bB,wB,kB,SB,TB,AB,MB,BB,UB,HB,YB,JB,nz,f3,rz,uz,fz,wz,$z,vz,Nz,Ez,Fz,_z,Mz,Bz,Vz,Uz,qz,jz,Jz,tV,sV,iV,lV,uV,pV,mV,bV,wV,CV,$V,vV,NV,EV,AV,FV,LV,BV,VV,GV,KV,XV,ZV,p3,QV,lz,eW,sW,rW,g3,aW,cW,hW,fW,xW,yW,CW,$W,SW,TW,RW,OW,LW,PW,WW,GW,qW,jW,YW,t4,s4,a4,p4,y3,x4,w4,$4,S4,qB,T4,R4,D4,_4,B4,b3,V4,U4,H4,K4,j4,KB,c4,Y4,Q4,nU,C3,iU,cU,pU,gU,wU,IU,kU,SU,EU,DU,_U,PU,VU,UU,KU,XU,LB,h4,ZU,JU,tG,nG,oG,iG,lG,uG,dG,fG,gG,bG,wG,$G,vG,NG,EG,u4,T3,AG,FG,_G,PG,WG,HG,E3,KG,XG,JG,E4];for(const n of QG)Xp(n);const pp=(n,t)=>{if(!n||n.length<t)return[];const e=[];for(let s=t-1;s<n.length;s++){const o=n.slice(s-t+1,s+1).reduce((r,i)=>r+i,0);e.push(o/t)}return e},Ki=(n,t)=>{if(n.length<t)return[];const e=2/(t+1);let s=[n[0]];for(let o=1;o<n.length;o++)s.push(n[o]*e+s[o-1]*(1-e));return s},tH=(n,t=14)=>{if(n.length<t+1)return[];let e=0,s=0;for(let l=1;l<=t;l++){const c=n[l]-n[l-1];c>=0?e+=c:s+=Math.abs(c)}let o=e/t,r=s/t;const i=[];let a=r===0?100:o/r;i.push(100-100/(1+a));for(let l=t+1;l<n.length;l++){const c=n[l]-n[l-1];c>=0?(o=(o*(t-1)+c)/t,r=(r*(t-1)+0)/t):(o=(o*(t-1)+0)/t,r=(r*(t-1)+Math.abs(c))/t),a=r===0?100:o/r,i.push(100-100/(1+a))}return i},Py=(n,t=20,e=2)=>{if(n.length<t)return[];const s=[],o=[],r=pp(n,t);for(let i=0;i<r.length;i++){const a=n.slice(i,i+t),l=r[i],u=a.map(d=>Math.pow(d-l,2)).reduce((d,p)=>d+p,0)/t,h=Math.sqrt(u);o.push(l+e*h),s.push(l-e*h)}return{basis:r,upper:o,lower:s}},eH=(n,t=12,e=26,s=9)=>{const o=Ki(n,t),r=Ki(n,e),i=[],a=Math.min(o.length,r.length);for(let u=0;u<a;u++)i.push(o[u]-r[u]);const l=Ki(i,s),c=[];for(let u=0;u<Math.min(i.length,l.length);u++)c.push(i[u]-l[u]);return{macdLine:i,signalLine:l,histogram:c}},nH=(n,t=[],e=[],s=[])=>{if(n.length<50)return[{name:"Insufficient Data",sentiment:"Neutral",confidence:0}];const o=n[n.length-1],r=n[n.length-2],i=s.length>0?s[s.length-1]:r,a=t.length>0?t[t.length-1]:Math.max(o,i),l=e.length>0?e[e.length-1]:Math.min(o,i),c=Math.abs(o-i),u=a-l||1e-4,h=(o+i)/2,d=o>i,p=o<i;let f=[];c/u<.1&&(h>a-u*.2?f.push({name:"Dragonfly Doji",sentiment:"Bullish",icon:"zap"}):h<l+u*.2?f.push({name:"Gravestone Doji",sentiment:"Bearish",icon:"trending-down"}):f.push({name:"Doji Star",sentiment:"Neutral",icon:"minus"}));const m=Math.min(i,o)-l,g=a-Math.max(i,o);if(m>c*2&&g<c*.5&&f.push({name:"Hammer",sentiment:"Bullish",icon:"thumbs-up"}),g>c*2&&m<c*.5&&f.push({name:d?"Inverted Hammer":"Shooting Star",sentiment:d?"Bullish":"Bearish",icon:d?"zap":"trending-down"}),s.length>1){const I=s[s.length-2],R=n[n.length-2],A=R<I,P=R>I;d&&A&&o>I&&i<R?f.push({name:"Bullish Engulfing",sentiment:"Bullish",icon:"zap"}):p&&P&&o<I&&i>R&&f.push({name:"Bearish Engulfing",sentiment:"Bearish",icon:"trending-down"})}if(s.length>1){const I=s[s.length-2],R=n[n.length-2],A=t[t.length-2]||Math.max(I,R),P=e[e.length-2]||Math.min(I,R);a<A&&l>P&&f.push({name:"Harami (Inside Bar)",sentiment:d?"Bullish":"Bearish",icon:"activity"})}c/u>.9&&f.push({name:d?"Bullish Marubozu":"Bearish Marubozu",sentiment:d?"Bullish":"Bearish",icon:"zap"});const x=pp(n.slice(-20),10),b=pp(n.slice(-50),40),w=x[x.length-1],y=b[b.length-1],C=w>y,{upper:$,lower:N}=Py(n,20),T=$[$.length-1],k=N[N.length-1];return(T-k)/o<.05&&f.push({name:"Volatility Squeeze",sentiment:"Neutral",icon:"activity"}),f.length===0&&(C?f.push({name:"Bullish Continuation",sentiment:"Bullish",icon:"trending-up"}):f.push({name:"Bearish Continuation",sentiment:"Bearish",icon:"trending-down"})),f},sH=(n,t,e,s=14)=>{let o=[];const r=Array.isArray(n)&&Array.isArray(t)&&Array.isArray(e),i=r||Array.isArray(n)?n.length:0;if(i<s+1)return[];for(let u=1;u<i;u++){let h;if(r){const d=n[u],p=t[u],f=e[u-1];h=Math.max(d-p,Math.abs(d-f),Math.abs(p-f))}else{const d=n[u],p=n[u-1],f=Math.abs(d-p);h=Math.max(f,d*.0075)}o.push(h)}const a=[];let l=o.slice(0,s).reduce((u,h)=>u+h,0)/s;a.push(l);for(let u=s;u<o.length;u++){const h=(a[a.length-1]*(s-1)+o[u])/s;a.push(h)}return[...new Array(i-a.length).fill(a[0]),...a]},oH=(n,t)=>{if(!n||n.length<t+1)return[];const e=[];for(let s=t;s<n.length;s++){const o=(n[s]-n[s-t])/n[s-t];e.push(o*100)}return e},rH=(n,t,e=252)=>{if(!n||n.length<t+1)return[];const s=[];for(let r=1;r<n.length;r++)s.push((n[r]-n[r-1])/n[r-1]);const o=[];for(let r=t-1;r<s.length;r++){const i=s.slice(r-t+1,r+1),a=i.reduce((u,h)=>u+h,0)/t,l=i.reduce((u,h)=>u+Math.pow(h-a,2),0)/t,c=Math.sqrt(l)*Math.sqrt(e);o.push(c)}return o};(async()=>{try{await Cf("webgl"),await Mw()}catch{await Cf("cpu")}})();const Wt=45,iH=25,aH=32,ji=8,Ve=(n,t,e)=>(n-t)/(e||1),tr=(n,t,e)=>(n-t)/(e||1),lH=n=>{const t=[...n].sort((i,a)=>i-a),e=t[Math.floor(t.length/2)],s=t[Math.floor(t.length*.25)],r=t[Math.floor(t.length*.75)]-s||1;return{median:e,iqr:r}},cH=n=>n.map(t=>{const e=(Math.random()-.5)*4e-4,s=.9992+Math.random()*.0016;return[t[0]*s+e,t[1]*s+e,t[2]*s+e,t[3]*s+e,t[4]*s+e,t[5]*s+e,t[6]*s+e,t[7]*s+e]}),uH=n=>{const{prices:t,rsi:e,macd:s,atr:o,roc:r,vol:i,emaRatio:a,bPercent:l}=n,c=Math.min(t.length,e.length,s.length,o?.length||0,r?.length||0,i?.length||0,a?.length||0,l?.length||0),u=t.slice(-c),h=e.slice(-c),d=s.slice(-c),p=o.slice(-c),f=r.slice(-c),m=i.slice(-c),g=a.slice(-c),x=l.slice(-c),b=[];return[h,d,p].forEach(w=>{const y=w.reduce(($,N)=>$+N,0)/(w.length||1),C=Math.sqrt(w.reduce(($,N)=>$+Math.pow(N-y,2),0)/(w.length||1))||1;b.push({mean:y,std:C})}),[f,m].forEach(w=>{const y=lH(w);b.push(y)}),[g,x].forEach(w=>{const y=w.reduce(($,N)=>$+N,0)/(w.length||1),C=Math.sqrt(w.reduce(($,N)=>$+Math.pow(N-y,2),0)/(w.length||1))||1;b.push({mean:y,std:C})}),[{base:u[0]},...b]},hH=(n,t)=>{const{prices:e,rsi:s,macd:o,atr:r,roc:i,vol:a,emaRatio:l,bPercent:c}=n,u=Math.min(e.length,s.length,o.length,r?.length||0,i?.length||0,a?.length||0,l?.length||0,c?.length||0),h=uH(n),d=e.slice(-u),p=s.slice(-u),f=o.slice(-u),m=r.slice(-u),g=i.slice(-u),x=a.slice(-u),b=l.slice(-u),w=c.slice(-u),y=u-t;if(y<=0)return{xs:Il([],[0,t,ji]),ys:Fo([],[0,1]),stats:h};const C=[],$=[];for(let k=0;k<y;k++){const v=d[k],I=[];for(let A=0;A<t;A++)I.push([(d[k+A]-v)/(v||1),Ve(p[k+A],h[1].mean,h[1].std),Ve(f[k+A],h[2].mean,h[2].std),Ve(m[k+A],h[3].mean,h[3].std),tr(g[k+A],h[4].median,h[4].iqr),tr(x[k+A],h[5].median,h[5].iqr),Ve(b[k+A],h[6].mean,h[6].std),Ve(w[k+A],h[7].mean,h[7].std)]);C.push(I),$.push((d[k+t]-v)/(v||1));const R=cH(I);C.push(R),$.push((d[k+t]-v)/(v||1)*(.999+Math.random()*.002))}const N=Il(C,[C.length,t,ji]),T=Fo($,[$.length,1]);return{xs:N,ys:T,stats:h}},By=()=>{const n=hE({shape:[Wt,ji]}),t=t0({units:64,returnSequences:!0,recurrentInitializer:"glorotUniform",kernelRegularizer:FE({l2:.001})}).apply(n),e=t0({units:32,returnSequences:!0,recurrentInitializer:"glorotUniform"}).apply(t),s=EE({rate:.25}).apply(e),o=Fi({units:32,activation:"tanh"}).apply(s),r=Fi({units:1,activation:"softmax"}).apply(o),i=RE().apply([s,r]),a=DE().apply(i),l=AE().apply(a),c=Fi({units:32,activation:"relu"}).apply(l),u=Fi({units:16,activation:"relu"}).apply(c),h=Fi({units:1,activation:"linear"}).apply(u),d=uE({inputs:n,outputs:h});return d.compile({optimizer:Xs.adam(.001),loss:"meanSquaredError"}),d},zy=async(n,t,e=null)=>{if(t.prices.length<Wt+10)return null;const{xs:s,ys:o,stats:r}=hH(t,Wt);return await n.fit(s,o,{epochs:e||iH,batchSize:aH,shuffle:!0,verbose:0}),s.dispose(),o.dispose(),{stats:r}},Vy=(n,t,e)=>{const s=t.prices[0],o=[];for(let l=0;l<Wt;l++)o.push([(t.prices[l]-s)/(s||1),Ve(t.rsi[l],e[1].mean,e[1].std),Ve(t.macd[l],e[2].mean,e[2].std),Ve(t.atr[l],e[3].mean,e[3].std),tr(t.roc[l],e[4].median,e[4].iqr),tr(t.vol[l],e[5].median,e[5].iqr),Ve(t.emaRatio[l],e[6].mean,e[6].std),Ve(t.bPercent[l],e[7].mean,e[7].std)]);const r=Il([o],[1,Wt,ji]),i=n.predict(r),a=i.dataSync()[0];return r.dispose(),i.dispose(),a*s+s},dH=async(n,t)=>{const e=tH(n,14),o=eH(n).histogram,r=sH(n,null,n,14),i=oH(n,10),a=rH(n,20),l=Ki(n,8),c=Ki(n,21),u=l.map((B,z)=>B/(c[z]||1)),{upper:h,lower:d}=Py(n,20),p=n.slice(-h.length).map((B,z)=>(B-d[z])/(h[z]-d[z]||1)),f=Math.min(n.length,e.length,o.length,r?.length||0,i?.length||0,a?.length||0,u?.length||0,p?.length||0),m=n.slice(-f),g=e.slice(-f),x=o.slice(-f),b=r.slice(-f),w=i.slice(-f),y=a.slice(-f),C=u.slice(-f),$=p.slice(-f),N=10;if(f<Wt+N+10)return{accuracy:50,hits:{neural:0,pattern:0,technical:0},recommendedWeights:{omega:.5,alpha:.3,gamma:.2},predictions:[]};const T=[];let k={neural:0,pattern:0,technical:0};const v=By(),I=f-N,R={prices:m.slice(0,I),rsi:g.slice(0,I),macd:x.slice(0,I),atr:b.slice(0,I),roc:w.slice(0,I),vol:y.slice(0,I),emaRatio:C.slice(0,I),bPercent:$.slice(0,I)};t&&t(10);const{stats:A}=await zy(v,R,12);t&&t(40);for(let B=0;B<N;B++){const z=I+B,W=prices[z],H=prices[z-1],K=W>H?1:-1,j=V(()=>{const ct={prices:m.slice(z-Wt,z),rsi:g.slice(z-Wt,z),macd:x.slice(z-Wt,z),atr:b.slice(z-Wt,z),roc:w.slice(z-Wt,z),vol:y.slice(z-Wt,z),emaRatio:C.slice(z-Wt,z),bPercent:$.slice(z-Wt,z)};return Vy(v,ct,A)}),X=j>H?1:-1;X===K&&k.neural++;const Y=prices.slice(0,z),tt=nH(Y);let Q=0;tt.sentiment==="Bullish"?Q=1:tt.sentiment==="Bearish"&&(Q=-1),Q===K&&k.pattern++;const ot=g[z-1];let at=0;ot<40?at=1:ot>60&&(at=-1),at===K&&k.technical++,T.push({step:B+1,actual:W,predicted:j,isCorrect:X===K}),V(()=>{const ct=prices[z-Wt],ht=[];for(let Tt=0;Tt<Wt;Tt++){const $t=z-Wt+Tt;ht.push([(m[$t]-ct)/(ct||1),Ve(g[$t],A[1].mean,A[1].std),Ve(x[$t],A[2].mean,A[2].std),Ve(b[$t],A[3].mean,A[3].std),tr(w[$t],A[4].median,A[4].iqr),tr(y[$t],A[5].median,A[5].iqr),Ve(C[$t],A[6].mean,A[6].std),Ve($[$t],A[7].mean,A[7].std)])}const gt=Il([ht],[1,Wt,ji]),xt=Fo([[(W-ct)/(ct||1)]],[1,1]);v.trainOnBatch(gt,xt)}),t&&t(40+(B+1)/N*60)}const P={omega:Math.max(.2,k.neural/(k.neural+k.pattern+k.technical||1)),alpha:Math.max(.15,k.pattern/(k.neural+k.pattern+k.technical||1)),gamma:Math.max(.15,k.technical/(k.neural+k.pattern+k.technical||1))},_=await v.save(Rm(async B=>B));if(_.weightData instanceof ArrayBuffer){const B=new Uint8Array(_.weightData);let z="";for(let W=0;W<B.byteLength;W++)z+=String.fromCharCode(B[W]);_.weightData=btoa(z)}return v.dispose(),{accuracy:(k.neural/N*100).toFixed(1),hits:k,recommendedWeights:P,predictions:T,modelArtifacts:_}};self.onmessage=async n=>{const{type:t,data:e}=n.data;try{if(t==="TRAIN_AND_PREDICT"){const{ticker:s,historicalPrices:o,rsi:r,macdHist:i,atr:a,roc:l,vol:c,emaRatio:u,bPercent:h}=e,d=By(),f=await zy(d,{prices:o,rsi:r,macd:i,atr:a,roc:l,vol:c,emaRatio:u,bPercent:h});if(!f)throw new Error(`Neural core training rejected: Insufficient data (${o.length} bars)`);const m={prices:o.slice(-Wt),rsi:r.slice(-Wt),macd:i.slice(-Wt),atr:a.slice(-Wt),roc:l.slice(-Wt),vol:c.slice(-Wt),emaRatio:u.slice(-Wt),bPercent:h.slice(-Wt)},g=Vy(d,m,f.stats),x=await d.save(Rm(async b=>b));if(x.weightData instanceof ArrayBuffer){const b=new Uint8Array(x.weightData);let w="";for(let y=0;y<b.byteLength;y++)w+=String.fromCharCode(b[y]);x.weightData=btoa(w)}self.postMessage({type:"TRAIN_SUCCESS",data:{predictedPrice:g,stats:f.stats,modelArtifacts:x}}),d.dispose()}else if(t==="ASSESS_ACCURACY"){const{fullPrices:s}=e,o=await dH(s,r=>{self.postMessage({type:"PROGRESS",data:r})});self.postMessage({type:"ASSESS_SUCCESS",data:o})}}catch(s){self.postMessage({type:"ERROR",data:s.message})}}})();
