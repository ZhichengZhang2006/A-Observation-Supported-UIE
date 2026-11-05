# 独立科学审计：Observation-Supported UIE

日期：2026-11-05。范围：任务书、研究结论、指定实现与已有制品的证据一致性；不是论文录用意见，也不是新的实验结果。

## 0. 审计边界与快照

- 本次只新增本文；不修改实现、测试、评分、数据或既有制品，不启动训练或全量 H3，不操作服务器进程。父代理负责实现修复及后续实验。
- 已完整阅读[任务书](<../AutoResearch_A_Observation_Supported_UIE.md>)、[README](<../README.md>)、[验收报告](<acceptance_report.md>)、[数据说明](<data_manifest.md>)、[测量表](<measurement_tables.md>)，以及指定的 [gauge](<../code/obsui/gauge.py>)、[gauge_error](<../code/tools/gauge_error.py>)、[gauge_pairwise](<../code/tools/gauge_pairwise.py>)、[H2](<../code/tools/h2_incremental.py>)、[mechanism](<../code/tools/mechanism.py>)；为核对结论追读了 [support](<../code/obsui/support.py>)、[metrics](<../code/obsui/metrics.py>) 和少量制品。
- 本文区分三种证据：**源码直接证明、本人执行的轻量反例、读取已有制品**。后者不是本人重新跑出的实验，也不保证原始训练/数据来源真实。
- 共享目录在审计中被父代理持续修改。下文“已修”指看到源码修复并用小数组复核，不代表旧制品已重算。不能把旧 JSON 配上新源码就当作同一版本的证据。
- 本轮轻量检查使用现有 CPU Python，均加 `-B` 禁止写字节码；未创建辅助脚本。两次只读探针曾因错误使用模块函数名/JSON 类型退出 1，修正调用后取得下列结果；没有把失败调用记作通过。

**总裁决：当前证据不支持强版核心主张，也不足以宣告整个研究问题被否证。** 可以保留“简化模型下的条件代数对称性”及“特定冻结输出上的探索性统计”；必须撤回“任意恢复均与实际观测一致”“族外误差即物理不相容”“misfit 归一化完成校准”“H2 已比较普通不确定性并被否证”等表述。

## 1. H1/H2/H3 与验收状态

| 项目 | 本次可支持的最窄结论 | 尚不能说什么 |
| --- | --- | --- |
| H1：低支持对应高恢复风险，超出内容/退化混杂 | 既有 UIEB 89 张的块级 `rho(logq,chroma error)=+0.172981`，控制后的数值 `+0.067585`；方向与“q 越低误差越高”不符。图级相关 `-0.172387,p=0.106219` 则方向符合、证据弱。二者不是同一统计对象 | 不能把负相关称为 H1 错号；不能把正的弱块相关称为已验证支持信息；不能据单模型、单数据、探索后改变目标宣告 H1 普遍成立或普遍失败 |
| H2：超越普通不确定性且可迁移 | 9 项手工统计加支持特征，在当前中心化诊断中色度 ΔR²=`+0.00737888`；绝对误差为 `-0.00373257`。是特定设置下很小的探索性增量 | 没有 heteroscedastic/MC-dropout/ensemble/学习退化对照，没有支持+不确定性对照、跨骨干/退化迁移，也没有 Δ 的图像簇置信区间；**未完成不等于否证** |
| H3：减少具体错误而非平滑/弱增强 | self-orbit 的零步恒等是正确的数学控制，不是实际观测一致性验证；旧四图门控结果存在已确认实现缺陷 | 不能用旧四图结果或非最优 `oracle_g` 宣告所有支持门控无效；不能把 oracle 的 PSNR 增益叫色度改善或物理可实现上界 |
| 条件 gauge 恒等式 | 在指定 tied-coefficient、线性数值模型及固定参数下，变换前后的**模型合成观测**相同 | 不证明模型拟合真实输入、不证明未知参数正确、不证明真实 sRGB 物理可恢复性、不证明新颖性 |

证据：[H1 制品](<../artifacts/h1_error_form_uieb.json>)、[H2 制品](<../artifacts/h2.json>)。H1/H2 的数字仅转录既有制品；支持实现中随后发现/修复的缺陷还要求版本化复算。

[任务书：247–254](<../AutoResearch_A_Observation_Supported_UIE.md#L247-L254>)明确规定状态枚举。当前[验收报告：8–11](<acceptance_report.md#L8-L11>)三处枚举不合法，且把未测性能写成失败。建议替换为：

```yaml
performance_status: unverifiable
novelty_status: unsupported
mechanism_status: unsupported
reproducibility_status: incomplete
```

这里 `unsupported` 只表示“现有证据尚未支撑主张”，**不是已证明该方向不可能**。若后续能独立验证一个有限子主张，另列该子主张的 `partial` 证据，不用代数恒等式替整个 H1/H2/H3 打勾。没有训练新网络本身不是不通过的理由：任务书允许冻结骨干上的最小机制；问题是尚无经过合法六项比较的最终候选。

## 2. 会无效化核心结论的错误与精确替换建议

### F01 · Blocker：self-projection identity 被错误推广为任意恢复与实际观测一致

**位置：**[README：67–87](<../README.md#L67-L87>)、[验收报告：89–108](<acceptance_report.md#L89-L108>)。[当前实现：143–191](<../code/tools/mechanism.py#L143-L191>)的函数头已经正确承认 objective 不读实际观测，但[内部注释：166–170](<../code/tools/mechanism.py#L166-L170>)仍称其为任意一致性修正的 ceiling，前后矛盾。

实际优化的是

`min_dd ||J_base - [b+(J_base-b) exp(beta*dd)]||²`。

当零步在网格内、输出未发生额外 clipping 时，`dd=0` 当然零代价。这既没有使用输入 I，也没有检验 `exists psi: G_psi(J_base)≈I`。正确的实际观测集合应固定 I，并使用事先定义的 C、Psi 和噪声阈值。恒等控制只证明自身属于自身轨道，不证明其属于这个实际集合。

**一像素反证：**固定某通道 `g=1,b=0.2,Jhat=0.1,d>=0`，则任意 `T∈[0,1]` 下 `Ihat=0.2-0.1T∈[0.1,0.2]`，不可能等于实际 `I=0.8`；但 self-orbit 的零步仍零距离。这就足以驳倒“不管什么恢复都一致”的推理。允许换任意水参数不是补救，必须先独立限定 Psi。

**建议将上述两处结论统一替换为：**

> 本实验的 self-orbit projection 未使用实际退化输入；因轨道包含零步，其恒等输出是目标函数的自明性质。该控制不提供实际观测一致性证书，也不限制一般物理约束、损失或门控方法的能力。

内部 `ceiling of any consistency-based correction` 建议替换为 `reference-assisted minimum error over this finite chosen orbit only`。

### F02 · Blocker：一条 gauge 族外不等于完整可行集外；0.217 不是“只有 21.7% 颜色受支持”

**位置：**[验收报告：144–157](<acceptance_report.md#L144-L157>)、[测量表：35–45](<measurement_tables.md#L35-L45>)、[数据说明：91–94](<data_manifest.md#L91-L94>)。实现实际上在 [gauge_error：71–88](<../code/tools/gauge_error.py#L71-L88>)使用**默认水参数与 GT 锚点**，再分别拟合 prediction/raw；并不是“只由观测定义、从不使用参考”。

`frac = 1 - min_dd ||pred - orbit(GT,dd)||² / ||pred-GT||²` 是 RGB 平方误差相对有限一参数族的减少比例。它不是物理支持概率，不是误差的正交分解，不是纯色度误差占比，也不是完整 F(I) 的覆盖率。当前 [best_step：351–386](<../code/obsui/gauge.py#L351-L386>)没有传入深度时不检查深度/辐射边界，仍会给 `admissible=True,range=None`。因此甚至不能把该比例无条件叫作“物理可采纳族解释量”。

**已执行的更强反例（见 §6）：**同一默认水参数，两个像素的场景均为 `.3`、深度均为 `1`；对两个像素分别作 `dd=[+.1,-.1]`。所得第二场景与第一场景的合成观测差**严格为 0**，且值域、深度合理。但全局常量 `best_step` 只能减少 `0.00309783` 的平方误差，其余约 **99.69%** 在这条常量族外。故“族外=观测不相容”有直接反例。

**建议替换验收报告/测量表：**

> 在默认水参数、参考图锚点和有限全局深度步搜索下，这个一参数 RGB 轨道可使 prediction-reference 平方误差平均减少 21.7144%。它是 reference-assisted 描述性拟合，不证明剩余误差物理不相容，也不估计真实输入的完整歧义集合。改变水参数、空间变化深度及模型类可能产生其他可行解释。

**建议替换数据说明：**

> 本诊断明确使用 reference 定义轨道，并用 reference-error 评估拟合；raw 不参与该轨道的物理可行性验证。它只能用于 oracle/解释性分析，不作为单图推理的证据特征或真物理正确性判断。

### F03 · High：gauge 的适用模型与真实 sRGB 被混为一谈

**位置：**[README：13–16、24–27、40–87](<../README.md#L13-L87>)、[gauge：20–38](<../code/obsui/gauge.py#L20-L38>)、[support render：119–125](<../code/obsui/support.py#L119-L125>)；约束来自[任务书：65–86](<../AutoResearch_A_Observation_Supported_UIE.md#L65-L86>)。

恒等式基于 `I=g[b+(J-b) exp(-beta*d)]`，也就是直接衰减和后向散射增长共用 T。任务书明确不允许未经论证地强绑二者。独立系数模型可以另行推导条件变换，但不能直接套用当前公式/数值/边界；真实输入还涉及相机响应、照明、波段积分、剪裁与量化。对 JPEG/PNG RGB 直接除 255 并不等于恢复线性辐射；当前指定流程未见针对这些实测图像的可信逆响应标定。

此外，`J'=b+(J-b)exp(beta*dd)` 一般同时改变亮度和色度。差异 beta 并不使它成为“纯颜色变换”。当 J 低于某个 b 时，正步会降低该通道并可能撞到 0，不能普遍称为更亮/更饱和；灰场景的某些退化也可能保持特定色比。单方向不可识别不能推出“所有颜色都不被约束、所有结构都被约束”。

**建议替换开头强结论：**

> 本工作探索一个简化模型下、依赖参数及场景约束的条件敏感度和 gauge 轨道。该轨道展示部分亮度/色度联合歧义，但不证明真实单幅 sRGB 的所有颜色均不可辨、结构均可辨。真实图像结果是模型依赖的描述性诊断，尚缺受控实拍验证。

将“bit-for-bit”改为“在未剪裁的指定浮点模型中，误差为舍入量级”；没有实拍响应验证前，不把数值 RGB 变化叫精确辐射恢复。

### F04 · High：misfit sigma 的归一化不是独立噪声校准，更不是校准成功证据

**位置：**[README：144–151](<../README.md#L144-L151>)、[support：330–342](<../code/obsui/support.py#L330-L342>)、[support：459–489](<../code/obsui/support.py#L459-L489>)。

代码 `infl=sqrt(chi2_raw/dof); sigma*=infl`，于是同一残差上的 `chi2/dof=1` 是代数恒等式，不能作为模型正确、噪声真实、置信度/覆盖率校准的证据。此尺度还依赖所拟合的 restoration，自然混入恢复误差与模型失配；它不是独立于逆网络误差的观测测量。它可以作为**残差归一化启发式**，但须与传感噪声/模型差异分开表述，独立数据校验覆盖率、风险或检测能力。不能因重新定标出现空间变化，就宣布科学仪器修复。

历史 `10^5–10^7` 的“chi2/dof”也不能原样沿用：现[诊断脚本：78–83](<../code/tools/diag_support_spatial.py#L78-L83>)用 `fit.get('dof',1)`；旧 fit 缺该键时会打印总 chi2，而不是除自由度的量。本文没有恢复每一个旧运行时版本，所以不把历史所有大值一概判定为该 bug，但这些数字**未完成版本/分母复核**，不能据其量级声称“水模型只解释百分之几方差”。chi2 本身也不是解释方差 R²。

**建议替换：**

> misfit 模式用同次拟合残差重新定标 sigma，使该拟合的归一化残差统计量为 1；这是定义而非独立校准结果。其信息量只在所选残差尺度与模型假设下解释，不视为真实传感噪声 Fisher 信息或可信覆盖概率。旧 chi2 数值需连同源码版本、自由度和原始残差复核。

## 3. H2：不是普通不确定性对照，也尚不是可部署风险预测

### F05 · High：比较对象不覆盖任务书 H2，缺失实验被写成否证

[H2：66–69](<../code/tools/h2_incremental.py#L66-L69>)的 F0 是亮度、色偏、梯度、Laplace 噪声、饱和、熵、tau 等 9 项手工特征，不是异方差风险、MC dropout、小 ensemble，也不是学习到的退化预测器。没有“普通不确定性 + support”嵌套对照和第二骨干/独立退化迁移。

因此[验收报告：55–57](<acceptance_report.md#L55-L57>)“与 off-the-shelf uncertainty map 不可分，判据 #5 已失败”没有对应实验。现有同一制品不能替代[任务书：110–118](<../AutoResearch_A_Observation_Supported_UIE.md#L110-L118>)要求的比较。

**建议替换：**

> 已比较的是 9 项手工内容/退化统计，而非普通学习不确定性。当前设置下色度 R² 的点估计增量为 +0.00738；其统计与实用意义尚未由图像簇置信区间或决策效果确认。普通不确定性、联合模型与跨骨干/退化迁移实验尚未完成，因此 H2 总体尚不受支持，不能记为被这些缺失对照否证。

“样本多所以 +0.7 个百分点在噪声内”不是统计检验。应按图像/场景配对重采样并报告 Δ 区间；若要主张等效/无实用增益，须预先定义等效区间并拥有足够精度。不得在看见结果后新增[验收报告：205–214](<acceptance_report.md#L205-L214>) `ΔR²>0.05` 或“必须超过 oracle 族内增益”的硬门槛。[任务书：241–245](<../AutoResearch_A_Observation_Supported_UIE.md#L241-L245>)未授权这些附加性能门槛。

### F06 · High：中心化 CV 只支持相对位置诊断；五项全局支持特征被完全消去

**位置：**[H2：167–174](<../code/tools/h2_incremental.py#L167-L174>)、[H2：218–256](<../code/tools/h2_incremental.py#L218-L256>)、[H2：308–349](<../code/tools/h2_incremental.py#L308-L349>)。

1. 全局 `lq_colour/lq_band/lq_pixel/lev_colour/qeff_ratio` 在每幅图所有块中为常数，按图中心化后全变为 0。所谓“加 7 项支持特征”的中心化结果实际上只检验最后两项局部特征（当前为红通道块均值，名称 `min/max` 也不能准确描述）；不能据此否定前五项的图级作用。
2. `yc[m]-=yc[m].mean()` 对留出图像使用了其 reference-error 均值。作为研究者评价“图内误差偏离”的诊断可以成立，**不是自动的训练标签泄漏**；但部署时这个真均值未知，不能据该 R² 给绝对误差、事件概率、跨图 risk–coverage 或阈值安全承诺。
3. 特征图内中心化可以用当前图的全部输入特征实现；标签中心化不能据此声称可部署。本人小数组检验给每图误差额外加上任意大常数后，中心化 R² 完全不变，说明它根本不检验绝对图级风险。
4. X 在划分 fold 之前用全部图像均值/方差标准化；这是使用测试协变量的 transductive 预处理，不是 reference 标签直接进训练，但会改变 ridge 正则含义。不能仅凭 fold 大便宣布无影响。应在每个训练 fold 拟合预处理，再应用到留出 fold。
5. “block 特征不可能预测留出图像的平均风险”没有理论根据；它们可能包含可迁移的退化/内容信息。pooled R² 应作为另一目标如实报告，不能因结果差就改口为该目标不可能。

**建议替换中心化结果标题/说明：**

> 本表为 reference-centred、按图留出的相对块误差诊断；五项图级支持特征在中心化后被消去。它不评价部署时绝对风险或跨图可靠性，pooled 与 centred 结果回答不同问题，均为探索性结果。预处理尚需改为训练 fold 内拟合。

### F07 · High：H1 正负号与统计层级解释错误

[验收报告：78–85](<acceptance_report.md#L78-L85>)把 `rho=-0.172` 称为“low support→high error 的错号”，恰好相反。若 q 是支持而不是风险，低 q 高误差预期就是负相关。真正相反的是[测量表：17–20](<measurement_tables.md#L17-L20>)的正块级相关。图级与块级可以因聚合、混杂或估计不同而异号，不能挑一个符号为总否证。

[测量表：22–30](<measurement_tables.md#L22-L30>)“partial 仍正所以不是退化重复、足够证明携带信息”也过度：偏相关不排除非线性混杂、估计器偏差、选择后推断或 reference-style 差异；当前符号还与 H1 方向不符。把相对/色度称为唯一“honest targets”掩盖了它们在初始结果之后探索的时间线。

**建议统一替换：**

> 该探索中，块级正相关不支持 H1 的预定方向；图级负相关方向符合但区间/检验不足以得出稳定结论。不同层级、误差定义及事后控制结果分开报告，不能互相替代。尚需预先固定事件、以独立场景为统计单位的验证。

## 4. 实现缺陷：已修项必须与旧结果断开，未修项不能带入下一轮

### F08 · High：方向信息和基底定义污染了“颜色 vs 结构”比较

**审计时已修：**旧 [support direction_info](<../code/obsui/support.py#L569-L582>)曾用 `(T*v/sigma)`，漏掉 renderer 的 gain；而 [analyse：546](<../code/obsui/support.py#L539-L553>)用 `(gain*T/sigma)^2`。单位 T/sigma、红通道 gain=2 时，旧方向函数给 q=1，实际 q=4。父代理现已改为 `r=sqrt(an['q'])*v`，本人复核给 q=4。该修复影响历史全局方向特征；旧 H2 不因此自动变成新实现的结果。

**尚见于复核快照：**[components](<../code/obsui/support.py#L908-L924>)将 9 列平滑 basis 减去列均值后仍 QR 并保留 9 列，去 DC 的矩阵实际秩为 8。本人在 16×16 上测得低频方向与 DC 最大内积 **0.64595965**；至少有一个任意补全的 QR 列，并不满足所声称“正交去 DC 低频方向”的语义。

**建议实现方向（本次未修改）：**用有容差的 SVD/秩揭示 QR，按非零奇异值保留真实秩，显式验证每个 low 与 DC 正交；记录实际方向数量。不要以 `range(k_low)` 强取秩亏列。对 nuisance 投影也优先验证 SVD/pseudoinverse 和 ridge 敏感性：ridge hat matrix 不是精确幂等正交投影，“exact absorption”须改为“正则化局部线性 nuisance 吸收指标”。这不是要求把 q 变成概率。

### F09 · High：可采纳边界和 PSNR 极值旧 bug 已修，但旧制品不可继续作为科学证据

**初读实现缺陷及本人反例：**

- 旧 `colour_budget` 忽略 J<b 通道的辐射下界。`J=[[.05,.4,.4]],d=[1],dd=1` 被标为 admissible，却生成 R=`-0.05874553`。
- 旧 `admissible_range` 错把负信号的约束放在负步一侧；每通道约束粗略用 beta 最大/最小值；max_tau 上限用最小深度。`J=.2,d=[.1,3]` 给 hi=`1.90189869`，最大 optical depth=`5.66306309`，超过声明的 3.5。
- 旧 `family_spread` 用 `argmin(PSNR)` 选 best、最大值作 worst，导致 `psnr_gain_max=0` 是算法方向错误，不是“免费增益实测为零”。中途修订又短暂保留 `worst-best` 的 spread；父代理随后一并修复。

**复核看到的修复：**[边界：284–306](<../code/obsui/gauge.py#L284-L306>)逐像素/通道推导上界且用 d.max；[colour_budget：170–178](<../code/obsui/gauge.py#L170-L178>)共享该边界；[spread：465–485](<../code/obsui/gauge.py#L465-L485>)显式包含零步、argmax/argmin 正确。反例现在分别得到 `admissible=False`、截断步 `.59998224` 且 R=0；tau 界 hi=`.02956989`。

本人构造目标为轨道 dd=.1 的例子，初版 spread 报 gain=0；修复版正确选 `.1`，报告 base PSNR=`35.84800808`、best=`99`（实现的 identity cap）、gain=`63.15199192`。这只是定位 bug 的极简 oracle 例子，不是 UIEB 提升或可部署方法。

**仍须撤回：**[验收报告：159–165](<acceptance_report.md#L159-L165>)“free gain≈0、带宽窄”的结论，以及[旧 gauge_pairwise 制品](<../artifacts/gauge_pairwise.json>)中 89 图全零 gain 的物理解读。修好极值不解决 F01/F02：其 [调用：92–96](<../code/tools/gauge_pairwise.py#L92-L96>)用 `(raw,GT)` 的 fit.depth，却拿另一个 default water 滑动方法输出；没有证明此输出与该真实观测成有效模型配对。oracle 在 reference 上选步，也绝不等于“无需观测外信息即可得到”的提升。

**建议替换：**

> 旧 family-spread 极值方向错误，历史零增益与相应带宽结论撤回。修复后只报告有限、reference-assisted 轨道搜索统计；在未验证基点实际观测拟合与完整可采纳约束前，不称其物理歧义带或一般方法上界。

### F10 · High：旧 H3 结果已知无效；当前局部修复不能倒推新结论

[旧 H3 制品](<../artifacts/h3_mechanism.json>)只有 n=4，缺少 v2/源码 hash；[后续协议：16–33](<continuation_protocol_20261105.md#L16-L33>)明确旧 chroma inverse 漏 EPS 导致零编辑也改变输出，旧四图保留但不作机制证据。父代理已在[当前机制：121–136](<../code/tools/mechanism.py#L121-L136>)修复逆变换；[实现核对制品](<../artifacts/implementation_checks_v2.json>)显示小数组 roundtrip 约 3e-16–6e-16。该核对只证明代数实现，不证明增强机制有效。

因此[验收报告：112–126](<acceptance_report.md#L112-L126>)对旧门控近似相同的“锚点问题”解释并未隔离实现缺陷，不能当作稳定负面科学发现。即便使用 reference，`oracle_g` 也只是误差加权 gate，不是给定修改方向下的最优 gate；它变差不提供 H3 上界。oracle projection 在旧四图中 PSNR `+2.037 dB`，色度误差从 `.080028` **升至** `.081767`（约 +2.17%，更坏），主要改善亮度，不能写成色度改善。

**建议替换：**

> 旧四图机制诊断因已确认的 chroma 逆变换缺陷失效，仅保留作错误追踪。修复实现的结果须以独立版本、逐图结果和源码哈希另报；本报告不预测父代理当前重跑的结果。reference-weighted gate 与 reference-selected orbit 都非部署方法，亦非 H3 一般上界。

## 5. 数据、协议、污染与历史 test

### F11 · High：参考重复与真实训练污染的证据层级被混用

旧[数据说明：44–51](<data_manifest.md#L44-L51>)有算术/术语错误：3423-3399=24 是重复**多余文件数**，不等于 24 个重复 hash 组；13×1+1×3=16，不是声称的 17。旧[UFO：61–70](<data_manifest.md#L61-L70>)也与[本地初筛：847–857](<../_ops/audit.json#L847-L857>) `exact_dup=1` 不符。该初筛只审了本地不完整副本，不能写成全来自服务器。[数据说明：74–81](<data_manifest.md#L74-L81>)“17 图已用于训练、所有方法受相同乐观偏差”均无足够证据。

**审计后段新到达的更强证据必须纳入，而不能停留在“只见 GT”：**已读取[服务器来源清单](<../artifacts/provenance/pairing_and_duplicates.json>)，其记录如下。本人只读取该制品及摘要，未重新读取全服务器图像以独立验证 hashes。

| 数据/交叉 split | held-out input 文件匹配数 | held-out GT 文件匹配数 | 完全相同 paired sample 数 |
| --- | ---: | ---: | ---: |
| UIEB train/val、train/test、val/test | 0 | 0 | 0 |
| LSUI train/val | 12 | 14 | 10 |
| LSUI train/test | 10 | 17 | 9 |
| LSUI val/test | 2 | 5 | 2 |
| UFO train/val | 1 | 1 | 1 |
| UFO train/test | 3 | 3 | 3 |
| UFO val/test | 2 | 2 | 2 |

这足以指出当前目录 split 存在输入和 paired 层面的重复风险，而不仅仅 reference 重复；但仍要区分 **目录中可获得** 与 **某 checkpoint 真实训练消费过**。还缺各权重训练清单/采样历史，不能把 17 个 GT 匹配全计成 17 个输入泄漏，也不能推断各方法偏差相同。[checkpoint 清单](<../artifacts/provenance/checkpoints.json>)本身标注来源和训练 provenance 未核实；其中 CCL 的两个路径被拼成一个字符串而 `exists=false`，不能据此断言两个文件都不存在，须逐文件核对。

**建议替换：**

> 哈希清单分别报告 input-only、GT-only 和 paired 重复。当前服务器目录中 LSUI train/test 有 10 个 held-out input 匹配、17 个 GT 匹配、9 个完整 paired 匹配；UFO train/test 为 3/3/3。是否构成各 checkpoint 的训练污染仍需其训练清单核实。未知污染保留为风险，不假定所有方法偏差相同。主协议结果与去重复敏感性分析分开，不事后删改标准测试集。

byte-hash 零匹配也不排除重编码、裁剪、连拍/序列或场景重复。有限 dHash 首 400×200 比较不是完整感知去重。UIEB 当前目录内零 byte 匹配不证明官方 canonical split 清洁。

### F12 · Blocker（性能验收）：未建立完整可比集合，saved-tile replay 不是官方权重统一重跑

**位置：**[README：106–133](<../README.md#L106-L133>)、[验收报告：32–43](<acceptance_report.md#L32-L43>)、[数据说明：21–32](<data_manifest.md#L21-L32>)；资格要求见[任务书：136–144](<../AutoResearch_A_Observation_Supported_UIE.md#L136-L144>)及[239–245](<../AutoResearch_A_Observation_Supported_UIE.md#L239-L245>)。

- UIEB 89 张不是凭 n 就能叫 UIEB-R90“少了一张”；须与官方清单比较缺失/替代项。LSUI 3423/428/428 也不是凭文件数可认定任务书主版本。此次不授权补图、重分割或回填分数。
- [verify_mambacue：21–42](<../code/tools/verify_mambacue.py#L21-L42>)只读保存的 raw|GT|pred 图片，没有载入 checkpoint 做推理，也不把每块统一重采样。其与历史 result.txt 接近只验证该 replay 与历史记录近似一致，不能推断原始推理预处理、训练访问或官方 provenance 相同。
- [pack_mambacue：3–10](<../code/tools/pack_mambacue.py#L3-L10>)自身承认 UFO tile 是 240×960，即每块 240×320；[verify_mambacue：3–5](<../code/tools/verify_mambacue.py#L3-L5>)还写了 UFO 的不同预处理路径。这不能进入任务书“整图直缩 256×256、不裁剪”的主表。
- [metrics：24–33](<../code/obsui/metrics.py#L24-L33>)实际为短边 256 保纵横比；本人 `240×320` 探针返回 `256×341`，非 `256×256`。其 Gaussian11 还使用 reflect 边界，而文档称 valid。[mechanism：223–239](<../code/tools/mechanism.py#L223-L239>)虽已把主 SSIM 换 win5，仍调用该 resize。方图上不显错不代表 UFO 路径合法。任务书没有硬指定 win5/11，但要求统一、预锁、可复核；这里是实现/文档/轨道不一致，不是建议事后挑更高分实现。
- 原先“七个 official-weight”只证明机器上现有 checkpoint 被运行，缺发布页原始下载/哈希对应和训练 split。最近[registry](<sota_registry.csv>)已将来源降为 `local_checkpoint_rerun_provenance_unverified`，MambaCue 也改为 `saved_tiles_only_not_qualified_unified_rerun`，这是正确收窄；但 MambaCue notes 后半仍残留 `FROZEN SOTA REFERENCE`，与前半矛盾，README/验收尚未同步。
- 当前 registry 已出现全部七方法三数据集本地重跑行，旧验收报告仍称 LSUI/UFO 在跑，属陈旧报告；完整列数本身不解决 canonical/provenance/comparability。
- S_m 是**各数据集各指标**的可比集最优，不必同一方法。[新 registry](<sota_registry.csv>)中 Phaseformer LSUI/UFO 已高于 MambaCue 历史 replay PSNR；不能固定用 MambaCue 三数据集作为最优。也不能拿四舍五入的 CSV 值做最终 5% 判定。

**建议替换协议/性能声明：**

> 当前结果分为来源未核实的本地 checkpoint 重跑、历史保存输出 replay、以及尚未完成的同协议重训练。MambaCue replay 不具备统一重跑资格；现有数据版本和权重训练访问尚未完成 canonical/污染核验。因此六项性能状态为 unverifiable，尚不计算正式 SOTA 通过/失败。待确定可比范围后逐数据集/指标取未舍入最优值，并明确未覆盖最新方法。

不因为“某方法本机没有权重”就声称官方不存在；只能写本次未获取/未核实。不能因现存结果比 workbook 更高/更低，就反推必由 crop/resize 引起或宣布第三方结果错误；原因还需原代码与权重/图像路径的可重复对照。

### F13 · High：历史 test 已探索，不能回写成预注册或盲测

H1 的误差定义、加入 tau、噪声模式及 H2 的中心化处理都已受同一 UIEB 89 test 结果启发；源码注释和既有结果足以看出这种探索。分组 CV 防止同幅块进训练/测试，但不能抹除研究者已据这批图像结果选特征/目标的适应性偏差。

[验收报告：55–57](<acceptance_report.md#L55-L57>)“fixed ... in advance”须提供带 commit/hash/时间的先验注册记录；没有则改为事后整理的判据。当前新增[后续协议：1–12](<continuation_protocol_20261105.md#L1-L12>)已明确“post-exploration”，应让所有主文同步。保留探索不利结果，不擅自重置测试名义；确需下一轮独立验证时按任务书另立未使用场景/退化或清楚披露，不能把旧 test 改名 dev 后仍把同图称盲 test。

**建议替换：**

> 这些分析使用已反复探索的 UIEB 89-image split；特征、噪声与目标定义存在观察结果后的调整，因此所有当前 H1/H2/H3 统计均为探索性。后续协议仅约束其登记后的新运行，不能追溯赋予旧结果预注册或盲测资格。

## 6. 本人已执行的轻量复核与复现方式

在仓库根目录，用具有 numpy/scipy 的现有 Python 运行下列片段即可；使用 `python -B` 不产生字节码文件。以下都不读大数据、不写结果、不运行完整研究脚本。

### 6.1 自身轨道恒等与同观测但常量轨道外反例

```python
import sys
import numpy as np
sys.path.insert(0, 'code')
from obsui import gauge as G, support as S
from tools.mechanism import project_onto_family
w = S.default_water()
a = np.full((2, 2, 3), .3)
p = project_onto_family(a, w.beta, w.b_inf)
print(p['dd'], np.max(abs(p['img'] - a)))  # 0.0, 0.0
j = np.full((2, 3), .3)
d = np.ones(2)
dd = np.array([.1, -.1])
j2, d2 = G.gauge_pair(j, d, w, dd)
print(G.verify_exact(j, d, w, dd))
print(G.best_step(j, j2, w))
```

实测 `max_abs_diff=0`；`frac=0.0030978302521589285`、`residual=0.0014003271542562107`、`total=0.0014046786101492917`。它反驳的是从有限族外推完整物理集合的逻辑，不是实拍数据结论。

### 6.2 misfit 的恒等归一化与中心化无法识别图级偏置

```python
import numpy as np
r = np.array([1., 2., 3.]); s = np.array([.1, .2, .3]); nu = 2.
snew = s * np.sqrt(((r/s)**2).sum()/nu)
print(((r/snew)**2).sum()/nu)  # 1.0 by construction
import sys
sys.path.insert(0, 'code')
from tools.h2_incremental import _grouped_cv
G = np.repeat(np.arange(10), 8)
X = np.arange(80.).reshape(-1, 1) % 8
y = X[:, 0] + G*100.
print(_grouped_cv(X, y, G), _grouped_cv(X, y+G*10000., G))
```

实测两次 R² 均 `0.9999999991142817`。这不指控 CV 训练直接用了留出标签，而是证明该评价对图级绝对风险偏移完全不敏感。

### 6.3 去 DC 基底的秩与任意补全列

```python
import sys, numpy as np
sys.path.insert(0, 'code')
from obsui import support as S
phi = S.depth_basis(16, 16, 9)
phi -= phi.mean(0)
print(np.linalg.matrix_rank(phi), phi.shape[1])
c = S.components(16, 16)
print(max(abs((v*c['dc_0']).sum())
          for k, v in c.items() if k.startswith('low_0')))
```

实测 `8,9` 与 `0.6459596501112235`。依赖 BLAS 的任意补全列可能不同；正确要求是只保留秩内列并与 DC 正交，而非固定复现此错误数值。

### 6.4 父代理修复后的边界、gain 与 spread 小数组核对

```python
import sys, numpy as np
sys.path.insert(0, 'code')
from obsui import gauge as G, support as S
w = S.default_water()
j = np.array([[.05,.4,.4]]); d = np.array([1.])
c = G.colour_budget(j, d, w, 1.)
print(c['admissible'], c['dd_used'])
print(G.gauge_pair(j, d, w, np.array([c['dd_used']]))[0])
print(G.admissible_range(np.full((2,3),.2), np.array([.1,3.]), w))
a = np.full((2,2,3), .3)
t = w.b_inf + (a-w.b_inf)*np.exp(w.beta*.1)
print(G.family_spread(a,t,w))
an = {'q':np.array([[4.,1.,1.]]), 'T':np.ones((1,3)),
      'sigma':np.ones((1,3)), 'A':np.zeros((3,1)),
      'cov':{'invG':np.eye(1)}}
print(S.direction_info(an, [[1,0,0]]))
```

修复后实测：`False,.5999822369362968`；R 截断到 0；range `(-.1,.02956989247311892)`；spread 选 dd≈`.1`、best99、worst14.8407494161、band84.1592505839；q=4。这些检查通过仅说明这几个实现反例得到修复，**不认可默认水参数为真实水体，也不验证历史 artifacts**。

### 6.5 只读抽取已有制品，不重跑研究

```python
import json
h = json.load(open('artifacts/h2.json'))
print(h['n_images'], h['n_blocks'], h['targets'])
d = json.load(open('artifacts/provenance/pairing_and_duplicates.json'))
print([(r['dataset'],r['splits'],r['input']['n_heldout_files'],
        r['GT']['n_heldout_files'],len(r['identical_paired_samples']))
       for r in d['duplicates']])
```

这些读取输出对应 §1、§5；无需再次使用 test 图像跑新估计器以核对报告数字。

## 7. 任务书未完成要求与建议处置顺序

以下是“尚未满足/尚未获足够证据”，不是把没做的实验记成失败，也不授权本审计启动它们。

| 要求 | 当前证据缺口 | 处置 |
| --- | --- | --- |
| 独立观测量与参数不确定性；线性/sRGB、独立衰减/散射 | tied 模型、restoration-dependent misfit、方向/基底问题；真实参数集未独立界定 | 先修定义与版本，保留简化条件结论；模型失配单列 |
| H1 控制内容、真实信息改变、独立检测 | 当前真实图相关弱/异号且探索；新 controlled-v2 制品不在本次深入审计范围 | 不预判新实验成功；区分条件仪器验证与 enhancer-error H1 |
| H2 超越普通不确定性并跨骨干/退化 | 手工 F0 不足，五个全局特征被消去，预处理不在 fold 内 | 明确未完成；不要用当前 CV 宣告对不确定性等效 |
| H3 特定色彩/结构错误、匹配质量与真实细节 | 旧四图无效，oracle 非最优上界；缺普通不确定性 gate、真实边缘 recall/false-edge、质量/强度 Pareto | 父代理接手，本文不重复实现/跑 H3 |
| PUIE-Net/BEM、强可复现 SOTA 与 canonical 数据 | 最新可比集/训练访问和官方权重未核实；replay 混入主表 | 分开 published / local rerun / retrain / replay，逐项资格审核 |
| 独立 dev/calibration、冻结/blind、≥3 决定性 seeds、图像级不确定性 | 历史 test 已探索；尚无完整最终候选及决定性多 seed | 新版本标 post-exploration，完整披露，不重写历史 |
| 六项 ≤5% 性能门槛、MSE/LPIPS/风险/效率 | 正式比较集与最终候选未齐；不能用 replay 当冻结 SOTA | `performance_status=unverifiable`，不增减原门槛 |
| 可复现交付包、ledger、环境锁/成本/失败与归档 | 现制品多未绑定运行时源码，缺完整训练/数据来源和全过程成本；本地 git 存在大量未提交文件，推送未由本审计验证 | `reproducibility_status=incomplete`；归档由父代理处理 |

推荐仅三步：

1. **立即撤回错误推论并同步正文/生成器。** F01/F02/F04/F07/F12 不等待重实验即可修正；保留历史结果并明确无效范围，不能擦掉失败记录。
2. **版本化修仪器与协议。** 已修 gain/边界/PSNR 极值还需解决秩亏基底、统一指标路径与 provenance；旧 JSON 不可背书新代码。冻结后再由父代理按后续协议重跑必要实验，不在旧 test 上继续挑有利指标。
3. **按原任务完成真正缺失的证据。** 普通不确定性、迁移、匹配质量/细节、最终六项比较与多 seed 未完成则如实交接，不把缺失改成“理论证明不必做”。

## 8. 复核版本记录

初读时 [gauge](<../code/obsui/gauge.py>) SHA256 为 `0f9a365b75fb2137b62af44f978eaacc3d43db7b764b8aa2413349a0e461e4b7`；边界/PSNR 反例已通知父代理。父代理修复后，本人复核的 SHA256 为 `9091765ce9f5dfebcd5b610763a69308cafa305232d9afd847230a67f7420776`。[support](<../code/obsui/support.py>)初读 SHA256 为 `02a86aead42f5b3795a7295cc5afa6ddc8c2e4f4f5cc6ca1687164e427ca5dfe`，gain 修复后为 `9ab6ef590e4d7c0b39ca3d3bcd47262b07d0a1f0f84f0e06677b74e011c2877e`。这些修复来自父代理，不是本审计修改。

其余核心快照：

| 文件 | SHA256 |
| --- | --- |
| [任务书](<../AutoResearch_A_Observation_Supported_UIE.md>) | `cc8a00da5f2f8c8f03c9b01295581b2f4e00926b94fd0ccaea7221feaa53689e` |
| [README](<../README.md>) | `6be6e2b643843f5661b7da65bc119f4a67b1c53a28972c7536b3996f08c96a58` |
| [验收报告](<acceptance_report.md>) | `9300895324641c8831a8439c04702a289676ad974929ddf4904afb1486dbc79f` |
| [数据说明](<data_manifest.md>) | `8cea1835556db4cadef1552608859735f41e9c6277bee53f2c3b7a6a64f01c40` |
| [测量表](<measurement_tables.md>) | `7c811bffe506d094fb754333b0233cf7a84750977fb6b7dce9c1c18b5e564033` |
| [gauge_error](<../code/tools/gauge_error.py>) | `8a94d860055088e46d13444fbbde5bc8842fadd87e6b2d3a5962bf6e21bc52d1` |
| [gauge_pairwise](<../code/tools/gauge_pairwise.py>) | `4044cc616da410b195567ece06c0c5493787c519f0b4499c504e4341b54ae826` |
| [H2](<../code/tools/h2_incremental.py>) | `204826abc282edbcd72b45e39b0e86af4262fd4d963727658641ca1d4214b3a2` |
| [mechanism](<../code/tools/mechanism.py>) | `84a98916add9c41cb3cc6e6c5e323d0a407f869c8e199c2709a6a006f72b43cc` |
| [H1 制品](<../artifacts/h1_error_form_uieb.json>) | `7a5c6df391eb9f3eddb42458aad8a09232b4228d32d2635612d98f7a53925e8d` |
| [H2 制品](<../artifacts/h2.json>) | `e0c7418a1d0b8b6f9fbd044a5ed058417e72a531cfeba24293af6399c29fe9b7` |
| [旧 H3 制品](<../artifacts/h3_mechanism.json>) | `e3f354e59a78930c019013c7c7c2031550d9c402795acdfc7dfc75854aa02226` |
| [旧 pairwise 制品](<../artifacts/gauge_pairwise.json>) | `b55304e4fd568eb2164282a914080896a8b32652c9968e7fb392226eba6ab8b1` |

**最终口径：**有一个值得保留的条件恒等式、一组有用但尚不足的探索性结果，以及若干已定位的仪器/解释错误。既不能将它包装成“证明颜色恢复均不可能”的理论成果，也不能因原机制尚未成立就把用户要求的后续研究悄然取消。下一步依据原验收推进，结论强度始终不超过实际证据。
