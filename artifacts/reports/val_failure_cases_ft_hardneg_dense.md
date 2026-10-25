# Validation failure cases: ft_hardneg_dense

> Validation split: model-selection data (not the held-out test).

> Retrieved documents that are not judged positives may still be relevant (qrels are sparse).

**Best-positive rank distribution — ft_hardneg_dense:** {'rank 1': 104, 'rank 2-10': 33, 'rank 11-100': 21, 'not in top 100': 4}

**Best-positive rank distribution — baseline:** {'rank 1': 90, 'rank 2-10': 40, 'rank 11-100': 23, 'not in top 100': 9}

## No judged positive in the top 10 (25 of 162 queries; showing 8)

### Query 0
**Claim:** 0-dimensional biomaterials lack inductive properties.

**Judged positive(s):** [31715818] New opportunities: the use of nanotechnologies to manipulate and track stem cells.
- best-positive rank: ft_hardneg_dense=16, baseline=4  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.431
- ft_hardneg_dense #1 (cos 0.313): [29638116] Complex Tissue and Disease Modeling using hiPSCs.
- ft_hardneg_dense #2 (cos 0.309): [4346436] Nonlinear Elasticity in Biological Gels
- ft_hardneg_dense #3 (cos 0.287): [10786948] An efficient nonviral method to generate integration-free human-induced pluripotent stem cells from cord blood

### Query 1008
**Claim:** RUNX is not expressed in skin tissue.

**Judged positive(s):** [2547636] Human epidermal stem cell function is regulated by circadian oscillations.
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.352): [14105446] Inhibition of limb regeneration in the axolotl after treatment of the skin with actinomycin D.
- ft_hardneg_dense #2 (cos 0.347): [14082855] Inflammatory Reaction as Determinant of Foreign Body Reaction Is an Early and Susceptible Event after Mesh Imp
- ft_hardneg_dense #3 (cos 0.341): [27944135] Silver sulfadiazine for the treatment of partial-thickness burns and venous stasis ulcers.

### Query 1098
**Claim:** Stacking is more stable when a purine is present in the +5 position (C1698).

**Judged positive(s):** [13552682] Decoding Mammalian Ribosome-mRNA States by Translational GTPase Complexes
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.353): [34498325] A conserved modified wobble nucleoside (mcm5s2U) in lysyl-tRNA is required for viability in yeast.
- ft_hardneg_dense #2 (cos 0.334): [7260461] position-specific gap penalties
- ft_hardneg_dense #3 (cos 0.330): [1967017] A Candidate Gene Approach Identifies the TRAF1/C5 Region as a Risk Factor for Rheumatoid Arthritis

### Query 1138
**Claim:** TRAP+ osteoclasts regulate mature osteoblast maintenance.

**Judged positive(s):** [6796297] PDGF-BB secreted by preosteoclasts induces angiogenesis during coupling with osteogenesis
- best-positive rank: ft_hardneg_dense=35, baseline=17  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.468): [39776978] Advances in osteoclast biology: old findings and new insights from mouse models
- ft_hardneg_dense #2 (cos 0.453): [17271462] Tie2/Angiopoietin-1 Signaling Regulates Hematopoietic Stem Cell Quiescence in the Bone Marrow Niche
- ft_hardneg_dense #3 (cos 0.416): [13509809] Matrix-embedded osteocytes regulate mobilization of hematopoietic stem/progenitor cells.

### Query 1300
**Claim:** Thiopurine active metabolites can be catabolized through dephosphorylation of thioguanine nucleotides.

**Judged positive(s):** [6421792] Activating mutations in the NT5C2 nucleotidase gene drive chemotherapy resistance in relapsed ALL
- best-positive rank: ft_hardneg_dense=None, baseline=81  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.440): [32975424] Threonine as a carbon source for Escherichia coli.
- ft_hardneg_dense #2 (cos 0.416): [9257019] Discovery of endogenous catecholamines in lymphocytes and evidence for catecholamine regulation of lymphocyte 
- ft_hardneg_dense #3 (cos 0.416): [5256564] Pharmacological Blockade of ASCT2-dependent Glutamine Transport Leads To Anti-tumor Efficacy in Preclinical Mo

### Query 15
**Claim:** 50% of patients exposed to radiation have activated markers of mesenchymal stem cells.

**Judged positive(s):** [22080671] KLF4-dependent phenotypic modulation of smooth muscle cells has a key role in atherosclerotic plaque pathogene
- best-positive rank: ft_hardneg_dense=39, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.452): [37437064] On the origin and impact of mesenchymal stem cell heterogeneity: new insights and emerging tools for single ce
- ft_hardneg_dense #2 (cos 0.451): [25238950] FGF-2 enhances intestinal stem cell survival and its expression is induced after radiation injury.
- ft_hardneg_dense #3 (cos 0.443): [16630060] Genotoxic Stress Abrogates Renewal of Melanocyte Stem Cells by Triggering Their Differentiation

### Query 211
**Claim:** COPI coatmer is involved in viral replication.

**Judged positive(s):** [13794374] COPI Complex Is a Regulator of Lipid Homeostasis
- best-positive rank: ft_hardneg_dense=None, baseline=None  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.395): [31560225] SIVrcm infection of macaques.
- ft_hardneg_dense #2 (cos 0.393): [44562058] Immune activation and HIV persistence: implications for curative approaches to HIV infection.
- ft_hardneg_dense #3 (cos 0.385): [44737533] Simian immunodeficiency viruses replication dynamics in African non-human primate hosts: common patterns and s

### Query 304
**Claim:** DMS3 protein enables Pol V transcription in vivo.

**Judged positive(s):** [14797520] Noncoding Transcription by RNA Polymerase Pol IVb/Pol V Mediates Transcriptional Silencing of Overlapping and 
- best-positive rank: ft_hardneg_dense=23, baseline=17  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.444): [5137019] Suppressor of cytokine signaling 3 inhibits antiviral IFN-beta signaling to enhance HIV-1 replication in macro
- ft_hardneg_dense #2 (cos 0.427): [30675656] Dishevelled Activates JNK and Discriminates between JNK Pathways in Planar Polarity and wingless Signaling
- ft_hardneg_dense #3 (cos 0.405): [34559336] A gradient of template dependence defines distinct biological roles for family X polymerases in nonhomologous 

## Largest regressions vs baseline (nDCG@10)

### Query 930
**Claim:** Patients with panic anxiety show decreased CSF levels of hypocretin.

**Judged positive(s):** [16056514] A KEY ROLE FOR OREXIN IN PANIC ANXIETY
- best-positive rank: ft_hardneg_dense=11, baseline=2  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.631
- ft_hardneg_dense #1 (cos 0.471): [32357890] Global prevalence of anxiety disorders: a systematic review and meta-regression.
- ft_hardneg_dense #2 (cos 0.404): [2158500] Leptin treatment ameliorates anxiety in ob/ob obese mice.
- ft_hardneg_dense #3 (cos 0.398): [791050] The relation between past exposure to fine particulate air pollution and prevalent anxiety: observational coho

### Query 896
**Claim:** Overexpressing Cnp1 N-tail variants exacerbates the temperature-sensitive growth defect of scm3-139.

**Judged positive(s):** [14338915] Fission Yeast Scm3: A CENP-A Receptor Required for Integrity of Subkinetochore Chromatin
- best-positive rank: ft_hardneg_dense=4, baseline=1  |  nDCG@10: ft_hardneg_dense=0.431, baseline=1.000
- ft_hardneg_dense #1 (cos 0.413): [18662787] NIPBL, encoding a homolog of fungal Scc2-type sister chromatid cohesion proteins and fly Nipped-B, is mutated 
- ft_hardneg_dense #2 (cos 0.388): [4325398] Pancreatic cancer genomes reveal aberrations in axon guidance pathway genes
- ft_hardneg_dense #3 (cos 0.388): [14926162] Identification of an allele of VAM3/SYP22 that confers a semi-dwarf phenotype in Arabidopsis thaliana.

### Query 897
**Claim:** Overexpressing Cnp1 N-tail variants rescues the temperature-sensitive growth defect of scm3-139.

**Judged positive(s):** [14338915] Fission Yeast Scm3: A CENP-A Receptor Required for Integrity of Subkinetochore Chromatin
- best-positive rank: ft_hardneg_dense=4, baseline=1  |  nDCG@10: ft_hardneg_dense=0.431, baseline=1.000
- ft_hardneg_dense #1 (cos 0.413): [18662787] NIPBL, encoding a homolog of fungal Scc2-type sister chromatid cohesion proteins and fly Nipped-B, is mutated 
- ft_hardneg_dense #2 (cos 0.383): [14926162] Identification of an allele of VAM3/SYP22 that confers a semi-dwarf phenotype in Arabidopsis thaliana.
- ft_hardneg_dense #3 (cos 0.378): [4325398] Pancreatic cancer genomes reveal aberrations in axon guidance pathway genes

### Query 726
**Claim:** Ly6C hi monocytes have a higher inflammatory capacity than Ly6C lo monocytes.

**Judged positive(s):** [36444198] Subpopulations of mouse blood monocytes differ in maturation stage and inflammatory response.; [7521113] Fate mapping reveals origins and dynamics of monocytes and tissue macrophages under homeostasis.
- best-positive rank: ft_hardneg_dense=6, baseline=2  |  nDCG@10: ft_hardneg_dense=0.218, baseline=0.651
- ft_hardneg_dense #1 (cos 0.617): [22406695] Origin and functions of tissue macrophages.
- ft_hardneg_dense #2 (cos 0.519): [22621251] Ly6Chi monocytes direct alternatively activated profibrotic macrophage regulation of lung fibrosis.
- ft_hardneg_dense #3 (cos 0.509): [22483580] Disturbed sleep in bipolar disorder is related to an elevation of IL-6 in peripheral monocytes.

### Query 0
**Claim:** 0-dimensional biomaterials lack inductive properties.

**Judged positive(s):** [31715818] New opportunities: the use of nanotechnologies to manipulate and track stem cells.
- best-positive rank: ft_hardneg_dense=16, baseline=4  |  nDCG@10: ft_hardneg_dense=0.000, baseline=0.431
- ft_hardneg_dense #1 (cos 0.313): [29638116] Complex Tissue and Disease Modeling using hiPSCs.
- ft_hardneg_dense #2 (cos 0.309): [4346436] Nonlinear Elasticity in Biological Gels
- ft_hardneg_dense #3 (cos 0.287): [10786948] An efficient nonviral method to generate integration-free human-induced pluripotent stem cells from cord blood

### Query 1165
**Claim:** The G34R/V mutation does not create a hypomethylated phenotype at the H3K36 histone.

**Judged positive(s):** [4455466] ZMYND11 links histone H3.3K36me3 to transcription elongation and tumour suppression
- best-positive rank: ft_hardneg_dense=2, baseline=1  |  nDCG@10: ft_hardneg_dense=0.631, baseline=1.000
- ft_hardneg_dense #1 (cos 0.525): [4312169] Driver mutations in histone H3.3 and chromatin remodelling genes in paediatric glioblastoma
- ft_hardneg_dense #2 (cos 0.512): [4455466] ZMYND11 links histone H3.3K36me3 to transcription elongation and tumour suppression
- ft_hardneg_dense #3 (cos 0.497): [41293601] Hotspot mutations in H3F3A and IDH1 define distinct epigenetic and biological subgroups of glioblastoma.

### Query 1228
**Claim:** The mean suicide rate in women is lower after miscarriage than live birth.

**Judged positive(s):** [25641414] Suicides after pregnancy in Finland, 1987-94: register linkage study.
- best-positive rank: ft_hardneg_dense=2, baseline=1  |  nDCG@10: ft_hardneg_dense=0.631, baseline=1.000
- ft_hardneg_dense #1 (cos 0.635): [881332] Effect of Previous Miscarriage on Depressive Symptoms During Subsequent Pregnancy and Postpartum in the First 
- ft_hardneg_dense #2 (cos 0.620): [25641414] Suicides after pregnancy in Finland, 1987-94: register linkage study.
- ft_hardneg_dense #3 (cos 0.475): [26611834] A meta-analysis of depression during pregnancy and the risk of preterm birth, low birth weight, and intrauteri

### Query 1263
**Claim:** The repeat-variable diresidue (RVD) in the loop of Transcription-Activator Like (TAL) effectors specifies the nucleotides-amino acid contact at the target promoter element.

**Judged positive(s):** [3981729] Structural basis for sequence-specific recognition of DNA by TAL effectors.
- best-positive rank: ft_hardneg_dense=2, baseline=1  |  nDCG@10: ft_hardneg_dense=0.631, baseline=1.000
- ft_hardneg_dense #1 (cos 0.589): [46816158] The crystal structure of TAL effector PthXo1 bound to its DNA target.
- ft_hardneg_dense #2 (cos 0.583): [3981729] Structural basis for sequence-specific recognition of DNA by TAL effectors.
- ft_hardneg_dense #3 (cos 0.506): [3893473] TAL effectors are remote controls for gene activation.

## Largest improvements vs baseline (nDCG@10)

### Query 1330
**Claim:** Tumor development occurs in conjunction with suppression of pro-inflammatory cytokines.

**Judged positive(s):** [14075252] Paraneoplastic thrombocytosis: the secrets of tumor self-promotion.
- best-positive rank: ft_hardneg_dense=1, baseline=None  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.534): [14075252] Paraneoplastic thrombocytosis: the secrets of tumor self-promotion.
- ft_hardneg_dense #2 (cos 0.523): [3419802] Microenvironment-dependent growth of pre-neoplastic and malignant plasma cells in humanized mice
- ft_hardneg_dense #3 (cos 0.513): [25726838] IL-17 promotes tumor development through the induction of tumor promoting microenvironments at tumor sites and

### Query 1158
**Claim:** The DESMOND program demonstrates no significant impact on lifestyles outcomes.

**Judged positive(s):** [12584053] Effectiveness of a diabetes education and self management programme (DESMOND) for people with newly diagnosed 
- best-positive rank: ft_hardneg_dense=1, baseline=77  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.381): [12584053] Effectiveness of a diabetes education and self management programme (DESMOND) for people with newly diagnosed 
- ft_hardneg_dense #2 (cos 0.356): [12794099] Combined Impact of Health Behaviours and Mortality in Men and Women: The EPIC-Norfolk Prospective Population S
- ft_hardneg_dense #3 (cos 0.346): [28806780] Poverty, Hunger, Education, and Residential Status Impact Survival in HIV

### Query 1157
**Claim:** The DESMOND program demonstrates no significant impact on biochemical outcomes.

**Judged positive(s):** [12584053] Effectiveness of a diabetes education and self management programme (DESMOND) for people with newly diagnosed 
- best-positive rank: ft_hardneg_dense=1, baseline=None  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.372): [12584053] Effectiveness of a diabetes education and self management programme (DESMOND) for people with newly diagnosed 
- ft_hardneg_dense #2 (cos 0.351): [16180601] Serum Soluble Corin Deficiency Predicts Major Disability within 3 Months after Acute Stroke
- ft_hardneg_dense #3 (cos 0.343): [5698494] The benefits of statins in people without established cardiovascular disease but with cardiovascular risk fact

### Query 1075
**Claim:** Sequence conservation in gene regulatory regions between species is a poor predictor of conserved functionality.

**Judged positive(s):** [14658685] Enhancer Evolution across 20 Mammalian Species
- best-positive rank: ft_hardneg_dense=1, baseline=35  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.544): [14658685] Enhancer Evolution across 20 Mammalian Species
- ft_hardneg_dense #2 (cos 0.521): [31902059] Large-scale candidate gene scan reveals the role of chemoreceptor genes in host plant specialization and speci
- ft_hardneg_dense #3 (cos 0.516): [45200347] Resequencing studies of nonmodel organisms using closely related reference genomes: optimal experimental desig

### Query 1047
**Claim:** Ribosome-inactivating protein-2 (RIP-2) interacts with the p75 NTR death domain

**Judged positive(s):** [14706752] Gamma-Secretase Represents a Therapeutic Target for the Treatment of Invasive Glioma Mediated by the p75 Neuro
- best-positive rank: ft_hardneg_dense=1, baseline=67  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.000
- ft_hardneg_dense #1 (cos 0.570): [14706752] Gamma-Secretase Represents a Therapeutic Target for the Treatment of Invasive Glioma Mediated by the p75 Neuro
- ft_hardneg_dense #2 (cos 0.481): [368506] Generation of mice with a conditional allele for the p75(NTR) neurotrophin receptor gene.
- ft_hardneg_dense #3 (cos 0.481): [2543135] Autophagic programmed cell death by selective catalase degradation.

### Query 414
**Claim:** Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for Alzheimer's disease.

**Judged positive(s):** [6309659] Reproductive period and risk of dementia in postmenopausal women.
- best-positive rank: ft_hardneg_dense=2, baseline=17  |  nDCG@10: ft_hardneg_dense=0.631, baseline=0.000
- ft_hardneg_dense #1 (cos 0.602): [37205759] Obesity occurring in apolipoprotein E-knockout mice has mild effects on fertility.
- ft_hardneg_dense #2 (cos 0.571): [6309659] Reproductive period and risk of dementia in postmenopausal women.
- ft_hardneg_dense #3 (cos 0.561): [51386222] Effects of Age, Sex, and Ethnicity on the Association Between Apolipoprotein E Genotype and Alzheimer Disease:

### Query 323
**Claim:** Deleting Raptor increases G-CSF levels.

**Judged positive(s):** [2014909] Oncogenic mTOR signaling recruits myeloid-derived suppressor cells to promote tumor initiation
- best-positive rank: ft_hardneg_dense=1, baseline=4  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.431
- ft_hardneg_dense #1 (cos 0.332): [2014909] Oncogenic mTOR signaling recruits myeloid-derived suppressor cells to promote tumor initiation
- ft_hardneg_dense #2 (cos 0.324): [29689140] B-catenin deficiency, but not Myc deletion, suppresses the immediate phenotypes of APC loss in the liver.
- ft_hardneg_dense #3 (cos 0.317): [6962472] G*Power 3: a flexible statistical power analysis program for the social, behavioral, and biomedical sciences.

### Query 223
**Claim:** Cancer cells can stimulate the accumulation of intra-tumoural myeloid-derived suppressor cells by promoting granulocyte colony stimulating factor production.

**Judged positive(s):** [2014909] Oncogenic mTOR signaling recruits myeloid-derived suppressor cells to promote tumor initiation
- best-positive rank: ft_hardneg_dense=1, baseline=4  |  nDCG@10: ft_hardneg_dense=1.000, baseline=0.431
- ft_hardneg_dense #1 (cos 0.632): [2014909] Oncogenic mTOR signaling recruits myeloid-derived suppressor cells to promote tumor initiation
- ft_hardneg_dense #2 (cos 0.555): [19510470] Induction of cell cycle entry eliminates human leukemia stem cells in a mouse model of AML
- ft_hardneg_dense #3 (cos 0.540): [3559136] Cancer-Associated Fibroblasts Neutralize the Anti-tumor Effect of CSF1 Receptor Blockade by Inducing PMN-MDSC 
