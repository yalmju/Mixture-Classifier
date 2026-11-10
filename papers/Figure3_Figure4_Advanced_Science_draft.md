# Draft manuscript text for Figures 3 and 4

## Working title

Physics-guided learning resolves nonlinear mixture information in spatially encoded surface-enhanced Raman scattering

## Scope and writing convention

This draft is written for a manuscript aimed at *Advanced Science*. Abbreviated technical names are avoided in the running text. Diquat, thiabendazole, and thiram are written in full rather than as letter codes. Surface-enhanced Raman scattering, random forest classification, non-negative least-squares regression, multilayer perceptron, principal component analysis, variable importance in projection, and explainable artificial intelligence are likewise written in full. Compact labels may still be used inside figures if space requires them.

Values that are directly supported by the current figure or analysis archive are stated as numbers. Experimental details that are not available in the repository are enclosed in square brackets and must be replaced before submission. The current Figure 4 may change; its text is therefore modular and can be rearranged without altering the central argument.

---

# Results and Discussion

## Spatially written single-component patterns establish the analytical basis of the platform

We first asked whether the surface-enhanced Raman scattering ink could preserve both the spatial form of a written pattern and the molecular identity of the deposited analyte. Diquat, thiabendazole, and thiram were separately incorporated into the ink and written on the surface as the letter patterns “DQ,” “TBZ,” and “THI,” respectively. Optical photographs retained the intended macroscopic patterns, whereas Raman mapping converted the visually similar marks into chemically specific images. Intensity maps reconstructed at 1570 inverse centimetres for diquat, 1270 inverse centimetres for thiabendazole, and 1370 inverse centimetres for thiram reproduced the corresponding letters with limited signal outside the written regions (Figure 3a). The agreement between the optical marks and the Raman maps demonstrates that the ink functions simultaneously as a deposition medium and a spatially addressable sensing interface.

The mean spectra extracted from the written regions contained the characteristic vibrational features of the corresponding compounds (Figure 3b). Diquat showed prominent bands in the vicinity of 1174 and 1574 inverse centimetres, thiabendazole was distinguished by bands near 780, 1011, and 1270 inverse centimetres, and thiram produced strong responses near 551 and 1364 inverse centimetres. The exact assignments are provided in Table [Sx] and should be supported by reference spectra and literature assignments. Importantly, the chemically informative peaks were retained after incorporation into the ink and spatial deposition. This observation excludes a trivial imaging explanation based solely on ink thickness or total scattering intensity: the maps can be generated from compound-selective bands rather than from an undifferentiated brightness contrast.

To quantify the discriminating information, six representative Raman-shift regions were compared across the blank ink and the three analytes (Figure 3c). Each material produced a distinct multiband response rather than a single isolated diagnostic peak. Thiram dominated the response near 551 and 1364 inverse centimetres, thiabendazole contributed strongly near 780 and 1011 inverse centimetres, and diquat was most clearly distinguished near 1174 and 1574 inverse centimetres. This distributed spectral contrast is advantageous for classification because the decision does not depend on one potentially unstable peak. It also motivates the use of the full spectrum while retaining band-level inspection as a chemical validity check.

Unsupervised projection of the pixel spectra further showed that the three analytes and the blank occupied distinct regions of spectral space (Figure 3d). The first and second principal components accounted for 36.8 and 33.5 percent of the total variance, respectively, and together separated the analyte clusters from the blank and from one another. The separation was especially clear for thiram and for the blank, while diquat and thiabendazole occupied neighboring but distinguishable manifolds. These results show that the written patterns contain sufficient multivariate information for automated single-component recognition.

A random forest classifier was therefore used as a deliberately simple nonlinear benchmark. The purpose of this model was not to claim that a complex architecture was required, but to establish that single-component identity can be read robustly from pixel-level spectra using a conventional supervised classifier. The classifier was trained using spatially or batch-separated spectra, rather than a random split of neighboring pixels, to avoid inflating performance through spatial autocorrelation. Its high held-out classification performance [insert accuracy, macro-averaged F1 score, and uncertainty from the finalized split] confirms that the three single compounds are readily distinguishable. Figure 3 thus defines the easy analytical regime: when one compound occupies a mapped location, compound-selective Raman bands, multivariate clustering, and a standard random forest model all provide concordant identification.

This result is important for the logic of the subsequent mixture analysis. Failure in a mixture cannot be attributed simply to indistinguishable pure spectra or to an inadequate classifier. The single-component spectra are separable. The difficulty emerges when several compounds compete for the same enhancing surface and their measured contributions no longer combine in direct proportion to the amounts placed in solution.

## Competitive adsorption breaks the direct correspondence between spectral contribution and solution composition

We next extended the platform from single-component recognition to mixtures of diquat, thiabendazole, and thiram. This change converts a classification problem into an inverse problem. For a single analyte, the presence of a characteristic band is sufficient for identification. For a mixture, however, the measured spectrum reflects not only the solution concentrations but also molecular adsorption affinity, Raman response strength, surface coverage, local enhancement, and competition among coexisting molecules. Consequently, the spectrum observed at the surface is not expected to be a linear image of the composition prepared in solution.

The ternary response surfaces in Figure 4a illustrate this distortion. Recovery and bias varied across the composition simplex, and each pure-component vertex exhibited a compound-dependent response. The direction and magnitude of the errors were not uniform across the triangle, indicating that a single global correction factor is insufficient. Concentration-response measurements independently supported this interpretation (Figure 4b). The estimated limits of detection were 1.56 micromolar for diquat, 1.74 micromolar for thiabendazole, and 0.37 micromolar for thiram when calculated from the blank variability and the low-concentration response. Thus, the three analytes entered the mixture problem with substantially different detectabilities, with thiram producing the strongest low-concentration response.

The effect became pronounced in equimolar ternary mixtures (Figure 4c). At 10 micromolar per component, linear spectral decomposition returned an apparent surface composition of approximately 30 percent diquat, 23 percent thiabendazole, and 47 percent thiram. At 33 micromolar per component, the corresponding values were approximately 18, 34, and 48 percent. At 100 and 333 micromolar per component, the reconstructed spectrum was assigned almost entirely to thiram, even though the solution contained equal concentrations of all three compounds. These observations should not be interpreted as a change in the prepared solution composition. Rather, they show that the surface-accessible Raman signal becomes dominated by thiram at high loading. Once the spectral contribution of a minor component approaches zero, no linear unmixing algorithm can recover its original solution concentration from that spectrum alone.

This experiment defines both an analytical range and an information boundary. The low-concentration design spanning 3, 6, 12, and 24 micromolar per component was retained for composition learning because all three compounds remained measurably represented over most of this range. The high-concentration equimolar series was used as a stress test and as an out-of-domain warning. Predictions outside the calibrated concentration range are not interpreted quantitatively, particularly when one component has displaced the spectral evidence of the others.

## Linear unmixing is useful for surface evidence but insufficient for recovering prepared mixture composition

Non-negative least-squares regression was first evaluated because it provides a transparent physical baseline. Each measured spectrum was represented as a non-negative combination of pure-component templates. This approach is valuable for identifying which reference spectra are visibly present at the surface and for excluding negative, chemically meaningless coefficients. It also provides a reconstruction residual that can be inspected directly.

The limitation is that the fitted coefficients describe apparent spectral contributions, not necessarily the concentrations initially mixed in solution. If the response of component (i) is written as

\[
B_i = g A_i \theta_i,
\]

where (B_i) is the fitted spectral coefficient, (g) is the local enhancement factor, (A_i) is the compound-dependent Raman response, and \(\theta_i\) is surface coverage, then the coefficient is already a product of several latent quantities. Under competitive adsorption,

\[
\theta_i = \frac{K_i C_i}{1 + \sum_j K_j C_j},
\]

where (K_i) is an apparent adsorption parameter and (C_i) is the solution concentration. Even this compact expression is an approximation because the measured data show compound-dependent heterogeneity and departures from an ideal single-site adsorption model. A fixed linear combination of pure spectra can therefore estimate what is spectrally visible without uniquely recovering what was prepared in solution.

The benchmark in Figure 4d makes this distinction explicit. Random forest classification, non-negative least-squares regression, one-dimensional convolutional neural-network regression, partial least-squares discriminant analysis, and a multilayer perceptron were compared for binary and ternary mixtures using the same held-out conditions [replace with the final harmonized comparison design]. Error increased for ternary mixtures across all methods, confirming that the third competing component adds a genuine information burden. Non-negative least-squares regression remained an interpretable reference but did not provide the lowest composition error. The learned multilayer perceptron was therefore introduced not as a replacement for physical analysis, but as a nonlinear inverse model that could learn reproducible deviations from linear additivity.

## Physics-guided nonlinear learning restores mixture composition within the calibrated design space

The hybrid model combined three sources of information. First, non-negative least-squares regression was retained as a gate to distinguish analyte-containing spectra from background and to prevent the neural model from creating components in pixels lacking sufficient spectral evidence. Second, calibration measurements were used to generate physically informed synthetic mixtures that reproduced compound-dependent response and adsorption nonlinearity. Third, labeled experimental mixtures were used to fine-tune the relationship between the measured full spectrum and the known solution composition. The final nonlinear mapping was implemented with a multilayer perceptron and a softmax output so that the three predicted composition fractions were non-negative and summed to unity.

The principal architecture consisted of a fully connected layer with 256 hidden units, batch normalization, a rectified linear activation, dropout with a probability of 0.15, a second fully connected layer with 64 hidden units, a second rectified linear activation, and a three-output composition layer. Training used the Adam optimizer with a learning rate of 0.0003 and a weight-decay coefficient of 0.001. The composition loss was a weighted absolute error in which minor components received greater weight than dominant components. Physics-guided pretraining was followed by fine-tuning on measured mixtures. Predictive performance was evaluated by leaving out an entire mixture condition, including all spectra from that condition, rather than by randomly dividing pixels from the same map between training and testing.

Across the low-concentration grid, the model reproduced the broad ternary composition landscape with recoveries of 129 plus or minus 9 percent for thiabendazole, 117 plus or minus 6 percent for thiram, and 120 plus or minus 7 percent for diquat in the Figure 4e summary [verify the component order against the final artwork before submission]. The remaining bias was compound dependent but substantially smaller than the complete collapse observed for linear decomposition at high concentration. Error maps across the 3, 6, 12, and 24 micromolar grid showed that performance was not uniform (Figure 4f). The most difficult regions were associated with low diquat abundance and with combinations in which the stronger responders dominated the measured spectrum. This structured error pattern is chemically informative: it follows the available signal and adsorption imbalance rather than appearing as random model failure.

The present archive contains closely related model versions with mean composition errors between approximately 12 and 14 percent for the 65-condition low-concentration data set. The exact value reported in the final manuscript must be recomputed from the same prediction file used to draw the final Figure 4. This harmonization is essential because earlier 34-condition and 35-condition analyses used different feature aggregation and validation units. The final claim should be restricted to condition-held-out performance on the finalized 64-condition factorial grid plus the designated reference condition.

The key result is therefore not that nonlinear learning can reconstruct a spectrum more accurately than a linear fit. Rather, the learned model maps a distorted surface spectrum back toward the known solution composition within a defined experimental domain. This distinction should be maintained throughout the manuscript. Non-negative least-squares regression reports surface-visible spectral contribution; the multilayer perceptron estimates the prepared composition after learning the reproducible nonlinear transformation imposed by the sensing interface.

## Composition recovery enables semiquantitative concentration screening, but not regulatory-grade quantification

The model was further evaluated for component-wise concentration estimation (Figure 4g,h). Above 6 micromolar per component, median fold errors were approximately 1.67 for diquat, 1.36 for thiabendazole, and 1.33 for thiram. The fractions of predictions falling within a factor of two of the prepared concentration were 65 percent for diquat, 87 percent for thiabendazole, and 89 percent for thiram. Within a factor of three, the corresponding values were 92, 98, and 98 percent. These results support semiquantitative screening rather than formal quantitative analysis.

The reporting threshold of 6 micromolar is independently consistent with the calibration-derived limits of quantification. The estimated limits of quantification were 4.72 micromolar for diquat, 5.26 micromolar for thiabendazole, and 1.13 micromolar for thiram. At 3 micromolar, only 19 percent of diquat estimates and 50 percent of thiabendazole estimates fell within a factor of two, whereas thiram reached 81 percent. The concentration dependence therefore follows the independently measured analytical sensitivity. Predictions below the component-specific reporting limit should be reported as below the quantifiable range rather than as precise concentrations.

This wording is important for an *Advanced Science* submission. A factor-of-two acceptance band is substantially wider than the recovery and precision criteria commonly used for confirmatory residue analysis. The present platform should therefore be framed as a spatially resolved screening and mixture-interpretation method. Its contribution is the recovery of actionable composition information from a nonlinear, competitively adsorbed Raman signal, not the replacement of validated chromatographic quantification.

## Explainable artificial intelligence connects model decisions to chemically meaningful spectral and physical features

To determine whether the nonlinear model relied on chemically plausible information, we incorporated an explainable artificial intelligence analysis rather than treating the network as an opaque predictor. Three complementary tests were designed. Integrated-gradient attribution localized spectral regions that influenced each component output. Band-wise permutation measured the increase in composition error when a Raman-shift interval was disrupted. Targeted ablation removed compound-associated variable-importance bands and measured the resulting change in the corresponding prediction. These analyses were compared with the characteristic bands identified in the single-component experiments and with the compound-dependent adsorption parameters used during physics-guided pretraining.

The intended interpretation is concordance, not causal proof. A credible model should place appreciable attribution near experimentally observed compound bands, should lose performance when those regions are perturbed, and should respond coherently when adsorption-related inputs are altered. Conversely, a model whose attribution is concentrated in featureless baseline regions or acquisition boundaries would raise concern that it learned batch or intensity artifacts.

The current full-data explanatory fit used 20 pixel spectra per map, physics-guided pretraining, an explicit blank class, and non-negative least-squares gating. Because this model was fitted to all available data, it is suitable for visualizing learned associations but not for reporting predictive accuracy. Moreover, the present stored band-permutation changes are very small and the targeted ablation changes range from approximately 1 to 5 percent. These preliminary results do not yet justify a strong claim that the model has discovered a unique adsorption mechanism. Before Figure 4 is finalized, attribution stability should be evaluated across condition-held-out folds and random seeds, and the observed importances should be compared with shuffled-label and shifted-band negative controls.

With those controls in place, Figure 4 can support a restrained but meaningful conclusion: the transition from random forest classification of isolated components to physics-guided nonlinear mixture inference does not abandon chemical interpretability. Instead, it combines spatial spectral evidence, adsorption-aware calibration, and band-level explanation in a single analytical framework. Figure 3 establishes that molecular identity is directly readable in the simple regime. Figure 4 then shows why mixture composition requires a different model class and how that added model complexity can remain anchored to measurable physical and spectroscopic features.

---

# Experimental Section

## Chemicals and materials

Diquat [specify salt form, purity, supplier, and catalogue number], thiabendazole [purity, supplier, and catalogue number], and thiram [purity, supplier, and catalogue number] were used as target analytes. [Specify the metallic nanostructure or nanoparticle formulation], [substrate material], [ink binder or carrier], and all solvents were obtained from [suppliers] and used as received unless otherwise noted. Ultrapure water with a resistivity of [value] megaohm centimetres was used for aqueous preparations. The chemical form used for converting diquat concentration to mass concentration must be stated explicitly because the ionic and salt forms have different molar masses.

## Preparation of surface-enhanced Raman scattering ink

The sensing ink was prepared by combining [mass or volume] of [enhancing material] with [mass or volume] of [binder, dispersant, and solvent] to a final solids content of [value]. The dispersion was mixed by [vortex mixing, sonication, or stirring] for [time] at [temperature]. Analyte-containing inks were prepared by adding diquat, thiabendazole, or thiram to final concentrations of [values]. A chemically matched blank ink containing no analyte was prepared in parallel. All formulations were prepared in [number] independent batches and used within [time] of preparation. If the enhancing material itself constitutes the ink, the term “binder” should be removed and the nanoparticle synthesis, purification, concentration, and particle characterization should be described here.

## Writing and drying of single-component patterns

The letters corresponding to thiram, thiabendazole, and diquat were written on [substrate] using [pen, brush, plotter, stencil, or printing method]. The deposited volume or areal loading was controlled at [value], and the patterns were dried for [time] under [temperature and humidity]. Optical images were acquired using [camera or microscope]. At least [number] independently written patterns were prepared for each compound. The manuscript should distinguish technical repeat maps from independently prepared ink and substrate replicates.

## Preparation of binary and ternary mixtures

Individual stock solutions were prepared at [concentrations] in [solvent] and stored under [conditions]. Working solutions were prepared by serial dilution immediately before analysis. For the low-concentration factorial design, the final concentration of each of the three compounds was independently set to 3, 6, 12, or 24 micromolar, producing 64 ternary concentration combinations. [Describe the additional blank, center, or reference condition that gives the reported total of 65 conditions.] Equal-volume mixing was accounted for when converting stock concentrations to final component concentrations. The high-concentration equimolar stress series contained 10, 33, 100, and 333 micromolar of each component. Binary mixtures used for model comparison were prepared according to Table [Sx]. Each condition was deposited onto [number] independently prepared substrates with [number] maps per substrate.

## Raman spectral acquisition and mapping

Raman measurements were performed using a [manufacturer and model] instrument equipped with a [wavelength]-nanometre excitation laser, a [objective magnification and numerical aperture] objective, and a [grating] grating. The laser power at the sample was [value], the integration time was [value] per spectrum, and [number] accumulations were collected. Raman shifts were calibrated against [standard] before each measurement session. Maps covered [width by height] with a step size of [value], giving [number] spectra per map. The spectral region from 300 to 1800 inverse centimetres was used for model development unless otherwise stated. Instrument sessions, substrate batches, and map acquisition order were randomized or blocked as described in Table [Sx]. Raw data, including saturated and rejected spectra, should be retained and the rejection criteria fixed before final analysis.

## Spectral preprocessing

Each pixel spectrum was processed independently. A baseline was estimated by asymmetric least-squares smoothing and subtracted, after which negative values were clipped to zero. [Insert the asymmetric least-squares smoothness and asymmetry parameters.] Cosmic-ray spikes were removed using [method and threshold] only if this operation was applied to the final data set. For the single-component random forest classifier and pure-template construction, spectra were normalized to unit Euclidean norm so that classification emphasized spectral shape. For mixture composition learning, the logarithm of one plus the baseline-corrected raw intensity was used without unit-norm normalization, thereby retaining intensity information relevant to concentration and surface response. All preprocessing parameters were fixed before condition-held-out evaluation and were applied identically to training and test spectra.

## Compound-selective mapping and exploratory spectral analysis

Compound-selective maps were generated from the baseline-corrected intensity at 1370 inverse centimetres for thiram, 1270 inverse centimetres for thiabendazole, and 1570 inverse centimetres for diquat, using a spectral window of [value] inverse centimetres around each center. Pixel intensities were displayed without condition-specific rescaling [or describe the exact normalization used]. Representative spectra were obtained by averaging pixels within the written regions. Band distributions were summarized at 551, 780, 1011, 1174, 1364, and 1574 inverse centimetres.

Principal component analysis was performed on [state normalized or unnormalized] pixel spectra after the preprocessing described above. The decomposition was fitted using only [training spectra or the combined exploratory data set]. Because this projection was used for visualization rather than predictive evaluation, no classification performance was inferred from cluster appearance alone.

## Random forest classification of single-component spectra

The random forest classifier was trained on pixel spectra from the blank ink, diquat, thiabendazole, and thiram classes. The model contained 300 decision trees, used bootstrap resampling, and estimated the learning curve from out-of-bag predictions. [State the number of candidate features considered at each split and any class weighting.] Generalization was evaluated using [leave-one-batch-out cross-validation, a spatially contiguous holdout, or the final selected design]. Pixels from the same map were never divided randomly across training and test sets for the reported primary performance. The confusion matrix, per-class precision, recall, F1 score, macro-averaged F1 score, and fold-to-fold uncertainty were calculated from pooled held-out predictions.

## Pure-component templates and non-negative least-squares regression

Reference maps for the blank, ink background, diquat, thiabendazole, and thiram were baseline corrected and averaged by class. The resulting templates were normalized to unit Euclidean norm. Each unknown spectrum or map-average spectrum was interpolated to the reference Raman-shift axis when required and represented as a non-negative linear combination of the reference templates using the Lawson–Hanson algorithm as implemented in the SciPy software package. A component was retained as surface-visible evidence when its relative fitted contribution exceeded 0.15 [confirm for the final Figure 4 analysis]. Reconstruction quality was assessed using the coefficient of determination, spectral angle, and residual spectrum. The fitted coefficients were described as apparent surface spectral contributions and were not equated directly with solution concentrations.

## Calibration and adsorption-response modeling

Single-component dilution series spanning 0.1 to 1000 micromolar were measured for each analyte. Response was read from compound-selective marker bands within plus or minus 8 inverse centimetres of the band center and cross-checked by pure-template projection. The concentration-response relation was described using

\[
B(C) = \frac{g A K C}{1 + K C},
\]

where (B) is the measured response, (g A) is an empirical response-amplitude term, (K) is an apparent adsorption parameter, and (C) is concentration. Nonlinear least-squares fitting used non-negative parameter bounds. Where surface heterogeneity was required, a Sips exponent was introduced as specified in Table [Sx]. The current hybrid model used a Sips-type response for diquat and Langmuir-type responses for thiabendazole and thiram; this choice must be synchronized with the final calibration analysis.

The limit of detection was calculated as 3.3 times the between-map standard deviation of seven blank maps divided by the slope in the low-concentration response region. The limit of quantification was calculated as ten times the same standard deviation divided by that slope. The response region was selected using a prespecified local log–log slope criterion and not by maximizing the reported detection performance.

## Physics-guided multilayer perceptron for mixture composition

The composition model received full-spectrum features from analyte-positive pixels. Its network consisted of a 256-unit fully connected layer, batch normalization, a rectified linear activation, dropout with probability 0.15, a 64-unit fully connected layer, a second rectified linear activation, and a three-unit output layer followed by a softmax transformation. The loss was a component-weighted absolute error, with weight (1 + 2(1-y_i)) for component (i), so that small true fractions contributed more strongly than dominant fractions. Optimization used Adam with a learning rate of 0.0003, weight decay of 0.001, full-batch updates, and a fixed training budget of 300 epochs for predictive evaluation. Random seeds were [list seeds].

Before training on measured mixtures, the network was pretrained using synthetic mixtures generated from the single-component calibration curves and the selected adsorption-response functions. Synthetic nuisance variation [state whether enabled in the final analysis] represented measurement noise, baseline variation, spectral shift, and multiplicative enhancement variation. The pretrained model was then fine-tuned using labeled experimental mixtures. No spectra from the held-out condition were used during fine-tuning, parameter selection, preprocessing selection, or epoch selection.

## Validation and error metrics

The primary validation left out one complete concentration condition at a time. All replicate maps and all pixels belonging to the held-out condition were excluded from model fitting. If multiple independently prepared maps were available for one condition, the split was grouped at the preparation level. Model selection and comparison used the same condition set and the same preprocessing representation for non-negative least-squares regression, partial least-squares discriminant analysis, the one-dimensional convolutional neural network, the random forest benchmark, and the multilayer perceptron.

For a three-component composition vector, composition error was defined as one half of the sum of the absolute differences between predicted and true percentage fractions:

\[
E_{\mathrm{composition}} = \frac{1}{2}\sum_{i=1}^{3}\left|\hat{y}_i-y_i\right|.
\]

Concentration error was summarized as the fold error,

\[
E_{\mathrm{fold}} = \max\left(\frac{\hat{C}}{C},\frac{C}{\hat{C}}\right).
\]

The median fold error and the fractions within factors of two and three were reported. Component concentrations below the finalized quantification limit were designated below the quantifiable range. Confidence intervals were obtained by resampling independent maps or preparation batches, not individual pixels, because neighboring pixels are not independent experimental replicates.

## Explainable artificial intelligence analysis

Model interpretation combined integrated gradients, spectral-window permutation, and targeted band ablation. Integrated gradients were calculated from [state baseline spectrum] to each observed spectrum and aggregated as the normalized absolute attribution for each component output. For permutation analysis, 60-inverse-centimetre windows centered from 530 to 2450 inverse centimetres were disrupted across maps, and the change in composition error was recorded. For targeted ablation, spectral regions centered on independently identified variable-importance bands were replaced by [local interpolation, zero, or baseline value], and the relative change in the corresponding component prediction was calculated.

The explanatory model was fitted to the full data set only for visualization and was not used to estimate predictive performance. Final attribution claims must be based on stability across held-out folds and at least [number] random seeds. Shuffled-label models, randomly shifted band masks, and baseline-only regions should be included as negative controls. Attribution overlap with known Raman bands should be assessed against a permutation-derived null distribution rather than by visual inspection alone.

## Statistical analysis and reproducibility

All statistical analyses were performed using Python [version] with NumPy [version], SciPy [version], scikit-learn [version], and PyTorch [version]. Data are reported as [mean plus or minus standard deviation, median and interquartile range, or confidence interval] as specified. The experimental unit was an independently prepared and measured map or substrate, whereas pixels were treated as subsamples. Sample sizes, exclusions, failed acquisitions, and the number of independent preparation batches are reported for every figure. Analysis code, trained model parameters, source spectra, condition labels, and figure-generation tables will be deposited at [repository and persistent identifier] upon publication.

---

# Proposed figure legends

## Figure 3. Spatially resolved identification of single-component surface-enhanced Raman scattering inks

(a) Optical photographs and compound-selective Raman intensity maps of patterns written with thiram, thiabendazole, and diquat inks. Maps were reconstructed at 1370, 1270, and 1570 inverse centimetres, respectively. Scale bars, [value]. (b) Representative mean spectra from the written regions. Asterisks mark characteristic compound bands used for chemical verification. (c) Distributions of normalized intensity at six discriminating Raman-shift regions for blank ink, thiram, thiabendazole, and diquat. Boxes show [definition], whiskers show [definition], and points represent [pixels or independent maps]. (d) Principal component projection of pixel spectra. Ellipses show [confidence or data ellipse definition]. The first and second principal components explain 36.8 and 33.5 percent of the variance. A random forest classifier trained with spatially or batch-separated validation identified the four classes with [accuracy] and a macro-averaged F1 score of [value].

## Figure 4. From linear surface decomposition to physics-guided nonlinear inference of ternary mixture composition

(a) Composition-dependent recovery and bias obtained from [linear decomposition or specified model] across the diquat–thiabendazole–thiram simplex. (b) Single-component concentration-response curves and blank-based limits of detection of 1.56, 1.74, and 0.37 micromolar for diquat, thiabendazole, and thiram, respectively. (c) Apparent surface composition of equimolar ternary mixtures. Increasing concentration causes thiram-dominated spectra despite equal solution concentrations, demonstrating the failure of direct coefficient-to-concentration conversion outside the calibrated range. (d) Condition-held-out composition errors for random forest classification, non-negative least-squares regression, a one-dimensional convolutional neural network, partial least-squares discriminant analysis, and a multilayer perceptron in binary and ternary mixtures. Bars and error bars show [definition]. (e) Recovery and bias across the low-concentration simplex after physics-guided nonlinear learning. (f) Composition-error maps for the complete factorial design spanning 3, 6, 12, and 24 micromolar of each component. (g) Held-out predicted versus prepared concentrations for thiram, thiabendazole, and diquat. Solid lines indicate equality and dashed lines indicate factors of two [and three if present]. (h) Cumulative distribution of fold error. Solid curves include measurements at or above 6 micromolar; dashed curves show all in-design measurements. All predictive panels use condition-held-out predictions; the full-data explanatory model is used only in the designated interpretation panel [if added].

---

# Claims that are safe now and claims that require revision before submission

## Supported by the current materials

- The three single compounds yield distinct written Raman maps and separable full spectra.
- A conventional random forest classifier is sufficient for the single-component task, provided performance is reported with a spatial or batch-level holdout.
- Equimolar solution composition does not produce equimolar surface spectral contribution, especially at high concentration.
- Non-negative least-squares regression is an interpretable measure of surface-visible spectral contribution but is not, by itself, a reliable estimator of prepared solution composition.
- A nonlinear multilayer perceptron trained on labeled mixtures and informed by calibration physics improves composition recovery within the low-concentration design space.
- Concentration performance is appropriately described as semiquantitative screening above a reporting threshold, not as validated quantitative residue analysis.

## Must be finalized or strengthened

- Insert the exact random forest validation design and held-out performance for Figure 3.
- Harmonize the Figure 4 method-comparison sample set, feature unit, and error numbers. Do not mix results from the earlier 34-condition data set with the later 64-condition factorial grid.
- Recompute every Figure 4 summary from the exact table used to draw the final artwork.
- Verify the component order in the recovery annotations of Figure 4e.
- Add independent preparation or substrate replicates wherever the current data contain only pixel-level replication.
- Run fold-wise and seed-wise attribution stability, shuffled-label controls, and shifted-band controls before claiming chemically meaningful explainability.
- Avoid causal statements about adsorption geometry unless supported by direct surface characterization or appropriately qualified literature evidence.
- Report high-concentration failure as an out-of-domain information loss, not as successful recovery by the neural model.

