# Flow Matching — math derivation and reference

## 1. Setup and notation

* Prior (latent) density: p0(x)p\_0(x).
* Data density: p1(x)p\_1(x).
* Choose a **coupling**π(x0,x1)\\pi(x\_0,x\_1) whose marginals are p0p\_0 and p1p\_1:
  ∫π(x0,x1) dx1=p0(x0),∫π(x0,x1) dx0=p1(x1).\\int \\pi(x\_0,x\_1)\\,dx\_1 = p\_0(x\_0), \\qquad \\int \\pi(x\_0,x\_1)\\,dx\_0 = p\_1(x\_1).A common simple choice is the *independent coupling*π(x0,x1)=p0(x0)p1(x1)\\pi(x\_0,x\_1)=p\_0(x\_0)p\_1(x\_1) (sample x0x\_0 and x1x\_1 independently). Other couplings (e.g. optimal transport couplings) are possible and change the behaviour of the learned flow.
* For a sampled pair (x0,x1)∼π(x\_0,x\_1)\\sim\\pi, define a **straight-line path**
  z(t)=(1−t)x0+tx1,t∈[0,1].z(t) = (1-t)x\_0 + t x\_1,\\qquad t\\in[0,1].This gives a random trajectory t↦z(t)t\\mapsto z(t). Let ptp\_t denote the marginal law of z(t)z(t) under π\\pi:

  pt(z)  =  ∫π(x0,x1) δ(z−((1−t)x0+tx1)) dx0 dx1.p\_t(z) \\;=\\; \\int \\pi(x\_0,x\_1)\\, \\delta\\big(z - ((1-t)x\_0+t x\_1)\\big)\\,dx\_0\\,dx\_1.

> **Important correction (common confusion):**
> It is **not generally true** that pt=(1−t)p0+tp1p\_t=(1-t)p\_0 + t p\_1. That linear mixture of densities is different from the pushforward/marginal of the interpolated samples. The correct definition of ptp\_t is the pushforward of π\\pi by the map (x0,x1)↦(1−t)x0+tx1(x\_0,x\_1)\\mapsto (1-t)x\_0+t x\_1, as written above.

## 2. True velocity field along trajectories

Differentiate the path:

ddtz(t)  =  x1−x0.\\frac{d}{dt} z(t) \\;=\\; x\_1 - x\_0.This is the instantaneous velocity along the *sample path*. Define the **pointwise conditional mean velocity**

vt(z)  =  E(x0,x1)∼π[ x1−x0  ∣  z(t)=z ].v\_t(z) \\;=\\; \\mathbb{E}\_{(x\_0,x\_1)\\sim\\pi}\\big[\\,x\_1 - x\_0 \\;\\big|\\; z(t)=z\\,\\big].So vt(z)v\_t(z) is the expected velocity at location zz and time tt.

## 3. Continuity equation satisfied by (pt,vt)(p\_t,v\_t)

Take a smooth test function φ\\varphi. Compute the time derivative of E[φ(z(t))]\\mathbb{E}[\\varphi(z(t))] under π\\pi:

ddtEπ[φ(z(t))]=Eπ[∇φ(z(t))⊤ (x1−x0)].\\frac{d}{dt}\\mathbb{E}\_{\\pi}\\big[\\varphi(z(t))\\big] = \\mathbb{E}\_{\\pi}\\big[ \\nabla\\varphi(z(t))^\\top \\, (x\_1-x\_0) \\big].Condition on z(t)=zz(t)=z:

=∫∇φ(z)⊤ vt(z) pt(z) dz=−∫φ(z) ∇ ⁣⋅(pt(z)vt(z)) dz,= \\int \\nabla\\varphi(z)^\\top \\, v\_t(z)\\, p\_t(z)\\,dz = -\\int \\varphi(z)\\,\\nabla\\!\\cdot\\big(p\_t(z)v\_t(z)\\big)\\,dz,where we integrated by parts in the last step. Comparing with ddt∫φ(z)pt(z) dz=∫φ(z)∂tpt(z) dz\\frac{d}{dt}\\int \\varphi(z)p\_t(z)\\,dz = \\int \\varphi(z)\\partial\_t p\_t(z)\\,dz, we obtain (in distributional form)

  ∂tpt(z)+∇⋅(pt(z) vt(z))  =  0  .  \\boxed{\\;\\partial\_t p\_t(z) + \\nabla\\cdot\\big( p\_t(z)\\,v\_t(z)\\big) \\;=\\; 0\\;. \\;}This is the **continuity (transport) equation** for the family (pt)t∈[0,1](p\_t)\_{t\\in[0,1]} driven by velocity field vtv\_t. It holds for any coupling π\\pi and the associated ptp\_t and vtv\_t.

Boundary conditions: pushing forward π\\pi at t=0,1t=0,1 recovers the marginals:

p0=Law((1−0)x0+0⋅x1)=Law(x0)=p0,p1=Law(x1)=p1.p\_{0} = \\text{Law}((1-0)x\_0+0\\cdot x\_1) = \\text{Law}(x\_0) = p\_0, \\qquad p\_{1} = \\text{Law}(x\_1) = p\_1.Thus (pt,vt)(p\_t,v\_t) is a valid transport from p0p\_0 to p1p\_1.

## 4. Flow-matching objective and why it recovers vtv\_t

Parameterize a time-dependent vector field fθ(z,t)f\_\\theta(z,t). The canonical squared-error flow-matching loss is

L(θ)=Et∼Unif[0,1]  E(x0,x1)∼π[∥fθ(z(t),t)−(x1−x0)∥2],\\mathcal{L}(\\theta) = \\mathbb{E}\_{t\\sim\\text{Unif}[0,1]}\\; \\mathbb{E}\_{(x\_0,x\_1)\\sim\\pi} \\Big[ \\big\\| f\_\\theta(z(t),t) - (x\_1-x\_0) \\big\\|^2 \\Big],where z(t)=(1−t)x0+tx1z(t)=(1-t)x\_0 + t x\_1.

View this as a conditional regression problem: for each (z,t)(z,t), the optimal square-loss predictor (in the L2L^2 sense) is the conditional mean

fθ⋆(z,t)  =  E[x1−x0∣z(t)=z]  =  vt(z).f\_\\theta^\\star(z,t) \\;=\\; \\mathbb{E}\\big[ x\_1 - x\_0 \\mid z(t)=z \\big] \\;=\\; v\_t(z).Hence the global minimizer (or the limit of training under sufficient model capacity and optimization) satisfies

fθ(z,t)≈vt(z).\\boxed{f\_\\theta(z,t) \\approx v\_t(z).}## 5. From learned field to pushing p0p\_0 to p1p\_1

If fθf\_\\theta equals vtv\_t, then by the continuity equation ∂tpt+∇⋅(ptvt)=0\\partial\_t p\_t + \\nabla\\cdot(p\_t v\_t)=0, the distribution ptp\_t along the deterministic ODE flow

dzdt=fθ(z,t),z(0)∼p0\\frac{dz}{dt} = f\_\\theta(z,t), \\qquad z(0)\\sim p\_0has marginals ptp\_t that satisfy the boundary condition p1p\_1 at t=1t=1. Concretely:

* If you sample z(0)∼p0z(0)\\sim p\_0 and integrate the ODE forward to time 11, the law of z(1)z(1) equals p1p\_1 (assuming perfect approximation and well-posed ODE).
* If you integrate backwards (negating the vector field) from a data sample, you map p1p\_1 to p0p\_0.

Thus *minimizing the regression loss recovers the velocity that implements the transport induced by the chosen coupling π\\pi.*

## 6. Practical clarifications & caveats

* **Coupling matters.** The marginal paths (pt)(p\_t) and the conditional velocity vtv\_t depend on the chosen coupling π\\pi. Independent coupling (x0x\_0 and x1x\_1 sampled independently) is the simplest and is commonly used in practice, but other couplings (e.g., nearest-neighbour or optimal transport couplings) alter the learned flow (can reduce variance or enforce more “direct” transports).
* **Identifiability.** The regression objective only pins down fθf\_\\theta almost-everywhere with respect to ptp\_t. Outside the support of ptp\_t the field is underconstrained; numerical ODE solvers may visit those regions and behave unpredictably, so model and solver choices matter.
* **Finite-capacity / optimization error.** In practice fθf\_\\theta is approximated; small errors produce small deviations in the transported law, but accumulation can matter. Use good architectures, weight regularization, and solver tolerances.
* **Variance of the target.** The target vector x1−x0x\_1-x\_0 can have high variance for independent couplings; this can increase training noise. Strategies: conditional couplings, importance weighting by tt, or variance-reduction estimators.
* **Time weighting.** Many implementations weight the loss by a function w(t)w(t) (e.g., to focus learning at early/late times) or sample tt non-uniformly.
* **Numerics / sampling:** After training, sample z0∼p0z\_0\\sim p\_0 and integrate the ODE z˙=fθ(z,t)\\dot z=f\_\\theta(z,t) with a standard solver (RK4, Dormand-Prince, etc.). Because solvers are approximate, use tolerances and stepsizes suitable for your application.

## 7. Short algorithm (training & sampling)

**Training:**

1. Repeat:
   * Sample x0∼p0x\_0\\sim p\_0, x1∼p1x\_1\\sim p\_1 according to coupling π\\pi.
   * Sample t∼t\\sim Uniform[0,1][0,1].
   * Compute z=(1−t)x0+tx1z=(1-t)x\_0 + t x\_1 and target u=x1−x0u=x\_1-x\_0.
   * Compute loss ∥fθ(z,t)−u∥2\\|f\_\\theta(z,t)-u\\|^2 (optionally times a weight w(t)w(t)).
   * Update θ\\theta by gradient descent.

**Sampling (generation):**

1. Sample z(0)∼p0z(0)\\sim p\_0 (e.g., Gaussian).
2. Numerically integrate z˙=fθ(z,t)\\dot z = f\_\\theta(z,t) from t=0t=0 to t=1t=1.
3. Output z(1)z(1) as generated sample.

## 8. Relation to other models

* **CNFs:** minimize (approximate) exact likelihood by integrating z˙=f(z,t)\\dot z=f(z,t) plus tracking log⁡p\\log p via −Tr(∂f/∂z)-\\mathrm{Tr}(\\partial f/\\partial z). This requires Jacobian-trace estimation and has exact likelihoods. Flow matching avoids trace terms by regressing velocities.
* **Diffusion / score-based models:** train a score (gradient log-density) and either sample with SDE or integrate the probability-flow ODE. Flow matching is closer in spirit to the deterministic probability-flow ODE perspective but supervised by simple interpolations instead of noise-conditional scores.

## 9. Short proof sketch (why MSE → continuity equation)

1. The conditional expectation vt(z)=E[x1−x0∣z]v\_t(z)=\\mathbb{E}[x\_1-x\_0\\mid z] minimizes the MSE at each (z,t)(z,t).
2. If fθ→vtf\_\\theta\\to v\_t in L2(pt)L^2(p\_t), then ∂tpt+∇⋅(ptfθ)→0\\partial\_t p\_t + \\nabla\\cdot(p\_t f\_\\theta)\\to 0 in distribution (because it equals ∂tpt+∇⋅(ptvt)=0\\partial\_t p\_t + \\nabla\\cdot(p\_t v\_t)=0).
3. So the ODE flow pushed forward by fθf\_\\theta has marginals close to ptp\_t, in particular mapping p0p\_0 close to p1p\_1 at t=1t=1.

(See section **3** above for the detailed distributional derivation using test functions.)

---

## 10. Quick implementation tips

* Use conditioning on time tt via positional embeddings or concatenation.
* Use architectures that preserve stability (e.g., Lipschitz control, spectral normalization for extremely high-dim problems).
* If training is noisy: increase batch size, or use variance-reduction couplings (pair samples with some structural relation).
* Use a robust ODE solver at sampling time. If speed is critical, small-step fixed-step integrators (RK4) often work well.
